"""R7: paired probability vs. signed-deviance residual proposer discovery.

Reviewer-only experiment. Original AG-NAM training and outputs are untouched.
MainEffectNAM residuals are read from audited, split-locked external-ISR caches.
Neither outer-test data nor ground-truth pairs enter proposer training/scoring.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol,
    candidate_count_from_feature_count,
    evaluate_pair_set_recovery,
)
from agnam.benchmarking.synthetic_runner import (
    SyntheticModelConfig, _build_discovery_configs,
)
from agnam.data.preprocessing import TabularPreprocessor
from agnam.interpretation.interaction_metrics import (
    canonical_pair, evaluate_interaction_ranking,
)
from agnam.interpretation.interaction_scoring import (
    RankedInteraction, compute_interaction_scores,
)
from agnam.interpretation.selection_reproducibility import (
    summarize_selection_reproducibility,
)
from agnam.models.attention_proposer import ResidualAttentionProposer
from agnam.revision.external_isr_runner import (
    _discovery_split, _fingerprint, _load_residual_cache,
    _prepare_development, _spec_for,
)
from agnam.training.proposer_trainer import train_residual_attention_proposer
from agnam.utils.reproducibility import get_device, seed_everything

SCHEMA = 'agnam-r7-residual-sensitivity-v1'
MODES = ('probability', 'deviance')
EPSILON = 1e-6
TOL = 1e-6
CACHE_ROOT = Path('results/revision/external_isr_hybrid/core_synthetic')
OUTPUT_ROOT = Path('results/revision/residual_sensitivity/core_synthetic_v1')
SCORE_COLUMNS = ['feature_j', 'feature_k', 'score', 'rank']


def _sha(path: Path) -> str:
    d = sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            d.update(chunk)
    return d.hexdigest()


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def _save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(tmp, index=False, float_format='%.17g')
    os.replace(tmp, path)


def recover_probabilities(y, prob_residual, *, epsilon: float = EPSILON):
    """Reconstruct p=y-r from saved NAM residuals; only safe clipping allowed."""
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    residual = np.asarray(prob_residual, dtype=np.float64).reshape(-1)
    if labels.shape != residual.shape or not np.isin(labels, (0, 1)).all():
        raise ValueError('Invalid labels/residual shapes or nonbinary labels')
    p = labels.astype(np.float64) - residual
    if not np.isfinite(p).all() or ((p < -TOL) | (p > 1 + TOL)).any():
        raise ValueError('Reconstructed probabilities invalid')
    p = np.clip(p, 0, 1)
    return p, np.clip(p, epsilon, 1.0 - epsilon)


def signed_deviance_residual(y, prob_residual, *, epsilon: float = EPSILON):
    """Signed Bernoulli deviance: +/-sqrt(-2 log predictive likelihood)."""
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    _, p = recover_probabilities(labels, prob_residual, epsilon=epsilon)
    # log1p avoids precision loss for negative-class deviance near p=0.
    value = np.where(
        labels == 1,
        np.sqrt(-2.0 * np.log(p)),
        -np.sqrt(-2.0 * np.log1p(-p)),
    )
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite deviance target')
    return value


def targets_for_mode(mode, y_train, y_val, arrays):
    if mode not in MODES:
        raise ValueError(f'Unknown residual mode: {mode}')
    r_train = np.asarray(arrays['train_residuals'], dtype=np.float64)
    r_val = np.asarray(arrays['validation_residuals'], dtype=np.float64)
    if mode == 'probability':
        recover_probabilities(y_train, r_train)
        recover_probabilities(y_val, r_val)
        return r_train.astype(np.float32), r_val.astype(np.float32)
    # Raw, UNSTANDARDIZED signed deviance. Difference in target scale is a
    # declared limitation; not silently normalizing only one experimental arm.
    return (
        signed_deviance_residual(y_train, r_train).astype(np.float32),
        signed_deviance_residual(y_val, r_val).astype(np.float32),
    )


def _score_paths(out_dir: Path, run_index: int, mode: str):
    stem = f'run_{run_index + 1:02d}_{mode}'
    return out_dir / 'runs' / f'{stem}_ranking.csv', out_dir / 'runs' / f'{stem}_manifest.json'


def _ranking_frame(ranking):
    return pd.DataFrame(
        [(x.feature_j, x.feature_k, float(x.score), x.rank) for x in ranking],
        columns=SCORE_COLUMNS,
    )


def _validate_ranking_frame(frame, features):
    if list(frame.columns) != SCORE_COLUMNS:
        raise ValueError('Unexpected ranking columns')
    n = len(features)
    if len(frame) != n * (n - 1) // 2:
        raise ValueError('Incomplete interaction universe')
    if frame.isna().any().any() or not np.isfinite(frame['score'].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite ranking scores')
    if list(frame['rank']) != list(range(1, len(frame) + 1)):
        raise ValueError('Noncontiguous ranks')
    idx = {str(x): i for i, x in enumerate(features)}
    seen = set()
    ranked = []
    for row in frame.itertuples(index=False):
        a, b = str(row.feature_j), str(row.feature_k)
        if a not in idx or b not in idx or idx[a] >= idx[b] or (a, b) in seen:
            raise ValueError('Invalid canonical pair')
        seen.add((a, b))
        ranked.append(RankedInteraction(a, b, idx[a], idx[b], float(row.score), int(row.rank)))
    if seen != set(combinations(features, 2)):
        raise ValueError('Missing interaction pairs')
    expected = sorted(ranked, key=lambda x: (-x.score, x.index_j, x.index_k))
    if tuple((x.feature_j, x.feature_k) for x in ranked) != tuple((x.feature_j, x.feature_k) for x in expected):
        raise ValueError('Ranking not deterministically ordered')
    return tuple(ranked)


def _read_ranking(score_path, manifest_path, *, expected, features):
    if not score_path.exists() or not manifest_path.exists():
        raise RuntimeError(f'Incomplete cached R7 ranking: {score_path}')
    meta = json.loads(manifest_path.read_text(encoding='utf-8'))
    for key, val in expected.items():
        if meta.get(key) != val:
            raise ValueError(f'R7 cache provenance mismatch: {score_path}: {key}')
    if _sha(score_path) != meta.get('ranking_sha256'):
        raise ValueError(f'R7 ranking hash mismatch: {score_path}')
    frame = pd.read_csv(score_path, float_precision='round_trip')
    return _validate_ranking_frame(frame, features), meta


def run_realization(scenario: str, realization: int, *, root=OUTPUT_ROOT, cache_root=CACHE_ROOT, force=False, device=None):
    """Run 2 x 5 matched proposer fits; rerunnable from cached rankings."""
    protocol = SyntheticBenchmarkProtocol()
    model_config = SyntheticModelConfig()
    if protocol.n_discovery_runs != 5 or protocol.n_realizations != 20 or protocol.selection_threshold != 0.60:
        raise RuntimeError('Locked synthetic R7 assumptions changed')
    spec = _spec_for(scenario, realization, protocol)
    X, y, dataset, _ = _prepare_development(spec, protocol)
    features = tuple(str(c) for c in X.columns)
    true_pairs = tuple(sorted({canonical_pair(a, b) for a, b in dataset.true_interactions}))
    k = candidate_count_from_feature_count(len(features), protocol=protocol)
    out_dir = Path(root) / spec.scenario / f'realization_{realization:02d}'
    device = get_device() if device is None else device
    proposer_cfg_base = _build_discovery_configs(spec, model_config)[1]
    rankings = {mode: [] for mode in MODES}
    times = {mode: [] for mode in MODES}
    epochs = {mode: [] for mode in MODES}
    cache_hashes = []

    for run_index in range(protocol.n_discovery_runs):
        run_seed = spec.discovery_base_seed + run_index
        split = _discovery_split(y, run_seed)
        cache_path = Path(cache_root) / spec.scenario / f'realization_{realization:02d}' / 'residual_runs' / f'run_{run_index + 1:02d}.npz'
        arrays, _ = _load_residual_cache(
            cache_path, spec=spec, run_index=run_index, split=split,
            n_development_rows=len(X), expected_feature_names=features,
        )
        cache_sha = _sha(cache_path)
        cache_hashes.append(cache_sha)
        expected_base = {
            'schema': SCHEMA, 'scenario': spec.scenario,
            'realization': realization, 'dataset_seed': spec.dataset_seed,
            'run_index': run_index, 'run_seed': run_seed,
            'split_sha256': _fingerprint(*split),
            'residual_cache_sha256': cache_sha, 'feature_names': list(features),
            'candidate_k': k, 'selection_threshold': protocol.selection_threshold,
            'epsilon': EPSILON,
        }
        pending = []
        for mode in MODES:
            score_path, meta_path = _score_paths(out_dir, run_index, mode)
            expected = {**expected_base, 'mode': mode}
            if not force and (score_path.exists() or meta_path.exists()):
                saved_ranking, meta = _read_ranking(
                    score_path, meta_path, expected=expected, features=features,
                )
                rankings[mode].append(saved_ranking)
                times[mode].append(float(meta['proposer_seconds']))
                epochs[mode].append(int(meta['best_epoch']))
                print(f'CACHED {spec.scenario} R{realization:02d} run {run_index+1}/5 {mode}', flush=True)
            else:
                pending.append(mode)
        if not pending:
            continue
        train_idx, val_idx, score_idx = split
        preprocessor = TabularPreprocessor()
        training_data = preprocessor.fit_transform(X.iloc[train_idx].reset_index(drop=True))
        validation_data = preprocessor.transform(X.iloc[val_idx].reset_index(drop=True))
        scoring_data = preprocessor.transform(X.iloc[score_idx].reset_index(drop=True))
        if tuple(x.name for x in training_data.feature_specs) != features:
            # If the preprocessing feature order ever differs, it must be
            # intentionally reconciled rather than silently relabeled.
            raise ValueError('Proposer feature order differs from input X')
        for mode in pending:
            target_train, target_val = targets_for_mode(mode, y[train_idx], y[val_idx], arrays)
            cfg = replace(proposer_cfg_base, seed=run_seed)
            seed_everything(run_seed, deterministic=True)
            model = ResidualAttentionProposer(
                feature_specs=training_data.feature_specs,
                d_model=model_config.proposer_d_model,
                n_heads=model_config.proposer_n_heads,
                n_layers=model_config.proposer_n_layers,
                dropout=model_config.proposer_dropout,
            )
            t0 = perf_counter()
            fit = train_residual_attention_proposer(
                model=model, train_data=training_data, train_residuals=target_train,
                val_data=validation_data, val_residuals=target_val,
                config=cfg, device=device,
            )
            elapsed = perf_counter() - t0
            ranking = compute_interaction_scores(
                model=model, transformed=scoring_data,
                batch_size=max(cfg.batch_size, 256), device=device,
            ).ranked_interactions
            frame = _ranking_frame(ranking)
            _validate_ranking_frame(frame, features)
            score_path, meta_path = _score_paths(out_dir, run_index, mode)
            _save_csv(score_path, frame)
            meta = {
                **expected_base, 'mode': mode,
                'proposer_seconds': float(elapsed),
                'best_epoch': int(fit.best_epoch),
                'best_validation_mse': float(fit.best_validation_mse),
                'ranking_sha256': _sha(score_path),
                'target_train_std': float(np.std(target_train, ddof=0)),
                'target_val_std': float(np.std(target_val, ddof=0)),
                'target_train_abs_mean': float(np.abs(target_train).mean()),
            }
            _save_json(meta_path, meta)
            checked, _ = _read_ranking(score_path, meta_path, expected={**expected_base, 'mode': mode}, features=features)
            rankings[mode].append(checked)
            times[mode].append(float(elapsed))
            epochs[mode].append(int(fit.best_epoch))
            print(f'DONE {spec.scenario} R{realization:02d} run {run_index+1}/5 {mode} [{elapsed:.2f}s]', flush=True)

    if any(len(rankings[mode]) != protocol.n_discovery_runs for mode in MODES):
        raise RuntimeError('R7 did not complete both modes and five runs')
    run_rows, summary_rows, stable_sets = [], [], {}
    for mode in MODES:
        per_run = []
        for run_index, ranking in enumerate(rankings[mode]):
            m = evaluate_interaction_ranking(ranking, true_pairs, k=k)
            row = {
                'scenario': spec.scenario, 'realization': realization,
                'mode': mode, 'run_index': run_index,
                'run_seed': spec.discovery_base_seed + run_index,
                'interaction_auprc': float(m.interaction_auprc),
                'precision_at_k': float(m.precision_at_k),
                'recall_at_k': float(m.recall_at_k),
                'f1_at_k': float(2*m.precision_at_k*m.recall_at_k/(m.precision_at_k+m.recall_at_k)) if m.precision_at_k+m.recall_at_k > 0 else 0.0,
                'proposer_seconds': float(times[mode][run_index]),
                'best_epoch': int(epochs[mode][run_index]),
            }
            run_rows.append(row)
            per_run.append(row)
        stable = summarize_selection_reproducibility(
            tuple(rankings[mode]), k=k, threshold=protocol.selection_threshold,
        )
        pairs = tuple(canonical_pair(x.feature_j, x.feature_k) for x in stable.accepted_interactions)
        stable_sets[mode] = set(pairs)
        recovery = evaluate_pair_set_recovery(pairs, true_pairs)
        summary_rows.append({
            'scenario': spec.scenario, 'realization': realization,
            'dataset_seed': spec.dataset_seed, 'mode': mode,
            'n_discovery_runs': protocol.n_discovery_runs,
            'candidate_k': k, 'n_true': len(true_pairs),
            'selection_threshold': protocol.selection_threshold,
            'interaction_auprc_mean': float(np.mean([x['interaction_auprc'] for x in per_run])),
            'interaction_auprc_std': float(np.std([x['interaction_auprc'] for x in per_run],ddof=1)),
            'mean_pairwise_jaccard': float(stable.mean_pairwise_jaccard),
            'selection_count': len(pairs), 'selection_tp': int(recovery.true_positive),
            'selection_fp': int(recovery.false_positive), 'selection_fn': int(recovery.false_negative),
            'selection_precision': float(recovery.precision),
            'selection_recall': float(recovery.recall), 'selection_f1': float(recovery.f1),
            'proposer_seconds_sum': float(sum(times[mode])),
        })
    a, b = stable_sets[MODES[0]], stable_sets[MODES[1]]
    union = len(a | b)
    jaccard = 1.0 if union == 0 else len(a & b) / union
    for row in summary_rows:
        row['cross_mode_selection_jaccard'] = float(jaccard)
        row['cross_mode_same_selection'] = a == b

    _save_csv(out_dir/'run_metrics.csv', pd.DataFrame(run_rows))
    _save_csv(out_dir/'summary.csv', pd.DataFrame(summary_rows))
    _save_json(out_dir/'realization_manifest.json', {
        'schema': SCHEMA, 'scenario': spec.scenario, 'realization': realization,
        'n_runs': protocol.n_discovery_runs, 'modes': list(MODES),
        'residual_cache_sha256_by_run': cache_hashes,
        'run_metrics_sha256': _sha(out_dir/'run_metrics.csv'),
        'summary_sha256': _sha(out_dir/'summary.csv'),
    })
    print(pd.DataFrame(summary_rows)[['scenario','realization','mode','interaction_auprc_mean','selection_f1','selection_count','proposer_seconds_sum']].to_string(index=False),flush=True)
    print('SAVED',out_dir,flush=True)
    return pd.DataFrame(summary_rows)
