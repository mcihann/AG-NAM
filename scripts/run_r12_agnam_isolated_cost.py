"""R12 isolated FULL AG-NAM path: discovery -> ISR -> fresh final AG-NAM.

This reproduces the *relevant* stages of the canonical synthetic runner
using its own functions/configs. It intentionally does not train the five
other predictive comparison models. It never updates R1-R12 benchmark files.

Default execution is dry-run. Use --execute for fresh model training.
Run from the AG-NAM repository root within its configured Python environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol, build_synthetic_schedule,
    evaluate_pair_set_recovery,
)
from agnam.benchmarking.synthetic_runner import (
    SyntheticModelConfig,
    _build_discovery_configs,
    _build_isr_config,
    _prepare_final_prediction_data,
    _train_interaction_model,
    generate_synthetic_scenario,
    normalize_pairs,
)
from agnam.training.discovery import run_reproducible_interaction_discovery
from agnam.training.isr import evaluate_isr_for_discovery
from agnam.utils.reproducibility import get_device


def hash_indices(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def hash_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def git_info(root):
    result = {}
    for k, args in [('head', ['rev-parse', 'HEAD']), ('branch', ['branch', '--show-current'])]:
        try:
            r = subprocess.run(['git', *args], cwd=root, capture_output=True,
                               text=True, timeout=6)
            result[k] = r.stdout.strip() if r.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            result[k] = None
    return result


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def require_no_overwrite(target):
    if target.exists():
        raise FileExistsError('Existing result protected: '+str(target))


def split_dataset(spec, protocol):
    dataset = generate_synthetic_scenario(spec.scenario, seed=spec.dataset_seed)
    X, y = dataset.X, dataset.y.to_numpy()
    indices = np.arange(len(X))
    dev_idx, test_idx = train_test_split(
        indices, test_size=protocol.outer_test_fraction,
        random_state=spec.outer_split_seed, stratify=y,
    )
    train_local, val_local = train_test_split(
        np.arange(len(dev_idx)), test_size=0.25,
        random_state=spec.final_split_seed, stratify=y[dev_idx],
    )
    split_hashes = {
        'train': hash_indices(dev_idx[train_local]),
        'validation': hash_indices(dev_idx[val_local]),
        'test': hash_indices(test_idx),
    }
    assert not (set(dev_idx[train_local]) & set(dev_idx[val_local]))
    assert not (set(dev_idx) & set(test_idx))
    return (X, y, dev_idx, test_idx, split_hashes, normalize_pairs(dataset.true_interactions))


def run_cost(spec, output, *, repo):
    protocol = SyntheticBenchmarkProtocol()
    config = SyntheticModelConfig()
    device = get_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    created = datetime.now(timezone.utc).isoformat()
    timings = {}
    full_start = perf_counter()
    synchronize(device)

    t = perf_counter()
    X, y, dev_idx, test_idx, split_hashes, true_pairs = split_dataset(spec, protocol)
    X_dev = X.iloc[dev_idx].reset_index(drop=True)
    y_dev = y[dev_idx]
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y[test_idx]
    synchronize(device)
    timings['data_generation_split_seconds'] = perf_counter() - t

    # When SIAN is present, enforce *exact* same raw sample identities.
    sian_summary = repo / 'results' / 'revision' / 'r12_native_sian_pairwise' / spec.scenario / (
        f'realization_{spec.realization_number:02d}') / 'summary.json'
    sian_split_check = 'not_available'
    if sian_summary.is_file():
        sj = json.loads(sian_summary.read_text(encoding='utf-8'))
        expected = sj.get('split_index_sha256', {})
        if split_hashes != expected:
            raise RuntimeError('SIAN and AG-NAM raw train/validation/test indices differ; refuse cost comparison')
        sian_split_check = 'exact_index_sha256_match'

    nam_cfg, prop_cfg = _build_discovery_configs(spec=spec, config=config)
    t = perf_counter()
    discovery = run_reproducible_interaction_discovery(
        X=X_dev, y=y_dev, n_runs=protocol.n_discovery_runs,
        base_seed=spec.discovery_base_seed,
        selection_threshold=protocol.selection_threshold,
        crossfit_n_splits=protocol.residual_crossfit_folds,
        main_early_stop_fraction=0.20,
        nam_hidden_dim=config.nam_hidden_dim,
        nam_depth=config.nam_depth,
        nam_dropout=config.nam_dropout,
        categorical_embedding_dim=config.categorical_embedding_dim,
        proposer_d_model=config.proposer_d_model,
        proposer_n_heads=config.proposer_n_heads,
        proposer_n_layers=config.proposer_n_layers,
        proposer_dropout=config.proposer_dropout,
        nam_config=nam_cfg, proposer_config=prop_cfg,
        device=device,
    )
    synchronize(device)
    timings['five_run_discovery_seconds'] = perf_counter() - t
    selection_pairs = normalize_pairs(
        (p.feature_j, p.feature_k) for p in discovery.selection.accepted_interactions)

    t = perf_counter()
    isr = evaluate_isr_for_discovery(
        X=X_dev, discovery=discovery,
        config=_build_isr_config(protocol=protocol, config=config),
        device=device, verbose=False,
    )
    synchronize(device)
    timings['isr_seconds'] = perf_counter() - t
    kept = normalize_pairs((p.feature_j,p.feature_k) for p in isr.retained_interactions)

    t = perf_counter()
    final_data = _prepare_final_prediction_data(
        X_outer_dev=X_dev, y_outer_dev=y_dev,
        X_test=X_test, y_test=y_test,
        final_split_seed=spec.final_split_seed,
    )
    result, max_decomp_error = _train_interaction_model(
        data=final_data, interaction_pairs=kept,
        seed=spec.final_model_seed, config=config, device=device,
    )
    synchronize(device)
    timings['final_preprocessing_training_test_seconds'] = perf_counter() - t
    timings['entire_full_agnam_path_seconds'] = perf_counter() - full_start

    metrics = evaluate_pair_set_recovery(kept, true_pairs)
    agnam_synthetic_csv = repo / 'results' / 'synthetic' / spec.scenario / (
        f'realization_{spec.realization_number:02d}.csv')
    original_comparison = {'source': 'not_available'}
    if agnam_synthetic_csv.is_file():
        row = pd.read_csv(agnam_synthetic_csv).iloc[0]
        original_comparison = {
            'source': agnam_synthetic_csv.relative_to(repo).as_posix(),
            'source_sha256': hash_file(agnam_synthetic_csv),
            'original_auroc': float(row['agnam_auroc']),
            'isolated_auroc_minus_original': float(result.auroc - row['agnam_auroc']),
            'original_auprc': float(row['agnam_auprc']),
            'isolated_auprc_minus_original': float(result.auprc - row['agnam_auprc']),
        }
    source_hashes = {}
    for rel in ('src/agnam/benchmarking/synthetic_runner.py',
                'src/agnam/benchmarking/synthetic_protocol.py',
                'src/agnam/training/discovery.py',
                'src/agnam/training/isr.py',
                'scripts/run_r12_agnam_isolated_cost.py'):
        src = repo / rel
        if src.is_file():
            source_hashes[rel] = hash_file(src)
    info = {
        'analysis': 'r12_isolated_complete_full_agnam_path_cost',
        'status': 'complete', 'created_utc': created,
        'scenario': spec.scenario, 'realization': spec.realization_number,
        'git': git_info(repo), 'python': sys.version.split()[0],
        'pytorch': torch.__version__, 'device': str(device),
        'peak_torch_cuda_allocated_mib': (
            float(torch.cuda.max_memory_allocated(device)/1048576)
            if device.type == 'cuda' else None),
        'split_seeds': {'data': spec.dataset_seed,'outer':spec.outer_split_seed,
                        'final':spec.final_split_seed,'discovery':spec.discovery_base_seed,
                        'final_model':spec.final_model_seed},
        'split_index_sha256': split_hashes,
        'siancode_raw_split_identity_check': sian_split_check,
        'n_discovery_runs': protocol.n_discovery_runs,
        'selection_threshold': protocol.selection_threshold,
        'isr_threshold': protocol.isr_threshold,
        'selection_pairs': [list(p) for p in selection_pairs],
        'isr_retained_pairs': [list(p) for p in kept],
        'isr_recovery': {k: getattr(metrics,k) for k in ('true_positive','false_positive','false_negative','precision','recall','f1')},
        'full_agnam_test': {'auroc':result.auroc,'auprc':result.auprc,
                            'balanced_accuracy':result.balanced_accuracy,'f1':result.f1,
                            'best_epoch':result.best_epoch,
                            'max_decomposition_error':max_decomp_error},
        'stage_times': timings,
        'original_synthetic_record_comparison': original_comparison,
        'source_sha256': source_hashes,
        'timing_scope': 'Complete Full AG-NAM path: data/split, five-run discovery, ISR, fresh final AG-NAM; no Main/Oracle/No-ISR/Random/Single-Run comparator trainings',
        'limitations': [
            'One fresh realization is only a pilot, not a replicated time estimate.',
            'No end-to-end runtime can be inferred from original six-model canonical benchmark.',
            'Recorded stage wall time is machine, implementation and warmup dependent.',
            'External sampled RSS/GPU memory is required for a comparable resource measurement.',
            'Different model capacities, optimizers and selection budgets limit causal speed claims.',
            'A single-iteration rerun is a reproducibility audit; stochastic/numerical variation may change metrics.',
        ],
    }
    output.write_text(json.dumps(info, indent=2, default=str), encoding='utf-8')
    print('AG-NAM isolated cost completed:', spec.scenario, spec.realization_number)
    print('AUROC:',result.auroc,'AUPRC:',result.auprc,'pairs:',len(kept))
    print('WALL_SECONDS:',round(timings['entire_full_agnam_path_seconds'],3))
    print('SIAN_SPLIT_MATCH:',sian_split_check)
    print('OUTPUT:', output)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=('S1','S2','S3','S4'), default='S1')
    parser.add_argument('--realization', type=int, default=1)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-root', default='results/revision/r12_agnam_fullpath_cost')
    args=parser.parse_args(argv)
    repo=Path.cwd().resolve()
    if not (repo/'src'/'agnam').is_dir():
        parser.error('Run from AG-NAM repo root (src/agnam missing)')
    protocol=SyntheticBenchmarkProtocol()
    spec=next((x for x in build_synthetic_schedule(protocol)
               if x.scenario==args.scenario and x.realization_number==args.realization),None)
    if spec is None:
        parser.error('Unknown scenario/realization')
    dest=Path(args.output_root).resolve()/args.scenario/f'realization_{args.realization:02d}'
    require_no_overwrite(dest)
    print('R12 FULL AG-NAM ONLY:',args.scenario,args.realization)
    print('OUTPUT_DIR:',dest)
    if not args.execute:
        print('DRY-RUN: No training, output, or benchmark changes. Add --execute for fresh cost run.')
        return 0
    dest.mkdir(parents=True)
    try:
        return run_cost(spec,dest/'summary.json',repo=repo)
    except BaseException as e:
        (dest/'failed.json').write_text(json.dumps({'status':'failed','error':repr(e)},indent=2),encoding='utf-8')
        raise

if __name__=='__main__':
    raise SystemExit(main())
