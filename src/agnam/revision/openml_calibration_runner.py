"""R8: exploratory final-model calibration and resource-use audit.

Reuses the frozen OpenML outer split, final split seed, model seed, model
architecture, and retained ISR pairs. Does NOT rerun discovery or modify any
original result. Stores holdout probabilities for reproducible calibration.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import threading
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from agnam.benchmarking.openml_data import audit_loaded_openml_task, load_openml_task
from agnam.benchmarking.openml_protocol import OpenMLBenchmarkProtocol
from agnam.benchmarking.openml_runner import (
    get_openml_execution_spec,
    get_openml_task_spec,
    prepare_final_prediction_data,
)
from agnam.benchmarking.synthetic_runner import SyntheticModelConfig, normalize_pairs
from agnam.models.agnam import AGNAM
from agnam.models.main_effect_nam import MainEffectNAM
from agnam.training.agnam_trainer import (
    AGNAMTrainingConfig,
    predict_agnam_probabilities,
    train_agnam,
)
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
    predict_probabilities,
    train_main_effect_nam,
)
from agnam.utils.reproducibility import get_device, seed_everything

R8_TASK_IDS = (49, 3918, 3021, 167141, 9952, 14952, 14965, 7592)
DEFAULT_MASTER = Path('results/openml/benchmark/master_results.csv')
DEFAULT_OUTPUT = Path('results/revision/openml_calibration_r8/v1')
N_BINS = 10
AUROC_PARITY_TOLERANCE = 1e-6


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def sha256_array(values) -> str:
    arr = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(str(arr.shape).encode('ascii') + str(arr.dtype).encode('ascii') + arr.tobytes()).hexdigest()


def read_retained_pairs(path: Path, *, feature_names: tuple[str, ...], expected: int):
    audit = pd.read_csv(path)
    needed = {'feature_j', 'feature_k', 'isr_retained'}
    if not needed.issubset(audit.columns):
        raise ValueError(f'Missing columns in {path}: {needed - set(audit.columns)}')
    flag = audit['isr_retained'].astype(str).str.strip().str.lower()
    if not flag.isin({'true', 'false'}).all():
        raise ValueError('Invalid boolean flags in pair audit')
    rows = audit.loc[flag.eq('true'), ['feature_j', 'feature_k']]
    if rows.isna().any().any():
        raise ValueError('Missing feature names in retained interactions')
    raw = [(str(row.feature_j), str(row.feature_k)) for row in rows.itertuples(index=False)]
    if len(raw) != expected:
        raise ValueError(f'ISR retained count mismatch: {len(raw)} != {expected}')
    valid = set(map(str, feature_names))
    seen = set()
    for j, k in raw:
        if j not in valid or k not in valid or j == k:
            raise ValueError(f'Unknown or self pair: {(j, k)}')
        key = frozenset((j, k))
        if key in seen:
            raise ValueError(f'Duplicate interaction pair: {(j, k)}')
        seen.add(key)
    return tuple(normalize_pairs(raw))


def calibration_bins(y_true, y_prob, n_bins: int = N_BINS) -> pd.DataFrame:
    """Positive-class equal-width reliability bins; empty bins kept as NaN."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if n_bins < 2 or len(y) != len(p) or not len(y):
        raise ValueError('Invalid calibration inputs')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and in [0,1]')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Labels must be binary 0/1')
    indices = np.minimum(np.floor(p * n_bins).astype(int), n_bins - 1)
    rows = []
    for k in range(n_bins):
        mask = indices == k
        rows.append({
            'bin': k,
            'lower': k / n_bins,
            'upper': (k + 1) / n_bins,
            'n': int(mask.sum()),
            'mean_probability': float(p[mask].mean()) if mask.any() else np.nan,
            'observed_fraction': float(y[mask].mean()) if mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def calibration_metrics(y_true, y_prob, n_bins: int = N_BINS):
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    bins = calibration_bins(y, p, n_bins)
    populated = bins.loc[bins['n'] > 0]
    ece = float((populated['n'] * (populated['mean_probability'] - populated['observed_fraction']).abs()).sum() / len(y))
    return {
        'brier': float(np.mean((p - y) ** 2)),
        'ece_10': ece if n_bins == 10 else float('nan'),
        'auroc': float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float('nan'),
        'auprc': float(average_precision_score(y, p)) if np.unique(y).size == 2 else float('nan'),
        'positive_fraction': float(y.mean()),
        'n_test': int(len(y)),
    }, bins


class _PeakRSS:
    """Optional sampled process-RSS peak (not exact instantaneous hardware peak)."""
    def __init__(self):
        self.peak = float('nan')
        self._stop = threading.Event()
        self._thread = None
        try:
            import psutil
            self._process = psutil.Process()
        except ImportError:
            self._process = None

    def __enter__(self):
        if self._process is not None:
            self.peak = float(self._process.memory_info().rss)
            def sample():
                while not self._stop.wait(0.05):
                    self.peak = max(self.peak, float(self._process.memory_info().rss))
            self._thread = threading.Thread(target=sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join()
        if self._process is not None:
            self.peak = max(self.peak, float(self._process.memory_info().rss))


def _new_model(mode: str, data, pairs, config: SyntheticModelConfig):
    specs = data.train_data.feature_specs
    common = dict(
        feature_specs=specs,
        main_hidden_dim=config.nam_hidden_dim,
        main_depth=config.nam_depth,
        main_dropout=config.nam_dropout,
        categorical_embedding_dim=config.categorical_embedding_dim,
    )
    if mode == 'main':
        return MainEffectNAM(
            feature_specs=specs,
            hidden_dim=config.nam_hidden_dim,
            depth=config.nam_depth,
            dropout=config.nam_dropout,
            categorical_embedding_dim=config.categorical_embedding_dim,
        )
    if mode == 'agnam':
        return AGNAM(
            **common,
            interaction_pairs=pairs,
            interaction_embedding_dim=config.interaction_embedding_dim,
            interaction_hidden_dim=config.interaction_hidden_dim,
            interaction_depth=config.interaction_depth,
            interaction_dropout=config.interaction_dropout,
        )
    raise ValueError(f'Unknown model mode {mode}')


def fit_final_model(mode: str, data, pairs, *, seed: int, config: SyntheticModelConfig, device: torch.device):
    """Exact same final model construction/training options as openml_runner."""
    seed_everything(seed, deterministic=True)
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        initial_allocated = torch.cuda.memory_allocated(device)
    else:
        initial_allocated = float('nan')
    with _PeakRSS() as monitor:
        start = perf_counter()
        model = _new_model(mode, data, pairs, config)
        args = dict(
            learning_rate=config.final_learning_rate,
            weight_decay=config.final_weight_decay,
            batch_size=config.final_batch_size,
            max_epochs=config.final_max_epochs,
            patience=config.final_patience,
            seed=seed,
            deterministic=True,
        )
        if mode == 'main':
            fitted = train_main_effect_nam(
                model=model, train_data=data.train_data, train_y=data.y_train,
                val_data=data.validation_data, val_y=data.y_validation,
                config=NAMTrainingConfig(**args), device=device,
            )
            predictions = predict_probabilities(model, data.test_data, device=device)
        else:
            fitted = train_agnam(
                model=model, train_data=data.train_data, train_y=data.y_train,
                val_data=data.validation_data, val_y=data.y_validation,
                config=AGNAMTrainingConfig(**args), device=device,
            )
            predictions = predict_agnam_probabilities(model, data.test_data, device=device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - start
        gpu_allocated = float(torch.cuda.max_memory_allocated(device)) / (1024**2) if device.type == 'cuda' else float('nan')
        gpu_reserved = float(torch.cuda.max_memory_reserved(device)) / (1024**2) if device.type == 'cuda' else float('nan')
        gpu_delta = float(torch.cuda.max_memory_allocated(device) - initial_allocated) / (1024**2) if device.type == 'cuda' else float('nan')
        del model
    info = {
        'final_fit_and_prediction_seconds': float(elapsed),
        'peak_cuda_allocated_mib': gpu_allocated,
        'peak_cuda_reserved_mib': gpu_reserved,
        'peak_cuda_allocated_above_baseline_mib': gpu_delta,
        'sampled_peak_process_rss_mib': float(monitor.peak) / (1024**2),
        'best_iteration': int(fitted.best_epoch + 1),
    }
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.asarray(predictions, dtype=np.float64).reshape(-1), info


def _record_for_task(master: pd.DataFrame, task_id: int):
    matches = master.loc[master['task_id'].eq(task_id)]
    if len(matches) != 1:
        raise ValueError(f'Expected one original master record for task {task_id}; found {len(matches)}')
    return matches.iloc[0]


def _metadata_matches(root: Path, master: Path, pair_path: Path, task_id: int) -> bool:
    meta_path = root / 'metadata.json'
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['task_id'] != task_id or meta['source_master_sha256'] != sha256_file(master) or meta['source_pair_sha256'] != sha256_file(pair_path):
            return False
        summary = pd.read_csv(root / 'summary.csv')
        predictions = pd.read_csv(root / 'test_probabilities.csv')
        bins = pd.read_csv(root / 'reliability_bins.csv')
        if set(summary['mode']) != {'main', 'agnam'} or len(summary) != 2:
            return False
        if set(predictions.columns) != {'test_row', 'y_true', 'main_prob', 'agnam_prob'}:
            return False
        if len(predictions) != int(meta['n_test']) or len(bins) != 2 * N_BINS:
            return False
        return all(sha256_file(root / fname) == digest for fname, digest in meta['output_sha256'].items())
    except (KeyError, ValueError, OSError):
        return False


def run_r8_task(task_id: int, *, output_root: Path = DEFAULT_OUTPUT, master_path: Path = DEFAULT_MASTER,
                device: torch.device | None = None, force: bool = False):
    if task_id not in R8_TASK_IDS:
        raise ValueError(f'Task {task_id} not in preregistered exploratory R8 set {R8_TASK_IDS}')
    master_path = Path(master_path)
    output_root = Path(output_root)
    pair_path = master_path.parent / f'task_{task_id}_pairs.csv'
    if not master_path.is_file() or not pair_path.is_file():
        raise FileNotFoundError(f'Missing original benchmark sources for task {task_id}')
    task_dir = output_root / f'task_{task_id}'
    if task_dir.exists() and not force:
        if _metadata_matches(task_dir, master_path, pair_path, task_id):
            print(f'SKIP task {task_id}: verified cached R8 outputs')
            return pd.read_csv(task_dir / 'summary.csv')
        raise RuntimeError(f'Existing R8 task directory is unverified: {task_dir}; investigate before using --force')
    proto = OpenMLBenchmarkProtocol()
    model_cfg = SyntheticModelConfig()
    exec_spec = get_openml_execution_spec(task_id, protocol=proto)
    baseline = _record_for_task(pd.read_csv(master_path), task_id)
    loaded = load_openml_task(get_openml_task_spec(task_id), protocol=proto)
    audit = audit_loaded_openml_task(loaded)
    if not audit.audit_pass or len(loaded.X) != int(baseline['n_samples']) or loaded.X.shape[1] != int(baseline['n_features']):
        raise RuntimeError('Frozen OpenML data/split audit mismatch')
    pairs = read_retained_pairs(pair_path, feature_names=tuple(map(str, loaded.X.columns)), expected=int(baseline['n_isr_retained']))
    data = prepare_final_prediction_data(
        loaded, final_split_seed=exec_spec.final_split_seed,
        validation_fraction=proto.final_validation_fraction,
    )
    device = device if device is not None else get_device()
    all_predictions = {}
    all_rows = []
    all_bins = []
    for mode, benchmark_col in (('main', 'main_auroc'), ('agnam', 'agnam_auroc')):
        print(f'R8 task {task_id} {mode}: fitting final predictor on {device}')
        p, resource = fit_final_model(mode, data, pairs, seed=exec_spec.final_model_seed,
                                      config=model_cfg, device=device)
        metrics, bins = calibration_metrics(data.y_test, p)
        reference_auroc = float(baseline[benchmark_col])
        difference = metrics['auroc'] - reference_auroc
        row = {
            'task_id': task_id, 'dataset_name': str(baseline['dataset_name']),
            'mode': mode, 'n_retained_pairs': len(pairs),
            **metrics, **resource,
            'original_auroc': reference_auroc,
            'delta_auroc_vs_original': float(difference),
            'auroc_parity_1e_minus_6': bool(abs(difference) <= AUROC_PARITY_TOLERANCE),
        }
        print(f"  AUROC={metrics['auroc']:.6f} original={reference_auroc:.6f} delta={difference:+.2e} ECE={metrics['ece_10']:.5f} Brier={metrics['brier']:.5f}")
        all_rows.append(row)
        all_predictions[mode] = p
        bins.insert(0, 'mode', mode)
        bins.insert(0, 'task_id', task_id)
        all_bins.append(bins)
    summary = pd.DataFrame(all_rows)
    pred = pd.DataFrame({'test_row': np.arange(len(data.y_test), dtype=int),
                         'y_true': np.asarray(data.y_test, dtype=int),
                         'main_prob': all_predictions['main'],
                         'agnam_prob': all_predictions['agnam']})
    # R8 is exploratory; surface any parity deviation and preserve all evidence.
    task_dir.mkdir(parents=True, exist_ok=True)
    paths = {'summary.csv': summary, 'test_probabilities.csv': pred,
             'reliability_bins.csv': pd.concat(all_bins, ignore_index=True)}
    for fname, frame in paths.items():
        frame.to_csv(task_dir / fname, index=False, float_format='%.17g')
    metadata = {
        'task_id': task_id, 'dataset_name': str(baseline['dataset_name']),
        'source_master_sha256': sha256_file(master_path),
        'source_pair_sha256': sha256_file(pair_path),
        'outer_development_indices_sha256': sha256_array(loaded.train_indices),
        'outer_test_indices_sha256': sha256_array(loaded.test_indices),
        'y_test_sha256': sha256_array(data.y_test),
        'n_test': len(data.y_test),
        'n_retained_pairs': len(pairs),
        'retained_pairs': [list(pair) for pair in pairs],
        'final_split_seed': exec_spec.final_split_seed,
        'final_model_seed': exec_spec.final_model_seed,
        'n_bins': N_BINS,
        'model_config': asdict(model_cfg),
        'device': str(device), 'torch_version': torch.__version__,
        'output_sha256': {fname: sha256_file(task_dir / fname) for fname in paths},
    }
    (task_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f'SAVED {task_dir}  AUROC parity: {int(summary.auroc_parity_1e_minus_6.sum())}/2')
    return summary
