"""R10 post-submission domain-evidence and held-out model-interaction audit.

Distinguish (i) retained pair selection, (ii) non-additivity of the fitted
pair network, and (iii) evidence external to the dataset/model. All computed
contributions and mixed differences are in logit units, never causal effects.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from agnam.benchmarking.openml_data import audit_loaded_openml_task, load_openml_task
from agnam.benchmarking.openml_protocol import OpenMLBenchmarkProtocol
from agnam.benchmarking.openml_runner import (
    get_openml_execution_spec, get_openml_task_spec, prepare_final_prediction_data,
)
from agnam.benchmarking.synthetic_runner import SyntheticModelConfig
from agnam.revision.openml_calibration_runner import (
    AUROC_PARITY_TOLERANCE, DEFAULT_MASTER, _new_model, read_retained_pairs,
    sha256_array, sha256_file,
)
from agnam.training.agnam_trainer import (
    AGNAMTrainingConfig, predict_agnam_probabilities, train_agnam,
)
from agnam.utils.reproducibility import get_device, seed_everything

LOCKED_CASE_TASK_IDS = (146819, 3021, 3)
POSTHOC_CASE_TASK_IDS = (14952,)
R10_TASK_IDS = LOCKED_CASE_TASK_IDS + POSTHOC_CASE_TASK_IDS
DEFAULT_OUTPUT = Path('results/revision/domain_grounded_r10/v1')
MIN_RECTANGLE_CELL = 3
MAX_CATEGORICAL_LEVELS = 6


def task_role(task_id: int) -> str:
    if task_id in LOCKED_CASE_TASK_IDS:
        return 'locked_case_study'
    if task_id in POSTHOC_CASE_TASK_IDS:
        return 'posthoc_supplement'
    raise ValueError(f'Task {task_id} not among R10 cases')


def build_pair_inventory(master_path: Path = DEFAULT_MASTER) -> pd.DataFrame:
    """Frozen pair inventory; external scientific evidence is NOT imputed."""
    master_path = Path(master_path)
    master = pd.read_csv(master_path)
    records = []
    for task_id in R10_TASK_IDS:
        rows = master.loc[master['task_id'].eq(task_id)]
        if len(rows) != 1:
            raise ValueError(f'Master lacks unique task {task_id}')
        record = rows.iloc[0]
        pair_file = master_path.parent / f'task_{task_id}_pairs.csv'
        if not pair_file.exists():
            raise FileNotFoundError(pair_file)
        audit = pd.read_csv(pair_file)
        if not {'feature_j', 'feature_k', 'selection_frequency', 'isr_retained'}.issubset(audit):
            raise ValueError(f'Missing pair-audit fields {pair_file}')
        flags = audit['isr_retained'].astype(str).str.lower()
        if not flags.isin({'true', 'false'}).all():
            raise ValueError(f'Unexpected retention flag {pair_file}')
        retained = audit.loc[flags.eq('true')].copy()
        if len(retained) != int(record['n_isr_retained']):
            raise ValueError(f'Pair-count parity failed for task {task_id}')
        seen = set()
        for pair in retained.itertuples(index=False):
            j, k = str(pair.feature_j), str(pair.feature_k)
            key = frozenset((j, k))
            if j == k or key in seen:
                raise ValueError(f'Duplicate/self pair for task {task_id}: {j,k}')
            seen.add(key)
            records.append(dict(
                task_id=task_id, dataset_name=str(record['dataset_name']),
                feature_type=str(record['feature_type']),
                selection_role=task_role(task_id), feature_j=j, feature_k=k,
                selection_frequency=float(pair.selection_frequency),
                source_pair_file_sha256=sha256_file(pair_file),
                external_evidence_status='not_yet_adjudicated',
                external_evidence_citation='',
                external_evidence_notes='',
            ))
    result = pd.DataFrame.from_records(records)
    if len(result) != 22 or result.loc[result.selection_role.eq('locked_case_study')].shape[0] != 13:
        raise RuntimeError('R10 pair inventory changed unexpectedly')
    return result


def export_pair_inventory(*, master_path: Path = DEFAULT_MASTER,
                          output_root: Path = DEFAULT_OUTPUT) -> Path:
    """Audit-only; no OpenML download or model fit."""
    output_root = Path(output_root)
    path = output_root / 'pair_inventory.csv'
    inventory = build_pair_inventory(master_path)
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        # Avoid overwriting human-edited evidence annotations.
        immutable = ['task_id', 'dataset_name', 'feature_type', 'selection_role',
                     'feature_j', 'feature_k', 'selection_frequency', 'source_pair_file_sha256']
        if (old[immutable].reset_index(drop=True)
                .equals(inventory[immutable].reset_index(drop=True))):
            print(f'R10 INVENTORY EXISTS: {path} (human annotations preserved)')
            return path
        raise RuntimeError(f'Unverified existing R10 pair inventory: {path}')
    output_root.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(path, index=False)
    print(f'R10 PAIR INVENTORY: 13 locked + 9 posthoc = {len(inventory)}; SAVED {path}')
    return path


def _feature_values(transformed, spec) -> tuple[np.ndarray, np.ndarray]:
    """Return encoded values and validity mask for held-out observations."""
    index = spec.transformed_index
    if spec.kind == 'numeric':
        values = np.asarray(transformed.numeric[:, index], dtype=np.float64)
        valid = ((np.asarray(transformed.numeric_missing[:, index]) == 0)
                 & np.isfinite(values))
    elif spec.kind == 'categorical':
        values = np.asarray(transformed.categorical[:, index], dtype=np.int64)
        valid = values >= 2  # exclude missing/unknown category codes
    else:
        raise ValueError(f'Unknown feature type: {spec.kind}')
    return values, valid


def _level_candidates(values: np.ndarray, valid: np.ndarray, kind: str):
    """Return distinct low/high values and observed-group masks."""
    if int(valid.sum()) < 4:
        return []
    if kind == 'numeric':
        low, high = np.quantile(values[valid], [0.25, 0.75])
        if not np.isfinite(low + high) or high <= low + 1e-8:
            return []
        cut = (low + high) / 2
        masks = [valid & (values <= cut), valid & (values > cut)]
        return [(float(low), float(high), masks)]
    unique, counts = np.unique(values[valid], return_counts=True)
    top = unique[np.argsort(-counts, kind='stable')[:MAX_CATEGORICAL_LEVELS]]
    result = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = int(top[i]), int(top[j])
            result.append((a, b, [valid & (values == a), valid & (values == b)]))
    return result


def supported_rectangle(transformed, first_spec, second_spec,
                        *, min_cell: int = MIN_RECTANGLE_CELL) -> dict:
    """Find a supported 2x2 contingency rectangle in held-out data.

    Numeric predictors use quartile representatives and a midpoint group split.
    'Supported' means each of the four broad joint groups contains min_cell
    observations; it does NOT guarantee that every exact numeric corner is
    densely supported or that counterfactual combinations are causal.
    """
    x, x_valid = _feature_values(transformed, first_spec)
    y, y_valid = _feature_values(transformed, second_spec)
    x_candidates = _level_candidates(x, x_valid, first_spec.kind)
    y_candidates = _level_candidates(y, y_valid, second_spec.kind)
    best = None
    for x0, x1, mx in x_candidates:
        for y0, y1, my in y_candidates:
            cells = [int(np.count_nonzero(mx[i] & my[j])) for i in range(2) for j in range(2)]
            quality = (min(cells), sum(cells))
            if best is None or quality > best['quality']:
                best = dict(x0=x0, x1=x1, y0=y0, y1=y1, cells=cells, quality=quality)
    if best is None or min(best['cells']) < min_cell:
        return {'supported': False, 'cell_counts': best['cells'] if best else [0,0,0,0],
                'levels': None, 'min_cell': min_cell}
    return {'supported': True, 'cell_counts': best['cells'],
            'levels': (best['x0'], best['x1'], best['y0'], best['y1']),
            'min_cell': min_cell}


def pair_mixed_difference(model, transformed, pair_index: int, *,
                          min_cell: int = MIN_RECTANGLE_CELL,
                          device: torch.device | None = None) -> dict:
    """Four-corner mixed difference of the *pair network alone* in logits."""
    if device is None:
        device = next(model.parameters()).device
    first_name, second_name = model.interaction_pairs[pair_index]
    lookup = {s.name: s for s in transformed.feature_specs}
    j, k = lookup[first_name], lookup[second_name]
    rect = supported_rectangle(transformed, j, k, min_cell=min_cell)
    if not rect['supported']:
        return dict(rectangle_supported=False, mixed_difference_logit=np.nan,
                    rectangle_cell_counts='|'.join(map(str, rect['cell_counts'])),
                    rectangle_min_cell=min_cell,
                    corner_levels='')
    x0, x1, y0, y1 = rect['levels']
    numeric = np.repeat(transformed.numeric[0:1], 4, axis=0).copy()
    missing = np.repeat(transformed.numeric_missing[0:1], 4, axis=0).copy()
    categorical = np.repeat(transformed.categorical[0:1], 4, axis=0).copy()
    # Row order 00, 01, 10, 11.
    for spec, vals in ((j, [x0,x0,x1,x1]), (k, [y0,y1,y0,y1])):
        if spec.kind == 'numeric':
            numeric[:,spec.transformed_index] = vals
            missing[:,spec.transformed_index] = 0
        else:
            categorical[:,spec.transformed_index] = vals
    model.eval()
    with torch.inference_mode():
        result = model.raw_interaction_contributions(
            numeric=torch.as_tensor(numeric, device=device),
            numeric_missing=torch.as_tensor(missing, device=device),
            categorical=torch.as_tensor(categorical, device=device),
        )[:,pair_index].detach().cpu().numpy().astype(np.float64)
    delta = result[0] - result[1] - result[2] + result[3]
    return dict(rectangle_supported=True, mixed_difference_logit=float(delta),
                rectangle_cell_counts='|'.join(map(str, rect['cell_counts'])),
                rectangle_min_cell=min_cell,
                corner_levels=json.dumps([float(x0),float(x1),float(y0),float(y1)]))


def heldout_contribution_audit(model, transformed, *, device: torch.device,
                               batch_size: int = 512) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, float]:
    """Exhaustive test partition contribution and exact-sum parity audit."""
    if len(transformed) == 0:
        raise ValueError('No test observations')
    model.eval()
    logits = []
    contributions = []
    maximum_error = 0.0
    with torch.inference_mode():
        for begin in range(0, len(transformed), batch_size):
            end = min(begin + batch_size, len(transformed))
            out = model(
                numeric=torch.as_tensor(transformed.numeric[begin:end], device=device),
                numeric_missing=torch.as_tensor(transformed.numeric_missing[begin:end],device=device),
                categorical=torch.as_tensor(transformed.categorical[begin:end],device=device),
            )
            reconstructed = (out.baseline + out.main_contributions.sum(1)
                             + out.interaction_contributions.sum(1))
            maximum_error = max(maximum_error,
                                float(torch.max(torch.abs(out.logits-reconstructed)).item()))
            logits.append(out.logits.cpu().numpy())
            contributions.append(out.interaction_contributions.cpu().numpy())
    all_logits = np.concatenate(logits).reshape(-1)
    all_contrib = np.concatenate(contributions, axis=0)
    if all_contrib.shape != (len(transformed), len(model.interaction_pairs)):
        raise RuntimeError('Unexpected contribution matrix dimensions')
    if maximum_error > 1e-5:
        raise RuntimeError(f'AG-NAM decomposition fails: {maximum_error}')
    probs = torch.sigmoid(torch.as_tensor(all_logits)).numpy().astype(np.float64)
    columns = [f'pair_{i:02d}' for i in range(len(model.interaction_pairs))]
    frame = pd.DataFrame(all_contrib.astype(np.float64), columns=columns)
    frame.insert(0, 'test_row', np.arange(len(transformed)))
    return probs, frame, all_logits, maximum_error


def _task_record(master: pd.DataFrame, task_id: int):
    matches = master.loc[master['task_id'].eq(task_id)]
    if len(matches) != 1:
        raise ValueError(f'Expected one master row for task {task_id}')
    return matches.iloc[0]


def _validate_cached(output_dir: Path, *, master_path: Path, pair_path: Path,
                     task_id: int) -> bool:
    meta_path = output_dir / 'metadata.json'
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if (meta['task_id'] != task_id
            or meta['master_sha256'] != sha256_file(master_path)
            or meta['pair_audit_sha256'] != sha256_file(pair_path)):
            return False
        return all(sha256_file(output_dir / name) == value
                   for name, value in meta['output_sha256'].items())
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def run_r10_task(task_id: int, *, master_path: Path = DEFAULT_MASTER,
                 output_root: Path = DEFAULT_OUTPUT,
                 device: torch.device | None = None) -> pd.DataFrame:
    """Train only a single final AG-NAM predictor on its frozen OpenML split.

    No discovery, no ISR rerun, no hyperparameter selection or main-NAM fit.
    Any original-AUROC mismatch aborts without exporting misleading results.
    """
    role = task_role(task_id)
    master_path = Path(master_path)
    output_root = Path(output_root)
    pair_path = master_path.parent / f'task_{task_id}_pairs.csv'
    if not master_path.exists() or not pair_path.exists():
        raise FileNotFoundError('Frozen benchmark master/pair audit not found')
    target = output_root / f'task_{task_id}'
    if target.exists():
        if _validate_cached(target, master_path=master_path, pair_path=pair_path, task_id=task_id):
            print(f'R10 SKIP verified task {task_id}')
            return pd.read_csv(target / 'pair_model_evidence.csv')
        raise RuntimeError(f'Existing R10 output unverified, investigate before rerunning: {target}')

    master = pd.read_csv(master_path)
    original = _task_record(master, task_id)
    protocol = OpenMLBenchmarkProtocol()
    execution = get_openml_execution_spec(task_id, protocol=protocol)
    loaded = load_openml_task(get_openml_task_spec(task_id), protocol=protocol)
    audit = audit_loaded_openml_task(loaded)
    if (not audit.audit_pass or len(loaded.X) != int(original['n_samples'])
        or loaded.X.shape[1] != int(original['n_features'])):
        raise RuntimeError('Frozen OpenML split/data audit mismatch')
    pairs = read_retained_pairs(pair_path, feature_names=tuple(map(str,loaded.X.columns)),
                                expected=int(original['n_isr_retained']))
    if not pairs:
        raise RuntimeError('R10 case has no retained pairs')
    data = prepare_final_prediction_data(
        loaded, final_split_seed=execution.final_split_seed,
        validation_fraction=protocol.final_validation_fraction,
    )
    device = device if device is not None else get_device()
    config = SyntheticModelConfig()
    seed_everything(execution.final_model_seed, deterministic=True)
    model = _new_model('agnam', data, pairs, config)
    training = train_agnam(
        model=model, train_data=data.train_data, train_y=data.y_train,
        val_data=data.validation_data, val_y=data.y_validation,
        config=AGNAMTrainingConfig(
            learning_rate=config.final_learning_rate,
            weight_decay=config.final_weight_decay,
            batch_size=config.final_batch_size,
            max_epochs=config.final_max_epochs,
            patience=config.final_patience,
            seed=execution.final_model_seed, deterministic=True,
        ), device=device,
    )
    model.eval()
    probability, contrib, logits, max_error = heldout_contribution_audit(
        model, data.test_data, device=device,
    )
    # Compare the standard prediction API against the decomposition's logits.
    standard_prob = np.asarray(predict_agnam_probabilities(model, data.test_data, device=device),
                               dtype=np.float64)
    max_pred_error = float(np.max(np.abs(probability-standard_prob)))
    if max_pred_error > 1e-6:
        raise RuntimeError(f'Prediction API mismatch: {max_pred_error}')
    auroc = float(roc_auc_score(data.y_test, probability))
    original_auroc = float(original['agnam_auroc'])
    if abs(auroc-original_auroc) > AUROC_PARITY_TOLERANCE:
        raise RuntimeError(f'Original predictive AUROC parity failed: R10={auroc}, benchmark={original_auroc}')

    summary_rows = []
    for i, (j, k) in enumerate(model.interaction_pairs):
        values = contrib[f'pair_{i:02d}'].to_numpy(dtype=np.float64)
        diagnostics = pair_mixed_difference(model, data.test_data, i, device=device)
        summary_rows.append(dict(
            task_id=task_id, dataset_name=str(original['dataset_name']), selection_role=role,
            feature_j=j, feature_k=k,
            mean_abs_centered_contribution_logit=float(np.mean(np.abs(values))),
            sd_centered_contribution_logit=float(np.std(values)),
            p95_abs_centered_contribution_logit=float(np.quantile(np.abs(values),0.95)),
            **diagnostics,
        ))
    pair_frame = pd.DataFrame(summary_rows)
    source_pairs = {frozenset((j,k)) for j,k in pairs}
    fitted_pairs = {frozenset((r['feature_j'],r['feature_k'])) for r in summary_rows}
    if len(pair_frame) != len(pairs) or source_pairs != fitted_pairs:
        raise RuntimeError('Pair identity mismatch after model fit')
    predictions = pd.DataFrame({
        'test_row': np.arange(len(data.y_test)), 'y_true': data.y_test,
        'agnam_probability': probability, 'agnam_logit': logits,
    })
    target.mkdir(parents=True, exist_ok=False)
    paths = {
        'pair_model_evidence.csv': pair_frame,
        'test_pair_contributions.csv': contrib,
        'test_predictions.csv': predictions,
    }
    for name, frame in paths.items():
        frame.to_csv(target / name, index=False, float_format='%.17g')
    # Portable state_dict for future re-analysis. Original benchmark remains unchanged.
    checkpoint = target / 'final_agnam_state_dict.pt'
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, checkpoint)
    meta = dict(
        analysis='R10 post-submission diagnostic; no external interaction ground truth',
        task_id=task_id, dataset_name=str(original['dataset_name']), selection_role=role,
        master_sha256=sha256_file(master_path), pair_audit_sha256=sha256_file(pair_path),
        outer_development_indices_sha256=sha256_array(loaded.train_indices),
        outer_test_indices_sha256=sha256_array(loaded.test_indices),
        y_test_sha256=sha256_array(data.y_test),
        original_auroc=original_auroc, r10_auroc=auroc,
        max_decomposition_error_logit=max_error, max_prediction_error=max_pred_error,
        n_test=len(data.y_test), n_retained_pairs=len(pairs),
        retained_pairs=[list(pair) for pair in model.interaction_pairs],
        final_split_seed=execution.final_split_seed,
        final_model_seed=execution.final_model_seed,
        model_config=asdict(config), best_epoch=int(training.best_epoch),
        device=str(device), torch_version=torch.__version__,
        output_sha256={name:sha256_file(target/name)
                       for name in list(paths) + ['final_agnam_state_dict.pt']},
    )
    (target / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False),encoding='utf-8')
    print(f'R10 TASK {task_id} COMPLETE: {len(pairs)} pairs, '
          f'AUROC={auroc:.6f} (original={original_auroc:.6f}), '
          f'decomposition error={max_error:.2e}, '
          f'supported rectangles={int(pair_frame.rectangle_supported.sum())}/{len(pair_frame)}')
    print('R10 SAVED:', target)
    return pair_frame
