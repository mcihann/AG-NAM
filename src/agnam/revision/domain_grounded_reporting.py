"""Independent read-only R10 result auditor and reporting helper.

This does not train models, select interactions, or adjudicate literature.
It validates frozen outputs and preserves explicit uncertainty in domain evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TASK_IDS = (146819, 3021, 3, 14952)
LOCKED_IDS = frozenset((146819, 3021, 3))
SOURCE_STATUSES = frozenset((
    "published_surrogate_term_identified",
    "pairwise_evidence_not_established_in_screened_sources",
))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def pair_key(first: object, second: object) -> str:
    a, b = str(first), str(second)
    if not a or not b or a == b:
        raise ValueError(f'Invalid pair: {a!r}, {b!r}')
    return '|'.join(sorted((a, b)))


def add_keys(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not {'task_id', 'feature_j', 'feature_k'}.issubset(frame):
        raise ValueError(f'{name} is missing task_id/feature_j/feature_k')
    result = frame.copy()
    result['task_id'] = result['task_id'].astype(int)
    result['pair_key'] = [pair_key(a, b) for a, b in
                          zip(result.feature_j, result.feature_k)]
    if result.duplicated(['task_id', 'pair_key']).any():
        raise ValueError(f'{name} contains duplicate feature pairs')
    return result


def validate_evidence_classes(screen: pd.DataFrame) -> None:
    if len(screen) != 22:
        raise ValueError(f'Expected 22 source-screened pairs, got {len(screen)}')
    statuses = screen['source_review_status'].astype(str)
    if not set(statuses).issubset(SOURCE_STATUSES):
        raise ValueError('Unknown source review status')
    positive = screen.loc[statuses.eq('published_surrogate_term_identified')]
    if len(positive) != 1 or int(positive.iloc[0].task_id) != 146819 or pair_key(
        positive.iloc[0].feature_j, positive.iloc[0].feature_k
    ) != 'vconst_2|vconst_corr':
        raise ValueError('The documented literature screening pair changed')
    if screen['source_url'].isna().any() or screen['source_url'].astype(str).str.strip().eq('').any():
        raise ValueError('Missing source URL')
    # Source classes are *screening outcomes*, not truth labels.
    nonpositive = statuses.eq('pairwise_evidence_not_established_in_screened_sources')
    if int(nonpositive.sum()) != 21:
        raise ValueError('Unexpected number of unestablished pairwise claims')


def reconcile_screen_model_fields(
    screen: pd.DataFrame,
    aggregate: pd.DataFrame,
    *,
    model_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Verify prejoined model values, then remove duplicate model fields.

    Source screening is about domain evidence. The separately audited
    aggregate is authoritative for model measurements. Screening drafts
    are allowed to carry earlier joined measurement columns, but may not
    silently contradict them.
    """
    repeated = [field for field in model_columns if field in screen.columns]
    if not repeated:
        return screen.copy()

    keys = ['task_id', 'pair_key']
    matched = screen[keys + repeated].merge(
        aggregate[keys + repeated],
        on=keys,
        how='outer',
        validate='one_to_one',
        indicator=True,
        suffixes=('_screen', '_model'),
    )
    if not matched['_merge'].eq('both').all():
        raise ValueError('Source-screened model pairs differ from frozen aggregate')

    for field in repeated:
        from_screen = matched[f'{field}_screen']
        from_model = matched[f'{field}_model']
        if field in ('rectangle_supported',):
            equal = from_screen.astype(str).str.lower().equals(
                from_model.astype(str).str.lower()
            )
            if not equal:
                raise ValueError(f'Source-screened model evidence mismatch: {field}')
        elif field in ('rectangle_cell_counts',):
            if not from_screen.fillna('').astype(str).equals(
                from_model.fillna('').astype(str)
            ):
                raise ValueError(f'Source-screened model evidence mismatch: {field}')
        else:
            try:
                _assert_numeric_close(
                    from_screen, from_model,
                    f'source-screened model evidence: {field}',
                )
            except ValueError as exc:
                raise ValueError(
                    f'Source-screened model evidence mismatch: {field}'
                ) from exc
    return screen.drop(columns=repeated).copy()


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _assert_numeric_close(left: pd.Series, right: pd.Series, label: str,
                          tolerance: float = 1e-11) -> None:
    a = pd.to_numeric(left, errors='coerce').to_numpy(dtype=np.float64)
    b = pd.to_numeric(right, errors='coerce').to_numpy(dtype=np.float64)
    if not np.allclose(a, b, rtol=0, atol=tolerance, equal_nan=True):
        raise ValueError(f'Numeric data changed: {label}')


def audit_r10(*, root: str | Path = 'results/revision/domain_grounded_r10/v1',
              master: str | Path = 'results/openml/benchmark/master_results.csv',
              write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root, master = Path(root), Path(master)
    inventory_path = root / 'pair_inventory.csv'
    screened_path = root / 'domain_evidence_source_screening_draft.csv'
    aggregate_path = root / 'all_pair_model_evidence.csv'
    inventory = add_keys(_read(inventory_path), name='pair inventory')
    screen = add_keys(_read(screened_path), name='source screening')
    aggregate = add_keys(_read(aggregate_path), name='model evidence aggregate')
    if len(inventory) != 22 or len(screen) != 22 or len(aggregate) != 22:
        raise ValueError('R10 requires exactly 22 unique pairs per source')
    if not (set(zip(inventory.task_id, inventory.pair_key))
            == set(zip(screen.task_id, screen.pair_key))
            == set(zip(aggregate.task_id, aggregate.pair_key))):
        raise ValueError('Pair identities differ among independent R10 sources')
    validate_evidence_classes(screen)
    if set(inventory.task_id) != set(TASK_IDS):
        raise ValueError('Task roster changed')
    locked = inventory.selection_role.astype(str).eq('locked_case_study')
    if int(locked.sum()) != 13 or int((~locked).sum()) != 9:
        raise ValueError('Locked/posthoc case roles changed')
    if not (inventory.loc[locked, 'task_id'].isin(LOCKED_IDS).all()
            and inventory.loc[~locked, 'task_id'].eq(14952).all()):
        raise ValueError('Locked/posthoc case mapping changed')
    master_hash = sha256(master)
    metadata = {}
    local_frames = []
    for task_id in TASK_IDS:
        task_dir = root / f'task_{task_id}'
        meta_path = task_dir / 'metadata.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        pair_file = master.parent / f'task_{task_id}_pairs.csv'
        if int(meta['task_id']) != task_id or meta['master_sha256'] != master_hash:
            raise ValueError(f'Original benchmark provenance mismatch: {task_id}')
        if meta['pair_audit_sha256'] != sha256(pair_file):
            raise ValueError(f'Original pair audit changed: {task_id}')
        for filename, expected in meta['output_sha256'].items():
            if sha256(task_dir / filename) != expected:
                raise ValueError(f'Invalid R10 file hash: {task_id}/{filename}')
        a = float(meta['r10_auroc'])
        b = float(meta['original_auroc'])
        if abs(a - b) > 1e-6:
            raise ValueError(f'Original AUROC parity mismatch: {task_id}')
        if float(meta['max_decomposition_error_logit']) > 1e-5:
            raise ValueError(f'Decomposition parity mismatch: {task_id}')
        if float(meta['max_prediction_error']) > 1e-6:
            raise ValueError(f'Probability parity mismatch: {task_id}')
        preds = _read(task_dir / 'test_predictions.csv')
        measured = float(roc_auc_score(preds['y_true'], preds['agnam_probability']))
        if abs(measured - a) > 1e-6 or len(preds) != int(meta['n_test']):
            raise ValueError(f'Predictions do not reproduce AUROC: {task_id}')
        local = add_keys(_read(task_dir / 'pair_model_evidence.csv'),
                         name=f'task {task_id} model evidence')
        if len(local) != int(meta['n_retained_pairs']):
            raise ValueError(f'Pair count mismatch: {task_id}')
        if set(local.task_id) != {task_id}:
            raise ValueError(f'Wrong task in local evidence: {task_id}')
        meta_pairs = {pair_key(*p) for p in meta['retained_pairs']}
        if meta_pairs != set(local.pair_key):
            raise ValueError(f'Model/pair identity mismatch: {task_id}')
        local_frames.append(local)
        metadata[task_id] = {
            'original_auroc': b, 'r10_auroc': a,
            'n_test': int(meta['n_test']),
            'max_decomposition_error_logit': float(meta['max_decomposition_error_logit']),
        }
    local = pd.concat(local_frames, ignore_index=True)
    combined = aggregate.merge(local, on=['task_id', 'pair_key'], how='outer',
                               validate='one_to_one', indicator=True, suffixes=('', '_local'))
    if not combined['_merge'].eq('both').all():
        raise ValueError('Aggregate/local model evidence differs')
    for field in ('mean_abs_centered_contribution_logit',
                  'sd_centered_contribution_logit',
                  'mixed_difference_logit'):
        _assert_numeric_close(combined[field], combined[f'{field}_local'], field)
    local_support = local['rectangle_supported'].astype(str).str.lower().eq('true')
    agg_support = aggregate['rectangle_supported'].astype(str).str.lower().eq('true')
    if int(local_support.sum()) != 20 or int(agg_support.sum()) != 20:
        raise ValueError('Expected 20/22 supported rectangles')
    if int(aggregate['mixed_difference_logit'].notna().sum()) != 20:
        raise ValueError('Unsupported rectangles must have missing values')
    if (aggregate.loc[~agg_support, 'mixed_difference_logit'].notna().any()
            or aggregate.loc[agg_support, 'mixed_difference_logit'].isna().any()):
        raise ValueError('Rectangle support and mixed difference inconsistent')
    cross_check = inventory.merge(
        screen[['task_id', 'pair_key', 'dataset_name', 'feature_type',
                'selection_role', 'selection_frequency', 'source_pair_file_sha256']],
        on=['task_id', 'pair_key'], how='outer', validate='one_to_one',
        indicator=True, suffixes=('_inventory', '_screen'))
    if not cross_check['_merge'].eq('both').all():
        raise ValueError('Source review does not cover all inventory pairs')
    for field in ('dataset_name', 'feature_type', 'selection_role',
                  'selection_frequency', 'source_pair_file_sha256'):
        left = cross_check[f'{field}_inventory']
        right = cross_check[f'{field}_screen']
        if field == 'selection_frequency':
            _assert_numeric_close(left, right, field)
        elif not left.astype(str).equals(right.astype(str)):
            raise ValueError(f'Source review modified frozen field: {field}')
    model_columns = (
        'mean_abs_centered_contribution_logit',
        'sd_centered_contribution_logit',
        'p95_abs_centered_contribution_logit',
        'rectangle_supported',
        'mixed_difference_logit',
        'rectangle_cell_counts',
    )
    # An earlier, valid source-screening draft may already include model
    # measurements from a manually assembled review matrix. Verify those
    # repeated columns against frozen, hash-checked model evidence, then
    # use the model source as the sole authoritative copy. Without this,
    # pandas appends _x/_y suffixes and the final audit crashes.
    screen = reconcile_screen_model_fields(
        screen, aggregate, model_columns=model_columns,
    )
    evidence = screen.merge(
        aggregate[['task_id', 'pair_key', *model_columns]],
        on=['task_id', 'pair_key'], how='inner', validate='one_to_one')
    evidence = evidence.sort_values(['task_id', 'pair_key'], kind='stable').reset_index(drop=True)
    evidence['abs_mixed_difference_logit'] = evidence.mixed_difference_logit.abs()
    summary_rows = []
    for task_id in TASK_IDS:
        frame = evidence[evidence.task_id.eq(task_id)]
        summary_rows.append(dict(
            task_id=task_id, dataset_name=frame.iloc[0].dataset_name,
            selection_role=frame.iloc[0].selection_role,
            n_pairs=len(frame), n_supported=int(frame.rectangle_supported.astype(str).str.lower().eq('true').sum()),
            mean_abs_centered_contribution_logit=float(frame.mean_abs_centered_contribution_logit.mean()),
            median_abs_mixed_difference_logit=float(frame.abs_mixed_difference_logit.median()),
            max_abs_mixed_difference_logit=float(frame.abs_mixed_difference_logit.max()),
            original_auroc=metadata[task_id]['original_auroc'],
            r10_auroc=metadata[task_id]['r10_auroc'],
        ))
    summary = pd.DataFrame(summary_rows)
    audit = dict(
        label='reviewer_requested_post_submission_R10_exploratory',
        original_benchmark_sha256=master_hash,
        pair_inventory_sha256=sha256(inventory_path),
        source_screening_sha256=sha256(screened_path),
        evidence_aggregate_sha256=sha256(aggregate_path),
        verified_tasks=4, verified_pairs=len(evidence),
        rectangle_supported=int(agg_support.sum()),
        published_surrogate_cross_term_count=1,
        pairwise_evidence_unestablished_count=21,
        max_original_auroc_delta=max(abs(v['r10_auroc']-v['original_auroc']) for v in metadata.values()),
        no_external_interaction_ground_truth=True,
        training_executed_by_analysis=False,
    )
    if write:
        out = root / 'analysis'
        out.mkdir(parents=True, exist_ok=True)
        p_evidence = out / 'domain_evidence_matrix.csv'
        p_summary = out / 'case_summary.csv'
        evidence.to_csv(p_evidence, index=False, float_format='%.17g')
        summary.to_csv(p_summary, index=False, float_format='%.17g')
        audit['output_sha256'] = {p.name: sha256(p) for p in (p_evidence, p_summary)}
        (out / 'analysis_manifest.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    return evidence, summary, audit
