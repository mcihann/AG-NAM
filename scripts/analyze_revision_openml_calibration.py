"""Validate completed R8 outputs, export summary tables/reliability figures.

No training or OpenML access. An eight-task post-hoc exploratory sample; no
confirmatory generalization or formal superiority claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agnam.revision.openml_calibration_runner import (
    DEFAULT_MASTER, DEFAULT_OUTPUT, N_BINS, R8_TASK_IDS,
    calibration_metrics, sha256_file,
)


def save_reliability_plot(bins: pd.DataFrame, path: Path, task_id: int):
    """Save an equal-width reliability diagram from audited bin summaries."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, color='gray', label='Perfect calibration')
    for mode, marker in (('main', 'o'), ('agnam', 's')):
        b = bins.loc[(bins['mode'] == mode) & (bins['n'] > 0)].sort_values('bin')
        ax.plot(b['mean_probability'], b['observed_fraction'], marker=marker,
                linewidth=1.5, markersize=5, label='Main NAM' if mode == 'main' else 'Full AG-NAM')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability',
           ylabel='Observed positive-class frequency', title=f'OpenML task {task_id} | R8 calibration')
    ax.grid(alpha=.2)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(root: Path = DEFAULT_OUTPUT, *, allow_partial: bool = False):
    root = Path(root)
    summaries, bins_all = [], []
    manifest = {'expected_task_ids': list(R8_TASK_IDS), 'tasks': {}}
    missing = []
    for task_id in R8_TASK_IDS:
        d = root / f'task_{task_id}'
        needed = ('summary.csv', 'test_probabilities.csv', 'reliability_bins.csv', 'metadata.json')
        if not all((d / fn).is_file() for fn in needed):
            missing.append(task_id)
            continue
        meta = json.loads((d / 'metadata.json').read_text(encoding='utf-8'))
        if meta['task_id'] != task_id or meta['n_bins'] != N_BINS:
            raise RuntimeError(f'Invalid metadata for task {task_id}')
        source_master = Path(DEFAULT_MASTER)
        pair_path = source_master.parent / f'task_{task_id}_pairs.csv'
        if meta['source_master_sha256'] != sha256_file(source_master) or meta['source_pair_sha256'] != sha256_file(pair_path):
            raise RuntimeError(f'Original input source hashes differ for task {task_id}')
        for fn, digest in meta['output_sha256'].items():
            if sha256_file(d / fn) != digest:
                raise RuntimeError(f'Output SHA-256 mismatch: {d / fn}')
        summary = pd.read_csv(d / 'summary.csv')
        probs = pd.read_csv(d / 'test_probabilities.csv')
        bins = pd.read_csv(d / 'reliability_bins.csv')
        if set(summary['mode']) != {'main', 'agnam'} or len(summary) != 2 or len(probs) != meta['n_test'] or len(bins) != 2*N_BINS:
            raise RuntimeError(f'Missing/invalid R8 records in task {task_id}')
        if sorted(probs['test_row'].astype(int)) != list(range(len(probs))):
            raise RuntimeError(f'Test order mismatch task {task_id}')
        for mode in ('main', 'agnam'):
            rec = summary.loc[summary['mode'].eq(mode)].iloc[0]
            calc, cbins = calibration_metrics(probs['y_true'], probs[f'{mode}_prob'])
            for metric in ('auroc', 'auprc', 'brier', 'ece_10'):
                if not np.isclose(calc[metric], float(rec[metric]), rtol=0, atol=1e-10, equal_nan=True):
                    raise RuntimeError(f'Metric {metric} mismatch for {task_id}/{mode}')
            obs = bins.loc[bins['mode'].eq(mode)].sort_values('bin').reset_index(drop=True)
            if len(obs) != N_BINS or not np.array_equal(obs['n'].astype(int), cbins['n'].astype(int)):
                raise RuntimeError(f'Calibration bin counts mismatch task {task_id}/{mode}')
        summaries.append(summary)
        bins_all.append(bins)
        manifest['tasks'][str(task_id)] = {'metadata_sha256': sha256_file(d/'metadata.json'),
                                          'output_sha256': meta['output_sha256']}
    if missing and not allow_partial:
        raise RuntimeError(f'R8 incomplete; missing tasks {missing}. Run --all first.')
    if not summaries:
        raise RuntimeError('No completed R8 tasks found')
    combined = pd.concat(summaries, ignore_index=True)
    by_task = combined.pivot(index=['task_id','dataset_name','n_retained_pairs'], columns='mode',
                             values=['auroc','brier','ece_10','final_fit_and_prediction_seconds','peak_cuda_allocated_mib','sampled_peak_process_rss_mib'])
    by_task.columns = [f'{metric}_{mode}' for metric, mode in by_task.columns]
    by_task = by_task.reset_index()
    for metric in ('auroc','brier','ece_10'):
        by_task[f'delta_agnam_minus_main_{metric}'] = by_task[f'{metric}_agnam'] - by_task[f'{metric}_main']
    descriptives = combined.groupby('mode', as_index=False).agg(
        n_tasks=('task_id','size'),
        auroc_mean=('auroc','mean'),
        brier_mean=('brier','mean'),
        ece_10_mean=('ece_10','mean'),
        training_and_prediction_seconds_mean=('final_fit_and_prediction_seconds','mean'),
        peak_cuda_allocated_mib_mean=('peak_cuda_allocated_mib','mean'),
    )
    out = root / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        'evaluation_by_task_model.csv': combined,
        'paired_by_task.csv': by_task,
        'calibration_bins.csv': pd.concat(bins_all, ignore_index=True),
        'exploratory_macro_summary.csv': descriptives,
    }
    for name, frame in paths.items():
        frame.to_csv(out / name, index=False, float_format='%.17g')
    figures = {}
    for task_id in manifest['tasks']:
        filename = f'reliability_task_{task_id}.png'
        save_reliability_plot(pd.concat(bins_all, ignore_index=True), out / filename, int(task_id))
        figures[filename] = sha256_file(out / filename)
    manifest['figure_sha256'] = figures
    manifest['missing_tasks'] = missing
    manifest['n_completed_tasks'] = len(summaries)
    manifest['n_models'] = len(combined)
    manifest['analysis_outputs_sha256'] = {name: sha256_file(out/name) for name in paths}
    (out/'analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('VERIFIED TASKS:', len(summaries), '/', len(R8_TASK_IDS), 'MODEL EVALUATIONS:', len(combined))
    print(by_task[['task_id','delta_agnam_minus_main_auroc','delta_agnam_minus_main_brier','delta_agnam_minus_main_ece_10']].round(5).to_string(index=False))
    print('SAVED:', out)
    return combined, by_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    analyze(allow_partial=args.allow_partial)


if __name__ == '__main__':
    main()
