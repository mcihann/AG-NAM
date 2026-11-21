#!/usr/bin/env python3
"""R12 AG-NAM isolated end-to-end cost runs (safe batch wrapper).

Defaults to a READ-ONLY dry run. Requires --execute to run fresh models.
Does not modify any pre-existing experimental outputs. Stops at incomplete
or inconsistent outputs rather than overwriting them.

Example:
  python scripts/run_r12_agnam_cost_batch.py --scenarios S1 S2 S3 S4 --start 1 --end 5
  python scripts/run_r12_agnam_cost_batch.py --scenarios S1 S2 S3 S4 --start 1 --end 5 --execute
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SCENARIOS = ('S1', 'S2', 'S3', 'S4')
ANCHOR_SCENARIO = 'S1'
ANCHOR_REALIZATION = 1


def require_repo(root: Path) -> None:
    for relative in (
        'src/agnam/benchmarking/synthetic_runner.py',
        'scripts/run_r12_agnam_isolated_cost.py',
        'scripts/collect_r12_agnam_fullpath_cost.py',
        'scripts/r12_profile_command.py',
    ):
        if not (root / relative).is_file():
            raise FileNotFoundError(f'Missing required file: {root / relative}')


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def paths(root: Path, scenario: str, realization: int) -> dict[str, Path | str]:
    rid = f'{realization:02d}'
    label = f'r12_{scenario.lower()}_{rid}_agnam_fullpath'
    sian_label = f'r12_{scenario.lower()}_{rid}_sian'
    return {
        'summary': root / 'results/revision/r12_agnam_fullpath_cost' / scenario / f'realization_{rid}/summary.json',
        'measurement': root / 'results/revision/r12_cost_audit' / label / 'measurement.json',
        'archive': root / f'AG_NAM_R12_{scenario}_R{rid}_Fullpath_Cost_Evidence.zip',
        'sian_summary': root / 'results/revision/r12_native_sian_pairwise' / scenario / f'realization_{rid}/summary.json',
        'sian_measurement': root / 'results/revision/r12_cost_audit' / sian_label / 'measurement.json',
        'label': label,
    }


def validate_completed(d: dict[str, Path | str], *, scenario: str, realization: int) -> None:
    summary = json.loads(Path(d['summary']).read_text(encoding='utf-8'))
    measure = json.loads(Path(d['measurement']).read_text(encoding='utf-8'))
    if summary.get('status') != 'complete' or summary.get('scenario') != scenario or summary.get('realization') != realization:
        raise RuntimeError(f'AG-NAM output incomplete/mismatched: {scenario} {realization:02d}')
    if summary.get('siancode_raw_split_identity_check') != 'exact_index_sha256_match':
        raise RuntimeError(f'AG-NAM/SIAN raw split not verified: {scenario} {realization:02d}')
    if measure.get('returncode') != 0 or measure.get('experiment_run_status') != 'complete':
        raise RuntimeError(f'Profiler failed: {scenario} {realization:02d}')
    if measure.get('label') != d['label']:
        raise RuntimeError(f'Profiler label mismatch: {scenario} {realization:02d}')
    for rel in ('entire_full_agnam_path_seconds', 'five_run_discovery_seconds', 'isr_seconds'):
        if float(summary.get('stage_times', {}).get(rel, 0)) <= 0:
            raise RuntimeError(f'Missing AG-NAM timing {rel}: {scenario} {realization:02d}')


def get_status(d: dict[str, Path | str], *, scenario: str, realization: int) -> str:
    exists = (Path(d['summary']).exists(), Path(d['measurement']).exists())
    if exists == (False, False):
        if Path(d['archive']).exists():
            raise RuntimeError('Archived result exists but underlying measured files missing: '+scenario+str(realization))
        return 'pending'
    if exists != (True, True):
        raise RuntimeError('Partial run found (manual audit needed, will not overwrite): '+scenario+str(realization))
    validate_completed(d, scenario=scenario, realization=realization)
    return 'completed+archived' if Path(d['archive']).exists() else 'completed (archive missing)'


def check_anchor(root: Path) -> dict:
    a = paths(root, ANCHOR_SCENARIO, ANCHOR_REALIZATION)
    validate_completed(a, scenario=ANCHOR_SCENARIO, realization=ANCHOR_REALIZATION)
    summary = json.loads(Path(a['summary']).read_text(encoding='utf-8'))
    for relative, expected in summary.get('source_sha256', {}).items():
        f = root / relative
        if not f.is_file() or sha_file(f) != expected:
            raise RuntimeError('Source changed after verified S1/01 cost pilot; STOP: '+relative)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenarios', choices=SCENARIOS, nargs='+', default=list(SCENARIOS))
    p.add_argument('--start', type=int, default=1)
    p.add_argument('--end', type=int, default=5)
    p.add_argument('--execute', action='store_true', help='Explicit consent for new model training')
    a = p.parse_args(argv)
    if not (1 <= a.start <= a.end <= 20):
        p.error('Expected 1 <= start <= end <= 20')
    scenarios = list(dict.fromkeys(a.scenarios))
    root = Path.cwd().resolve()
    require_repo(root)
    anchor = check_anchor(root)
    print('=== R12 AG-NAM COST BATCH ===')
    print('Mode:', 'EXECUTE NEW TRAINING' if a.execute else 'DRY RUN (NO TRAINING)')
    print('Scenarios:', ', '.join(scenarios), 'realizations:', a.start, 'to', a.end)
    print('Anchor git:', anchor.get('git', {}).get('head'))
    rows = []
    for scenario in scenarios:
        for num in range(a.start, a.end + 1):
            d = paths(root, scenario, num)
            if not Path(d['sian_summary']).is_file() or not Path(d['sian_measurement']).is_file():
                raise FileNotFoundError(f'SIAN summary/measurement missing: {scenario} {num:02d}')
            sm = json.loads(Path(d['sian_measurement']).read_text(encoding='utf-8'))
            if sm.get('returncode') != 0 or sm.get('experiment_run_status') != 'complete':
                raise RuntimeError(f'SIAN cost measurement incomplete: {scenario} {num:02d}')
            status = get_status(d, scenario=scenario, realization=num)
            rows.append((scenario, num, d, status))
            print(f'{scenario} R{num:02d}: {status}')
    print('Pending new trainings:', sum(row[3] == 'pending' for row in rows))
    if not a.execute:
        print('DRY RUN COMPLETE: no training and no output written.')
        return 0
    for scenario, num, d, status in rows:
        if status == 'pending':
            cmd = [sys.executable, 'scripts/r12_profile_command.py',
                   '--label', str(d['label']), '--execute', '--',
                   sys.executable, 'scripts/run_r12_agnam_isolated_cost.py',
                   '--scenario', scenario, '--realization', str(num), '--execute']
            print('LAUNCH:', scenario, num, flush=True)
            subprocess.run(cmd, cwd=root, check=True)
            validate_completed(d, scenario=scenario, realization=num)
        if not Path(d['archive']).exists():
            cmd = [sys.executable, 'scripts/collect_r12_agnam_fullpath_cost.py',
                   '--scenario', scenario, '--realization', str(num)]
            subprocess.run(cmd, cwd=root, check=True)
            if not Path(d['archive']).is_file():
                raise RuntimeError(f'Expected archive not created: {d["archive"]}')
    print('COST BATCH COMPLETE. Next: python scripts/collect_r12_agnam_cost_batch.py --scenarios '
          + ' '.join(scenarios) + f' --start {a.start} --end {a.end}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
