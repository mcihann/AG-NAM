"""Collect existing AG-NAM R6--R10 reviewer-evidence outputs (read only).

Run from your AG-NAM repository root in the existing Windows environment:
    python collect_r6_r10_missing_evidence.py

This script only reads files and creates a ZIP. It does NOT train models,
change results, or run analysis. Missing folders are reported, not hidden.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path.cwd().resolve()
OUTPUT = ROOT / 'AG_NAM_R6_R10_Additional_Evidence.zip'
FOLDERS = {
    'R6_realx_external': 'results/revision/external_interaction_baselines/core_realx_v1',
    'R7_deviance_residual': 'results/revision/residual_sensitivity/core_synthetic_v1',
    'R8_calibration': 'results/revision/openml_calibration_r8/v1',
    'R9_strength_fnr': 'results/revision/strength_fnr_r9/v1',
    'R9_predictive_threshold': 'results/revision/predictive_threshold_r9/v1',
    'R10_domain_grounded': 'results/revision/domain_grounded_r10/v1',
}
EXTRA_FILES = [
    'results/realx/benchmark/master_results.csv',
    'docs/revision_r6_realx_external_recovery.md',
    'docs/revision_r7_residual_sensitivity_protocol.md',
    'docs/revision_r8_openml_calibration_protocol.md',
    'docs/revision_r9_strength_fnr_predictive_protocol.md',
    'docs/revision_r9_predictive_analysis_addendum.md',
    'docs/revision_r10_domain_grounded_protocol.md',
    'docs/revision_r10_domain_evidence_findings.md',
]
# Large model checkpoints and intermediate arrays are intentionally omitted.
ALLOWED_SUFFIXES = {'.csv', '.json', '.txt', '.md', '.tsv', '.png', '.log', '.yaml', '.yml'}
MAX_FILE_BYTES = 50 * 1024 * 1024


def main() -> None:
    if not (ROOT / 'src' / 'agnam').exists():
        raise SystemExit('ERROR: Run this script from the AG-NAM repository root.')
    to_add: list[Path] = []
    missing_folders: list[str] = []
    present_counts: dict[str, int] = {}
    skipped_large: list[str] = []
    for label, rel in FOLDERS.items():
        folder = ROOT / rel
        if not folder.is_dir():
            missing_folders.append(rel)
            present_counts[label] = 0
            continue
        paths = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES]
        present_counts[label] = len(paths)
        to_add.extend(paths)
    for rel in EXTRA_FILES:
        path = ROOT / rel
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES:
            to_add.append(path)
    entries = []
    unique_paths = sorted(set(to_add), key=lambda p: p.relative_to(ROOT).as_posix())
    with ZipFile(OUTPUT, 'w', compression=ZIP_DEFLATED, compresslevel=7) as archive:
        for path in unique_paths:
            rel = path.relative_to(ROOT).as_posix()
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                skipped_large.append(rel)
                continue
            raw = path.read_bytes()
            archive.writestr(rel, raw)
            entries.append({'path': rel, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        manifest = {
            'purpose': 'AG-NAM R6-R10 missing evidence completion',
            'root': 'AG-NAM relative paths',
            'read_only_source_collection': True,
            'training_performed': False,
            'existing_results_modified': False,
            'files_included': len(entries),
            'present_counts': present_counts,
            'missing_folders': missing_folders,
            'skipped_large_files': skipped_large,
            'files': entries,
            'note': 'Generated artifacts are source-preserving. Missing directories may mean that analyses were never completed, were stored elsewhere, or were not copied to this repository.'
        }
        archive.writestr('R6_R10_ADDITIONAL_COLLECTION_MANIFEST.json', json.dumps(manifest, indent=2, ensure_ascii=False))
    print('=== AG-NAM R6-R10 ADDITIONAL EVIDENCE COLLECTION ===')
    print('ARCHIVE:', OUTPUT)
    print('FILES:', len(entries))
    print('ZIP MiB:', round(OUTPUT.stat().st_size / (1024 * 1024), 3))
    print('MODULE COUNTS:', present_counts)
    print('MISSING FOLDERS:', missing_folders or 'NONE')
    print('SKIPPED LARGE:', len(skipped_large))
    print('Training: NO | Existing results modified: NO')
    print('NEXT: Upload AG_NAM_R6_R10_Additional_Evidence.zip to the conversation.')


if __name__ == '__main__':
    main()
