"""Read-only R12 cost evidence collection; does not launch training or modify results."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',choices=('S1','S2','S3','S4'),default='S1')
    p.add_argument('--realization',type=int,default=1)
    p.add_argument('--label', default=None)
    a=p.parse_args(argv)
    if not 1<=a.realization<=20:p.error('realization must be 1..20')
    root=Path.cwd().resolve()
    if not (root/'src'/'agnam').is_dir():p.error('run from repository root')
    rid=f'{a.realization:02d}'
    label=a.label or f'r12_{a.scenario.lower()}_{rid}_agnam_fullpath'
    files=[
      root/'results'/'revision'/'r12_agnam_fullpath_cost'/a.scenario/f'realization_{rid}'/'summary.json',
      root/'results'/'revision'/'r12_cost_audit'/label/'measurement.json',
    ]
    optional=[
      root/'results'/'revision'/'r12_native_sian_pairwise'/a.scenario/f'realization_{rid}'/'summary.json',
      root/'results'/'synthetic'/a.scenario/f'realization_{rid}.csv',
    ]
    for f in files:
        if not f.is_file():raise FileNotFoundError(f'Required measured artifact missing: {f}')
    isolated=json.loads(files[0].read_text(encoding='utf8'))
    measured=json.loads(files[1].read_text(encoding='utf8'))
    if isolated.get('status')!='complete' or measured.get('experiment_run_status')!='complete' or measured.get('returncode')!=0:
        raise ValueError('Incomplete AG-NAM cost evidence, refusing to archive')
    if isolated['scenario']!=a.scenario or isolated['realization']!=a.realization:
        raise ValueError('Scenario/realization mismatch')
    if isolated.get('siancode_raw_split_identity_check')!='exact_index_sha256_match':
        raise ValueError('Raw AG-NAM/SIAN split identity not proven, refusing archive')
    files.extend(f for f in optional if f.is_file())
    out=root/f'AG_NAM_R12_{a.scenario}_R{rid}_Fullpath_Cost_Evidence.zip'
    if out.exists():raise FileExistsError(f'Archive exists, no overwrite: {out}')
    stored={}
    with ZipFile(out,'w',ZIP_DEFLATED,compresslevel=9) as z:
        for f in files:
            rel=f.relative_to(root).as_posix()
            data=f.read_bytes()
            z.writestr(rel,data)
            stored[rel]=sha(data)
        z.writestr('R12_COST_EVIDENCE_MANIFEST.json',json.dumps({
            'analysis':'r12_full_agnam_cost_isolated_path',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'scenario':a.scenario,'realization':a.realization,
            'training':'NO (collection only)', 'original_results_modified':'NO',
            'file_sha256':stored,
        },indent=2))
    assert ZipFile(out).testzip() is None
    print('=== R12 ISOLATED AG-NAM COST EVIDENCE ===')
    print('FILES:',len(stored),'ARCHIVE:',out)
    print('TRAINING: NO | EXISTING OUTPUTS MODIFIED: NO')
    return 0

if __name__=='__main__':raise SystemExit(main())
