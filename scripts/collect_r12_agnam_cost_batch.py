#!/usr/bin/env python3
"""Read-only validated aggregate of matched AG-NAM/SIAN full-command cost records."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from datetime import datetime, timezone
from zipfile import ZipFile, ZIP_DEFLATED

SCENARIOS = ('S1', 'S2', 'S3', 'S4')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jread(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def collect(root: Path, scenarios: list[str], start: int, end: int, out: Path) -> tuple[int, int]:
    if out.exists():
        raise FileExistsError('Refusing to overwrite existing aggregate archive: '+str(out))
    contents: dict[str, bytes] = {}
    rows: list[dict] = []
    anchor_hashes = None
    anchor_head = None
    for scenario in scenarios:
        for num in range(start, end + 1):
            rid = f'{num:02d}'
            aname = f'r12_{scenario.lower()}_{rid}_agnam_fullpath'
            sname = f'r12_{scenario.lower()}_{rid}_sian'
            fpaths = {
                'ag_summary':root / f'results/revision/r12_agnam_fullpath_cost/{scenario}/realization_{rid}/summary.json',
                'ag_measure':root / f'results/revision/r12_cost_audit/{aname}/measurement.json',
                'ag_log':root / f'results/revision/r12_cost_audit/{aname}/command_output.log',
                'sian_summary':root / f'results/revision/r12_native_sian_pairwise/{scenario}/realization_{rid}/summary.json',
                'sian_measure':root / f'results/revision/r12_cost_audit/{sname}/measurement.json',
                'synthetic':root / f'results/synthetic/{scenario}/realization_{rid}.csv',
            }
            for role, path in fpaths.items():
                if not path.is_file():
                    raise FileNotFoundError(f'Missing {role}: {path}')
            ag=jread(fpaths['ag_summary']); am=jread(fpaths['ag_measure'])
            si=jread(fpaths['sian_summary']); sm=jread(fpaths['sian_measure'])
            if ag.get('status')!='complete' or (ag.get('scenario'),ag.get('realization'))!=(scenario,num):
                raise RuntimeError(f'Invalid AG-NAM result: {scenario} {rid}')
            for obj,role in ((am,'AG-NAM'),(sm,'SIAN')):
                if obj.get('returncode')!=0 or obj.get('experiment_run_status')!='complete':
                    raise RuntimeError(f'{role} runtime record incomplete: {scenario} {rid}')
            if am.get('label')!=aname or sm.get('label')!=sname:
                raise RuntimeError(f'Incorrect profiler label: {scenario} {rid}')
            if ag.get('split_index_sha256')!=si.get('split_index_sha256'):
                raise RuntimeError(f'Raw sample splits differ: {scenario} {rid}')
            if ag.get('siancode_raw_split_identity_check')!='exact_index_sha256_match':
                raise RuntimeError(f'Missing split audit: {scenario} {rid}')
            hashset=ag.get('source_sha256',{})
            head=ag.get('git',{}).get('head')
            if not hashset or not head:
                raise RuntimeError(f'Missing source/git provenance: {scenario} {rid}')
            if anchor_hashes is None:
                anchor_hashes=hashset;anchor_head=head
            if hashset!=anchor_hashes or head!=anchor_head:
                raise RuntimeError(f'AG-NAM source/Git commit mismatch: {scenario} {rid}')
            comparison=ag.get('original_synthetic_record_comparison', {})
            if 'isolated_auroc_minus_original' not in comparison or 'isolated_auprc_minus_original' not in comparison:
                raise RuntimeError(f'Original-run comparison missing: {scenario} {rid}')
            # A fresh CUDA rerun can show nondeterministic metric differences; record them
            # rather than silently declaring perfect replication or discarding the run.
            auroc_delta=float(comparison['isolated_auroc_minus_original'])
            auprc_delta=float(comparison['isolated_auprc_minus_original'])
            import math
            if not (math.isfinite(auroc_delta) and math.isfinite(auprc_delta)):
                raise RuntimeError(f'Invalid replication difference: {scenario} {rid}')
            if am.get('log_sha256')!=sha(fpaths['ag_log'].read_bytes()):
                raise RuntimeError(f'AG-NAM profiler log digest differs: {scenario} {rid}')
            a=float(am.get('elapsed_wall_seconds',0))
            s=float(sm.get('elapsed_wall_seconds',0))
            if a<=0 or s<=0: raise RuntimeError('Missing positive wall-time')
            rows.append({'scenario':scenario,'realization':num,
                'agnam_wall_seconds':a,'sian_wall_seconds':s,
                'agnam_to_sian_wall_ratio':a/s,
                'agnam_rss_mib':am.get('peak_process_tree_rss_mib_sampled'),
                'sian_rss_mib':sm.get('peak_process_tree_rss_mib_sampled'),
                'agnam_torch_cuda_allocated_mib':ag.get('peak_torch_cuda_allocated_mib'),
                'sian_torch_cuda_allocated_mib':si.get('peak_torch_cuda_allocated_mib'),
                'agnam_internal_wall_seconds':ag.get('stage_times',{}).get('entire_full_agnam_path_seconds'),
                'agnam_discovery_seconds':ag.get('stage_times',{}).get('five_run_discovery_seconds'),
                'agnam_isr_seconds':ag.get('stage_times',{}).get('isr_seconds'),
                'agnam_final_seconds':ag.get('stage_times',{}).get('final_preprocessing_training_test_seconds'),
                'agnam_auroc':ag.get('full_agnam_test',{}).get('auroc'),
                'sian_auroc':si.get('test_predictive_metrics',{}).get('test_auroc'),
                'agnam_vs_original_auroc_delta':auroc_delta,
                'agnam_vs_original_auprc_delta':auprc_delta,
                'agnam_retained_pairs':len(ag.get('isr_retained_pairs',[])),
                'sian_selected_pairs':si.get('n_selected_pairs')})
            for path in fpaths.values():
                rel=path.relative_to(root).as_posix()
                contents[rel]=path.read_bytes()
    if not rows: raise RuntimeError('No runs')
    csvbuf=io.StringIO(newline='')
    writer=csv.DictWriter(csvbuf,fieldnames=list(rows[0].keys()))
    writer.writeheader();writer.writerows(rows)
    contents['R12_COST_AGGREGATE_RUNS.csv']=csvbuf.getvalue().encode('utf-8')
    import statistics
    summaries=[]
    for scenario in scenarios:
        sub=[r for r in rows if r['scenario']==scenario]
        def describe(field):
            v=[float(r[field]) for r in sub if r[field] is not None]
            return {'n':len(v),'mean':statistics.mean(v) if v else None,
                    'sample_sd':statistics.stdev(v) if len(v)>1 else None}
        summaries.append({'scenario':scenario,'n_realizations':len(sub),
            'agnam_wall_seconds':describe('agnam_wall_seconds'),
            'sian_wall_seconds':describe('sian_wall_seconds'),
            'agnam_to_sian_wall_ratio':describe('agnam_to_sian_wall_ratio'),
            'agnam_rss_mib':describe('agnam_rss_mib'),
            'sian_rss_mib':describe('sian_rss_mib')})
    max_abs_delta=max(max(abs(r['agnam_vs_original_auroc_delta']),abs(r['agnam_vs_original_auprc_delta'])) for r in rows)
    contents['R12_COST_AGGREGATE_SUMMARY.json']=json.dumps({
        'analysis':'R12 matched descriptive end-to-end cost audit',
        'data_generation':'raw train validation test split hashes equal per realization',
        'git_commit':anchor_head,'scenario_summaries':summaries,
        'max_abs_original_predictive_metric_difference':max_abs_delta,
        'requires_review_of_rerun_metric_differences':max_abs_delta>1e-8,
        'cautions':['Descriptive cost comparison only, implementations and model capacities differ',
            'CPU RSS sampled (not instantaneous true peak)',
            'PyTorch allocated CUDA memory not equivalent to total GPU process memory',
            'R12 post-review comparisons are exploratory, not prespecified confirmatory endpoints',
            'Differences between fresh CUDA cost reruns and frozen original metrics are recorded, not treated as perfect replication'],
    },indent=2).encode('utf-8')
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
              'analysis':'r12_matched_full_command_cost',
              'scenarios':scenarios,'start':start,'end':end,'n_runs':len(rows),
              'sha256':{rel:sha(data) for rel,data in contents.items()}}
    with ZipFile(out,'w',ZIP_DEFLATED,compresslevel=9) as z:
        for rel,data in sorted(contents.items()): z.writestr(rel,data)
        z.writestr('R12_COST_BATCH_MANIFEST.json',json.dumps(manifest,indent=2))
    with ZipFile(out) as z:
        if z.testzip() is not None: raise RuntimeError('ZIP integrity failure')
    print('=== R12 AG-NAM/SIAN MATCHED COST COLLECTION ===')
    print('Run count:',len(rows),'Archive:',out)
    for s in summaries:
        print(s['scenario'],'mean AG-NAM seconds:',round(s['agnam_wall_seconds']['mean'],2),
              'mean SIAN seconds:',round(s['sian_wall_seconds']['mean'],2))
    print('Training: NO | Source results modified: NO')
    return len(rows),len(contents)


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenarios',choices=SCENARIOS,nargs='+',default=list(SCENARIOS))
    p.add_argument('--start',type=int,default=1)
    p.add_argument('--end',type=int,default=5)
    a=p.parse_args(argv)
    if not 1<=a.start<=a.end<=20:p.error('Expected 1 <= start <= end <= 20')
    root=Path.cwd().resolve()
    if not (root/'src/agnam').is_dir():p.error('Run from AG-NAM root')
    scenarios=list(dict.fromkeys(a.scenarios))
    spec=f'{scenarios[0]}' if len(scenarios)==1 else f'{scenarios[0]}_{scenarios[-1]}'
    out=root/f'AG_NAM_R12_AGNAM_{spec}_R{a.start:02d}_R{a.end:02d}_Matched_Cost_Evidence.zip'
    collect(root,scenarios,a.start,a.end,out)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
