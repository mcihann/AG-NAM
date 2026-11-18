"""Post-hoc R7 summary; no training or modifications to primary results."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from agnam.revision.residual_sensitivity_runner import MODES, OUTPUT_ROOT, SCHEMA, _sha


def main():
    root = OUTPUT_ROOT
    summaries=[]; runs=[]; inputs=[]
    for s in ('S1','S2','S3','S4'):
        for r in range(1,21):
            d = root/s/f'realization_{r:02d}'
            meta_file=d/'realization_manifest.json'
            if not meta_file.is_file():
                raise FileNotFoundError(meta_file)
            meta=json.loads(meta_file.read_text(encoding='utf-8'))
            if meta['schema']!=SCHEMA or meta['scenario']!=s or meta['realization']!=r:
                raise ValueError(f'Mismatched manifest: {meta_file}')
            for key,path in [('run_metrics_sha256',d/'run_metrics.csv'),('summary_sha256',d/'summary.csv')]:
                if not path.is_file() or _sha(path)!=meta[key]:
                    raise ValueError(f'R7 summary hash mismatch: {path}')
            summary=pd.read_csv(d/'summary.csv'); run=pd.read_csv(d/'run_metrics.csv')
            if len(summary)!=2 or set(summary['mode'])!=set(MODES) or len(run)!=10:
                raise ValueError(f'Incomplete R7 comparison: {d}')
            if set(zip(run['mode'],run['run_index']))!=set((m,i) for m in MODES for i in range(5)):
                raise ValueError(f'Run mismatch: {d}')
            if not (summary['scenario']==s).all() or not (summary['realization']==r).all():
                raise ValueError(f'Wrong scenario/realization: {d}')
            summaries.append(summary);runs.append(run)
            for p in d.glob('runs/*'):
                if p.is_file():inputs.append({'path':p.as_posix(),'sha256':_sha(p)})
            inputs.extend([{'path':str(d/x),'sha256':_sha(d/x)} for x in ('summary.csv','run_metrics.csv','realization_manifest.json')])
    summary=pd.concat(summaries,ignore_index=True)
    runs=pd.concat(runs,ignore_index=True)
    cols=['interaction_auprc_mean','selection_f1','selection_precision','selection_recall','selection_count','selection_tp','selection_fp','proposer_seconds_sum','cross_mode_selection_jaccard']
    grouped=summary.groupby(['scenario','mode'])[cols].agg(['mean','std'])
    grouped.columns=['_'.join(c) for c in grouped.columns]
    grouped=grouped.reset_index()
    paired=[]
    for s,g in summary.groupby('scenario'):
        wide=g.pivot(index='realization',columns='mode',values=cols)
        for metric in ('interaction_auprc_mean','selection_f1'):
            delta=(wide[metric]['deviance']-wide[metric]['probability']).to_numpy(dtype=float)
            p=1.0 if np.all(delta==0) else float(wilcoxon(delta,zero_method='wilcox',alternative='two-sided').pvalue)
            # Paired bootstrap CI, fixed seed, realization is the unit.
            rng=np.random.default_rng(20261118)
            vals=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
            lower,upper=np.quantile(vals,[0.025,0.975])
            paired.append({'scenario':s,'comparison':'deviance_minus_probability','metric':metric,
                           'n':len(delta),'mean_delta':float(delta.mean()),
                           'ci95_bootstrap_low':float(lower),'ci95_bootstrap_high':float(upper),
                           'wins':int((delta>1e-12).sum()),'losses':int((delta< -1e-12).sum()),'ties':int((abs(delta)<=1e-12).sum()),
                           'p_raw':p})
    tests=pd.DataFrame(paired)
    tests['p_holm']=multipletests(tests['p_raw'].to_numpy(),method='holm')[1]
    tests['significant_global_holm_0p05']=tests['p_holm']<0.05
    out=root/'analysis'
    out.mkdir(parents=True,exist_ok=True)
    outputs={
        'summary_160.csv':summary,
        'run_metrics_800.csv':runs,
        'scenario_mode_summary.csv':grouped,
        'paired_tests_global_holm.csv':tests,
        'input_hashes.csv':pd.DataFrame(inputs).sort_values('path'),
    }
    for name,frame in outputs.items():
        frame.to_csv(out/name,index=False,float_format='%.17g')
    (out/'analysis_manifest.json').write_text(json.dumps({
        'schema':SCHEMA,'n_realizations':80,'n_summary_rows':len(summary),'n_run_rows':len(runs),
        'n_paired_tests':len(tests),'outputs_sha256':{n:_sha(out/n) for n in outputs},
    },sort_keys=True,indent=2),encoding='utf-8')
    print('VERIFIED R7 REALIZATIONS:',len(summaries),'SUMMARY ROWS:',len(summary),'RUN ROWS:',len(runs))
    print(grouped[['scenario','mode','interaction_auprc_mean_mean','selection_f1_mean']].to_string(index=False))
    print(tests[['scenario','metric','mean_delta','p_holm','significant_global_holm_0p05']].to_string(index=False))
    print('SAVED',out)


if __name__=='__main__':
    main()
