"""Read-only audit of the 240 external-proposal + ISR outcomes.
Run from the AG-NAM repository root. Writes a NEW analysis directory only.
No training, threshold changes, or changes to existing benchmark outputs.
"""
from pathlib import Path
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t, wilcoxon

ROOT = Path('results/revision/external_isr_hybrid/core_synthetic')
OUT = ROOT.parent / 'final_analysis_v1'
TRUTH = {'S1': 3, 'S2': 4, 'S3': 2, 'S4': 4}
METHODS = ['friedman_h_gbdt', 'nid_shared_mlp', 'sian_fid_shared_mlp']
P, R, I = 'proposal_standalone_seconds_sum', 'residual_seconds_sum_shared', 'isr_wall_seconds'
PROXY = {frozenset(p) for p in [('x2','x11'),('x1','x12'),('x11','x12'),('x6','x13'),('x5','x14'),('x13','x14')]}
KEY = ['scenario', 'method']
EPS = 1e-10


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return bool(np.isclose(a, b, atol=1e-8, rtol=1e-7))


def main():
    if OUT.exists():
        raise FileExistsError(f'Output already exists, refusing to overwrite: {OUT}')
    summaries, matched, proxy_rows, inputs = [], [], [], []
    for s, nt in TRUTH.items():
        for r in range(1, 21):
            for m in METHODS:
                p = ROOT / s / f'realization_{r:02d}'
                sf, pf = p / f'{m}_summary.csv', p / f'{m}_isr_pairs.csv'
                tag = f'{s}/R{r:02d}/{m}'
                a0, q = pd.read_csv(sf), pd.read_csv(pf)
                require(len(a0) == 1, f'{tag}: summary row count')
                a = a0.iloc[0]
                require(a.scenario == s and a.method == m and int(a.realization_number) == r, f'{tag}: identity mismatch')
                require(int(a.n_true) == nt and int(a.n_runs) == 5 and close(a.selection_threshold, .6) and close(a.isr_threshold, .6), f'{tag}: protocol mismatch')
                require(all(np.isfinite(float(a[c])) and float(a[c]) >= 0 for c in [P,R,I]), f'{tag}: invalid runtime')
                require(q.is_true.isin([True,False]).all() and q.isr_retained.isin([True,False]).all(), f'{tag}: invalid pair flags')
                require(q.selection_probability.between(.6,1).all(), f'{tag}: invalid selection probability')
                require(not q.isr.isna().any(), f'{tag}: missing ISR')
                require((q.isr_retained.eq(True) == q.isr.ge(float(a.isr_threshold))).all(), f'{tag}: ISR decision mismatch')
                require(len({frozenset((x,y)) for x,y in zip(q.feature_a,q.feature_b)}) == len(q), f'{tag}: duplicate pairs')
                q = q.sort_values(['selection_probability','feature_a','feature_b'], ascending=[False,True,True])
                true, keep = q.is_true.eq(True), q.isr_retained.eq(True)
                n, k, tp, itp = len(q), int(keep.sum()), int(true.sum()), int((true & keep).sum())
                require(tp <= nt, f'{tag}: too many true pairs')
                expected = {'n_selection':n,'n_isr':k,'selection_tp':tp,'selection_fp':n-tp,'selection_fn':nt-tp,'isr_tp':itp,'isr_fp':k-itp,'isr_fn':nt-itp}
                require(all(a[c] == v for c,v in expected.items()), f'{tag}: summary/pair count mismatch')
                f1, sf1 = 2*itp/(k+nt), 2*tp/(n+nt)
                require(close(a.isr_f1,f1) and close(a.selection_f1,sf1), f'{tag}: F1 mismatch')
                ptp = int(q.head(k).is_true.eq(True).sum())
                cutoff = q.iloc[k-1].selection_probability if k else float('inf')
                h, e = q[q.selection_probability>cutoff], q[q.selection_probability==cutoff]
                ht, et, slots = int(h.is_true.eq(True).sum()), int(e.is_true.eq(True).sum()), k-len(h)
                lo, hi = ht+max(0,slots-(len(e)-et)), ht+min(slots,et)
                uniform = ht+slots*et/len(e) if len(e) else ht
                require(lo <= ptp <= hi and lo <= uniform <= hi, f'{tag}: tie bounds mismatch')
                base = {'scenario':s,'realization':r,'method':m}
                summaries.append({**a.to_dict(), **base, 'delta_f1':float(a.isr_f1-a.selection_f1), 'standalone_component_total_seconds':float(a[P]+a[R]+a[I])})
                matched.append({**base, 'k':k,'isr_tp':itp,'prob_tp':ptp,'prob_tp_min':lo,'prob_tp_max':hi,'boundary_tie':int(0<slots<len(e)),'robust_win':int(itp>hi),'robust_loss':int(itp<lo),'selection_f1':float(a.selection_f1),'isr_f1':f1,'prob_f1':2*ptp/(k+nt),'prob_uniform_f1':2*uniform/(k+nt),'prob_best_f1':2*hi/(k+nt),'prob_worst_f1':2*lo/(k+nt)})
                if s == 'S3':
                    pm = pd.Series([frozenset((x,y)) in PROXY for x,y in zip(q.feature_a,q.feature_b)],index=q.index,dtype=bool)
                    proxy_rows.append({**base,'proxy_before':int((pm & ~true).sum()),'proxy_after':int((pm & ~true & keep).sum()),'other_before':int((~pm & ~true).sum()),'other_after':int((~pm & ~true & keep).sum()),'tp_before':tp,'tp_after':itp})
                for f in [sf,pf]:
                    inputs.append({'path':f.as_posix(),'sha256':sha(f)})
    a, d, px = pd.DataFrame(summaries), pd.DataFrame(matched), pd.DataFrame(proxy_rows)
    require(len(a)==len(d)==240 and len(px)==60 and len(inputs)==480, 'Unexpected experiment count')
    require(a.groupby(KEY).size().eq(20).all() and a.groupby(['scenario','realization'])[R].nunique().eq(1).all(), 'Group size or shared residual inconsistency')
    # Descriptive recovery metrics; undefined FP reduction remains NaN.
    metrics = ['interaction_ap_mean','selection_jaccard','selection_precision','selection_recall','selection_f1','isr_precision','isr_recall','isr_f1','delta_f1','fp_reduction']
    recovery = a.groupby(KEY)[metrics].agg(['mean','std','count'])
    recovery.columns = ['_'.join(c) for c in recovery.columns]
    counts = a.groupby(KEY)[['n_selection','n_isr','selection_tp','selection_fp','isr_tp','isr_fp']].sum()
    counts['pooled_fp_reduction'] = (counts.selection_fp-counts.isr_fp)/counts.selection_fp.replace(0,np.nan)
    counts['pooled_selected_tp_retention'] = counts.isr_tp/counts.selection_tp.replace(0,np.nan)
    recovery = recovery.join(counts)
    # The same two comparisons and global 24-test Holm correction as the terminal audit.
    stat = []
    for (s,m),g in d.groupby(KEY):
        for label,b in [('ISR_vs_Selection','selection_f1'),('ISR_vs_Probability','prob_f1')]:
            v = (g.isr_f1-g[b]).round(12).to_numpy()
            half = t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
            p = 1.0 if np.all(abs(v)<=EPS) else float(wilcoxon(v,zero_method='wilcox',alternative='two-sided').pvalue)
            stat.append(dict(scenario=s,method=m,comparison=label,n=len(v),mean_delta=v.mean(),ci_low=v.mean()-half,ci_high=v.mean()+half,wins=int((v>EPS).sum()),losses=int((v<-EPS).sum()),ties=int((abs(v)<=EPS).sum()),p_raw=p))
    z = pd.DataFrame(stat).sort_values('p_raw')
    z['p_holm'] = np.minimum(1,np.maximum.accumulate(z.p_raw.to_numpy()*np.arange(len(z),0,-1)))
    z['significant'] = z.p_holm < .05
    z = z.sort_values(['comparison','scenario','method'])
    mcounts = ['k','isr_tp','prob_tp','prob_tp_min','prob_tp_max','boundary_tie','robust_win','robust_loss']
    d['delta_f1'] = d.isr_f1-d.prob_f1
    mtable = d.groupby(KEY)[mcounts].sum().join(d.groupby(KEY)[['isr_f1','prob_f1','prob_uniform_f1','prob_best_f1','prob_worst_f1','delta_f1']].mean())
    runtime = a.groupby(KEY)[[P,R,I,'standalone_component_total_seconds']].agg(['mean','std'])
    runtime.columns = ['_'.join(c) for c in runtime.columns]
    joint = a.groupby(['scenario','realization']).agg(proposal_sum=(P,'sum'),isr_sum=(I,'sum'),shared_residual=(R,'first'))
    joint['joint_component_total_seconds'] = joint.sum(axis=1)
    joint_summary = joint.groupby(level='scenario').agg(['mean','std'])
    joint_summary.columns = ['_'.join(c) for c in joint_summary.columns]
    proxy_summary = px.groupby('method')[['proxy_before','proxy_after','other_before','other_after','tp_before','tp_after']].sum()
    outputs = {
        'experiment_summary_240.csv':(a,False),
        'recovery_by_scenario_method.csv':(recovery,True),
        'matched_cardinality_240.csv':(d,False),
        'matched_cardinality_by_scenario_method.csv':(mtable,True),
        'paired_statistics_global_holm.csv':(z,False),
        'runtime_by_scenario_method.csv':(runtime,True),
        'runtime_joint_by_realization.csv':(joint,True),
        'runtime_joint_by_scenario.csv':(joint_summary,True),
        's3_proxy_by_realization.csv':(px,False),
        's3_proxy_by_method.csv':(proxy_summary,True),
    }
    OUT.mkdir(parents=True,exist_ok=False)
    for name,(frame,index) in outputs.items():
        frame.to_csv(OUT/name,index=index)
    manifest = {'n_experiments':len(a),'n_inputs':len(inputs),'n_paired_tests':len(z),'input_files':inputs,'outputs':{name:sha(OUT/name) for name in outputs},'script_sha256':sha(Path(__file__)),'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'notes':['Post-submission exploratory analysis. No model training.','F1 confidence intervals are unadjusted paired t intervals. Holm across 24 Wilcoxon tests.','Uniform tie F1 is the analytical expectation; oracle bounds use labels and are not executable baselines.','Pair counts are pooled observations, not independent sample sizes.','Runtime totals are sums of recorded components, not independently measured end-to-end wall time.']}
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('VERIFIED EXPERIMENTS:',len(a),' / 240')
    print('HASHED INPUT FILES:',len(inputs),' / 480')
    print('PAIRED TESTS:',len(z),' / 24')
    print('SIGNIFICANT BY COMPARISON:',z.groupby('comparison').significant.sum().to_dict())
    print('MATCHED TOTALS:',d[mcounts].sum().to_dict())
    print('OUTPUT FILES:',len(outputs)+1)
    print('SAVED:',OUT)
    print('FINAL ANALYSIS COMPLETE')


if __name__ == '__main__':
    main()
