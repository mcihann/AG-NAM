from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from agnam.revision import openml_calibration_runner as r


def test_r8_locked_eight_tasks_unique():
    assert len(r.R8_TASK_IDS) == len(set(r.R8_TASK_IDS)) == 8


def test_bins_all_zero_and_one_probabilities():
    frame = r.calibration_bins(np.array([0, 1]), np.array([0, 1]))
    assert frame['n'].sum() == 2
    assert int(frame.loc[0, 'n']) == 1
    assert int(frame.loc[9, 'n']) == 1
    assert int(frame.loc[9, 'observed_fraction']) == 1


def test_perfect_calibration_has_zero_ece_and_brier():
    result, _ = r.calibration_metrics([0, 1, 0, 1], [0, 1, 0, 1])
    assert result['brier'] == 0
    assert result['ece_10'] == 0
    assert result['auroc'] == 1


def test_calibration_expected_brier_and_ece():
    result, _ = r.calibration_metrics([0,1], [0.2,0.8])
    assert result['brier'] == pytest.approx(0.04)
    assert result['ece_10'] == pytest.approx(0.2)


@pytest.mark.parametrize('bad', [[-0.1, 0.2], [0.0, 1.2], [0.0, np.nan]])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        r.calibration_bins([0, 1], bad)


def test_invalid_binary_labels_rejected():
    with pytest.raises(ValueError):
        r.calibration_metrics([0, 2], [0.1, 0.9])


def test_empty_bin_keeps_nan():
    _, bins = r.calibration_metrics([0, 1], [0.1,0.1])
    assert (bins['n']==0).sum() == 9
    assert pd.isna(bins.loc[bins['n'].eq(0), 'mean_probability']).all()


def test_read_retained_pairs_count_and_feature_names(tmp_path):
    p = tmp_path/'pairs.csv'
    pd.DataFrame({'feature_j':['a','a','b'], 'feature_k':['b','c','c'],
                  'isr_retained':[True, False, True]}).to_csv(p,index=False)
    pairs = r.read_retained_pairs(p, feature_names=('a','b','c'), expected=2)
    assert set(map(frozenset,pairs)) == {frozenset(('a','b')),frozenset(('b','c'))}


def test_read_retained_pairs_allows_zero(tmp_path):
    p=tmp_path/'pairs.csv'
    pd.DataFrame({'feature_j':['a'],'feature_k':['b'],'isr_retained':[False]}).to_csv(p,index=False)
    assert r.read_retained_pairs(p,feature_names=('a','b'), expected=0)==()


def test_read_retained_pairs_rejects_invalid_feature(tmp_path):
    p=tmp_path/'pairs.csv'
    pd.DataFrame({'feature_j':['a'],'feature_k':['absent'],'isr_retained':[True]}).to_csv(p,index=False)
    with pytest.raises(ValueError):r.read_retained_pairs(p,feature_names=('a','b'),expected=1)


def test_read_retained_pairs_rejects_duplicate_undirected(tmp_path):
    p=tmp_path/'pairs.csv'
    pd.DataFrame({'feature_j':['a','b'],'feature_k':['b','a'],'isr_retained':[True, True]}).to_csv(p,index=False)
    with pytest.raises(ValueError):r.read_retained_pairs(p,feature_names=('a','b'),expected=2)


def test_read_retained_pairs_rejects_bad_boolean(tmp_path):
    p=tmp_path/'pairs.csv'
    pd.DataFrame({'feature_j':['a'],'feature_k':['b'],'isr_retained':['maybe']}).to_csv(p,index=False)
    with pytest.raises(ValueError):r.read_retained_pairs(p,feature_names=('a','b'),expected=1)


def test_sha256_array_sensitive_to_value():
    assert r.sha256_array([1,2]) != r.sha256_array([1,3])


def test_fit_final_model_config_matches_locked_defaults(monkeypatch):
    calls = {}
    class Model:
        pass
    monkeypatch.setattr(r, '_new_model', lambda mode,d,p,c: Model())
    def fake_train(**kwargs):
        calls['config'] = kwargs['config']
        return SimpleNamespace(best_epoch=7)
    monkeypatch.setattr(r, 'train_main_effect_nam', fake_train)
    monkeypatch.setattr(r, 'predict_probabilities', lambda model,transformed,device:np.array([0.3,0.8]))
    config=r.SyntheticModelConfig()
    data=SimpleNamespace(train_data=None,validation_data=None,test_data=None,y_train=[0,1],y_validation=[0,1])
    pred,info=r.fit_final_model('main',data,(),seed=123,config=config,device=torch.device('cpu'))
    assert pred.tolist() == [0.3,0.8]
    assert info['best_iteration']==8
    assert calls['config'].seed == 123
    assert calls['config'].learning_rate==config.final_learning_rate
    assert calls['config'].max_epochs==config.final_max_epochs
    assert np.isnan(info['peak_cuda_allocated_mib'])


def test_cached_files_verified_with_hash(tmp_path):
    master=tmp_path/'master.csv'; pairs=tmp_path/'task_49_pairs.csv'
    master.write_text('task_id\n49\n');pairs.write_text('x\n')
    taskdir=tmp_path/'out'; taskdir.mkdir()
    summary=pd.DataFrame({'mode':['main','agnam'],'some':[1,2]})
    probs=pd.DataFrame({'test_row':[0], 'y_true':[0], 'main_prob':[0.2], 'agnam_prob':[0.3]})
    bins=pd.DataFrame({'bin':list(range(20))})
    for name,data in [('summary.csv',summary),('test_probabilities.csv',probs),('reliability_bins.csv',bins)]:
        data.to_csv(taskdir/name,index=False)
    meta={'task_id':49,'n_test':1,'source_master_sha256':r.sha256_file(master),
          'source_pair_sha256':r.sha256_file(pairs),
          'output_sha256':{name:r.sha256_file(taskdir/name) for name in ['summary.csv','test_probabilities.csv','reliability_bins.csv']}}
    (taskdir/'metadata.json').write_text(json.dumps(meta))
    assert r._metadata_matches(taskdir,master,pairs,49)
    probs.to_csv(taskdir/'test_probabilities.csv',index=False,float_format='%.10f')
    assert not r._metadata_matches(taskdir,master,pairs,49)


def test_r8_end_to_end_fake_training_and_resume(tmp_path, monkeypatch):
    master = tmp_path/'master_results.csv'
    pd.DataFrame([{'task_id':49,'dataset_name':'fake','n_samples':10,'n_features':2,
                   'n_isr_retained':0,'main_auroc':1.,'agnam_auroc':1.}]).to_csv(master,index=False)
    pd.DataFrame({'feature_j':['a'],'feature_k':['b'],'isr_retained':[False]}).to_csv(tmp_path/'task_49_pairs.csv',index=False)
    loaded = SimpleNamespace(X=pd.DataFrame({'a':range(10),'b':range(10)}),
                             train_indices=np.arange(6), test_indices=np.arange(6,10))
    monkeypatch.setattr(r,'load_openml_task',lambda spec,protocol:loaded)
    monkeypatch.setattr(r,'audit_loaded_openml_task',lambda _:SimpleNamespace(audit_pass=True))
    data = SimpleNamespace(y_test=np.array([0,1,0,1]))
    monkeypatch.setattr(r,'prepare_final_prediction_data',lambda loaded,final_split_seed,validation_fraction:data)
    trained=[]
    def fit(mode, data, pairs, **kwargs):
        trained.append(mode)
        return np.array([.1,.9,.2,.8]), {'final_fit_and_prediction_seconds':1.,
            'peak_cuda_allocated_mib':np.nan,'peak_cuda_reserved_mib':np.nan,
            'peak_cuda_allocated_above_baseline_mib':np.nan,
            'sampled_peak_process_rss_mib':42.,'best_iteration':5}
    monkeypatch.setattr(r,'fit_final_model',fit)
    root=tmp_path/'results'
    result=r.run_r8_task(49, output_root=root, master_path=master, device=torch.device('cpu'))
    assert len(result)==2
    assert trained==['main','agnam']
    assert (root/'task_49'/'metadata.json').exists()
    second=r.run_r8_task(49, output_root=root, master_path=master, device=torch.device('cpu'))
    assert len(second)==2 and trained==['main','agnam']
    (root/'task_49'/'test_probabilities.csv').write_text('corrupt')
    with pytest.raises(RuntimeError,match='unverified'):
        r.run_r8_task(49, output_root=root, master_path=master, device=torch.device('cpu'))
