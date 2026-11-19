from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from agnam.data.preprocessing import FeatureSpec, TransformedTabular
from agnam.revision import domain_grounded_validation as d


def make_tabular(x, y, kinds=('numeric','numeric')):
    x=np.asarray(x); y=np.asarray(y)
    specs=[]
    num=[]; missing=[]; cat=[]
    ni=ci=0
    for col, (v, kind) in enumerate(zip([x,y], kinds)):
        if kind=='numeric':
            specs.append(FeatureSpec(f'x{col}',kind,col,ni))
            num.append(v.astype('float32')); missing.append(np.zeros(len(v),dtype='float32'))
            ni+=1
        else:
            specs.append(FeatureSpec(f'x{col}',kind,col,ci,int(np.max(v))+1))
            cat.append(v.astype('int64'));ci+=1
    return TransformedTabular(
        numeric=np.column_stack(num).astype('float32') if num else np.zeros((len(x),0),dtype='float32'),
        numeric_missing=np.column_stack(missing).astype('float32') if missing else np.zeros((len(x),0),dtype='float32'),
        categorical=np.column_stack(cat).astype('int64') if cat else np.zeros((len(x),0),dtype='int64'),
        numeric_feature_names=tuple(s.name for s in specs if s.kind=='numeric'),
        categorical_feature_names=tuple(s.name for s in specs if s.kind=='categorical'),
        feature_specs=tuple(specs),
    )


def test_task_role_is_explicit():
    assert d.task_role(146819)=='locked_case_study'
    assert d.task_role(14952)=='posthoc_supplement'
    with pytest.raises(ValueError):d.task_role(37)


def test_inventory_exact_counts(tmp_path):
    master = tmp_path/'master_results.csv'
    records=[]
    for task, name, count in [(146819,'climate',2),(3021,'sick',4),(3,'chess',7),(14952,'phish',9)]:
        records.append({'task_id':task,'dataset_name':name,'feature_type':'numeric','n_isr_retained':count})
        pd.DataFrame({'feature_j':[f'x{i}' for i in range(count)],
                      'feature_k':[f'y{i}' for i in range(count)],
                      'selection_frequency':[.6]*count,
                      'isr_retained':[True]*count}).to_csv(tmp_path/f'task_{task}_pairs.csv',index=False)
    pd.DataFrame(records).to_csv(master,index=False)
    inv=d.build_pair_inventory(master)
    assert len(inv)==22
    assert (inv.selection_role=='locked_case_study').sum()==13
    assert (inv.selection_role=='posthoc_supplement').sum()==9
    assert set(inv.external_evidence_status)=={'not_yet_adjudicated'}
    out=tmp_path/'export'
    path=d.export_pair_inventory(master_path=master,output_root=out)
    frame=pd.read_csv(path)
    frame.loc[0,'external_evidence_status']='reviewed'
    frame.to_csv(path,index=False)
    assert d.export_pair_inventory(master_path=master,output_root=out)==path
    assert pd.read_csv(path).loc[0,'external_evidence_status']=='reviewed'


def test_inventory_rejects_duplicate(tmp_path):
    p=tmp_path/'master_results.csv'
    pd.DataFrame([{'task_id':3,'dataset_name':'chess','feature_type':'categorical',
                   'n_isr_retained':7}]).to_csv(p,index=False)
    pd.DataFrame({'feature_j':['a']*7,'feature_k':['b']*7,
                  'selection_frequency':[.6]*7,'isr_retained':[True]*7}).to_csv(tmp_path/'task_3_pairs.csv',index=False)
    with pytest.raises(ValueError,match='Master lacks unique task'):
        d.build_pair_inventory(p)


def test_numeric_numeric_support():
    xx=np.array([-2,-2,2,2]*6,dtype=float)
    yy=np.array([-2,2,-2,2]*6,dtype=float)
    tab=make_tabular(xx,yy)
    rect=d.supported_rectangle(tab,*tab.feature_specs)
    assert rect['supported']
    assert rect['cell_counts']==[6,6,6,6]


def test_categorical_support_and_out_of_support():
    x=np.array([2,2,3,3]*5)
    y=np.array([2,3,2,3]*5)
    tab=make_tabular(x,y,('categorical','categorical'))
    assert d.supported_rectangle(tab,*tab.feature_specs)['supported']
    bad=make_tabular([2]*15+[3]*15,[2]*15+[3]*15,('categorical','categorical'))
    rect=d.supported_rectangle(bad,*bad.feature_specs)
    assert not rect['supported']
    assert 0 in rect['cell_counts']


def test_unknown_categories_do_not_support():
    tab=make_tabular(np.full(20,1),np.full(20,2),('categorical','categorical'))
    assert not d.supported_rectangle(tab,*tab.feature_specs)['supported']


class MultiplyPair(torch.nn.Module):
    interaction_pairs=(('x0','x1'),)
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(1.))
    def raw_interaction_contributions(self,numeric,numeric_missing,categorical):
        return (numeric[:,0]*numeric[:,1]*self.weight).unsqueeze(1)


def test_true_nonadditivity_is_detected_in_logit_units():
    tab=make_tabular([-1,-1,1,1]*4,[-1,1,-1,1]*4)
    result=d.pair_mixed_difference(MultiplyPair(),tab,0,device=torch.device('cpu'))
    assert result['rectangle_supported']
    assert result['mixed_difference_logit']==pytest.approx(4.0)


def test_additive_pair_function_has_zero_mixed_difference():
    class Additive(MultiplyPair):
        def raw_interaction_contributions(self,numeric,numeric_missing,categorical):
            return (numeric[:,0]+numeric[:,1]).unsqueeze(1)
    tab=make_tabular([-1,-1,1,1]*4,[-1,1,-1,1]*4)
    result=d.pair_mixed_difference(Additive(),tab,0,device=torch.device('cpu'))
    assert result['rectangle_supported']
    assert result['mixed_difference_logit']==pytest.approx(0.)


def test_unsupported_rectangle_is_nan_not_zero():
    tab=make_tabular(np.ones(20),np.ones(20))
    result=d.pair_mixed_difference(MultiplyPair(),tab,0,device=torch.device('cpu'))
    assert not result['rectangle_supported']
    assert np.isnan(result['mixed_difference_logit'])


class FakeDecomposition(torch.nn.Module):
    interaction_pairs=(('x0','x1'),)
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(1.))
    def forward(self,numeric,numeric_missing,categorical):
        main=torch.stack([numeric[:,0], numeric[:,1]],dim=1)
        inter=(numeric[:,0]*numeric[:,1]).unsqueeze(1)
        baseline=torch.tensor(0.5, device=numeric.device)
        return SimpleNamespace(logits=baseline+main.sum(1)+inter.sum(1),
                               main_contributions=main,interaction_contributions=inter,baseline=baseline)


def test_heldout_decomposition_is_exact():
    tab=make_tabular([-1,-1,1,1]*3,[-1,1,-1,1]*3)
    p,frame,logits,error=d.heldout_contribution_audit(FakeDecomposition(),tab,
                                                       device=torch.device('cpu'),batch_size=5)
    assert len(p)==len(tab)==len(frame)==len(logits)
    assert error<1e-7
    assert p.min()>0 and p.max()<1


def test_cached_metadata_hashes(tmp_path):
    master=tmp_path/'master_results.csv'; pairs=tmp_path/'task_3_pairs.csv'
    master.write_text('foo');pairs.write_text('bar')
    output=tmp_path/'task_3';output.mkdir()
    artifact=output/'pair_model_evidence.csv';artifact.write_text('x\n1\n')
    meta={'task_id':3,'master_sha256':d.sha256_file(master),
          'pair_audit_sha256':d.sha256_file(pairs),
          'output_sha256':{'pair_model_evidence.csv':d.sha256_file(artifact)}}
    (output/'metadata.json').write_text(json.dumps(meta))
    assert d._validate_cached(output,master_path=master,pair_path=pairs,task_id=3)
    artifact.write_text('x\n2\n')
    assert not d._validate_cached(output,master_path=master,pair_path=pairs,task_id=3)
