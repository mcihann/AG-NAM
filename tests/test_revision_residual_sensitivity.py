"""Cheap numerical/provenance tests; do not fit GPU models."""
import json
from itertools import combinations
import numpy as np
import pandas as pd
import pytest

from agnam.revision.residual_sensitivity_runner import (
    MODES, _ranking_frame, _read_ranking, _save_csv, _save_json,
    _sha, _validate_ranking_frame, recover_probabilities,
    signed_deviance_residual, targets_for_mode,
)
from agnam.interpretation.interaction_scoring import RankedInteraction


def test_recovered_probabilities_roundtrip():
    y=np.array([0,1,1,0]);p=np.array([.1,.3,.99,.8]);r=y-p
    raw,safe=recover_probabilities(y,r)
    np.testing.assert_allclose(raw,p)
    np.testing.assert_allclose(safe,p)


def test_signed_deviance_expected_values_and_signs():
    y=np.array([0,1]); r=np.array([-.2,.2]); d=signed_deviance_residual(y,r)
    np.testing.assert_allclose(d,[ -np.sqrt(-2*np.log(.8)), np.sqrt(-2*np.log(.8)) ])
    assert d[0]<0<d[1]


def test_clip_rare_extreme_probs():
    d=signed_deviance_residual([0,1],[0.,0.]); assert np.isfinite(d).all()
    assert d[0]<0 and d[1]>0


@pytest.mark.parametrize('y,r', [([0],[1.2]),([1],[-.2]),([2],[0.]),([0],[np.nan])])
def test_invalid_cache_fails(y,r):
    with pytest.raises(ValueError): recover_probabilities(y,r)


def test_probability_targets_are_unmodified_float32():
    arrays={'train_residuals':np.array([-.2,.3]),'validation_residuals':np.array([-.4,.1])}
    a,b=targets_for_mode('probability',[0,1],[0,1],arrays)
    np.testing.assert_allclose(a,arrays['train_residuals'],rtol=1e-6)
    assert a.dtype==np.float32 and b.dtype==np.float32


def test_deviance_and_probability_have_same_label_signs():
    arrays={'train_residuals':np.array([-.2,.3]),'validation_residuals':np.array([-.4,.1])}
    a,b=targets_for_mode('deviance',[0,1],[0,1],arrays)
    assert a[0]<0<a[1] and b[0]<0<b[1]


def test_ranking_full_universe_and_order():
    features=('x1','x2','x3')
    ranking=(RankedInteraction('x1','x2',0,1,0.9,1),RankedInteraction('x1','x3',0,2,0.7,2),RankedInteraction('x2','x3',1,2,0.2,3))
    read=_validate_ranking_frame(_ranking_frame(ranking),features)
    assert len(read)==3
    bad=_ranking_frame(ranking).iloc[:-1]
    with pytest.raises(ValueError): _validate_ranking_frame(bad,features)


def test_ranking_rejects_unsorted():
    f=pd.DataFrame([['x1','x2',.1,1],['x1','x3',.8,2],['x2','x3',.3,3]],columns=['feature_j','feature_k','score','rank'])
    with pytest.raises(ValueError):_validate_ranking_frame(f,('x1','x2','x3'))


def test_manifest_hash_validation(tmp_path):
    ranks=(RankedInteraction('x1','x2',0,1,.5,1),)
    score=tmp_path/'score.csv'; meta=tmp_path/'meta.json'
    _save_csv(score,_ranking_frame(ranks));_save_json(meta,{'mode':'probability','ranking_sha256':_sha(score)})
    got,_=_read_ranking(score,meta,expected={'mode':'probability'},features=('x1','x2'))
    assert len(got)==1
    score.write_text('bad',encoding='utf-8')
    with pytest.raises(ValueError):_read_ranking(score,meta,expected={'mode':'probability'},features=('x1','x2'))


def test_no_unknown_mode():
    with pytest.raises(ValueError):targets_for_mode('logit',[1],[1],{'train_residuals':[0.2],'validation_residuals':[0.2]})
