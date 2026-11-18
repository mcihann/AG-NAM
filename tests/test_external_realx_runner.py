from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from agnam.revision import external_realx_runner as r


def test_locked_scope():
    assert len(r.select_specs()) == 45
    assert len(r.select_specs(strength="null")) == 45
    assert len(r.select_specs(strength="both")) == 90
    s = r.select_specs(task_id=49, realization=1)[0]
    assert s.run_id == "realx_49_r01_moderate"
    with pytest.raises(ValueError):
        r.select_specs(strength="weak")


def test_encoder_training_only_original_order():
    fit = pd.DataFrame({"cat": ["b", "a", None], "num": [0., 2., np.nan], "other": ["x", "x", "y"]})
    val = pd.DataFrame({"cat": ["z", None], "num": [100., np.nan], "other": ["y", "z"]})
    a,b,names = r.encode_original_features(fit,val)
    assert names == ("cat", "num", "other")
    np.testing.assert_array_equal(a[:,0], [3,2,0])
    np.testing.assert_array_equal(b[:,0], [1,0])
    assert b[0,1] > 100 and b[1,1] == 0
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert b[1,2] == 1


def test_ranking_moderate():
    names = ("a","b","c","d")
    scores = {p: float(i) for i,p in enumerate(r.full_pair_universe(names))}
    truth = {("c","d"),("a","b")}
    m, rank = r.ranking_metrics(names,scores,truth,2,1)
    assert rank[0][:2] == ("c","d")
    assert m["tp_at_k"] == 1 and m["fp_at_k"] == 1 and m["fn_at_k"] == 1
    assert m["f1_at_k"] == .5 and m["f1_at_matched"] == pytest.approx(2/3)
    assert 0 <= m["interaction_auprc"] <= 1


def test_null_no_fake_auc():
    names = ("a","b","c")
    m, _ = r.ranking_metrics(names,{p:1. for p in r.full_pair_universe(names)},set(),2,0)
    for key in ("interaction_auprc","interaction_auroc","ndcg","recall_at_k","f1_at_k","f1_at_true_count","precision_at_matched"):
        assert np.isnan(m[key])
    assert m["fp_at_k"] == 2 and m["fp_at_matched"] == 0
    assert m["precision_at_k"] == 0


def test_bad_ranking_refused():
    with pytest.raises(ValueError):
        r.ranking_metrics(("a","b","c"),{("a","b"):1.},set(),2,0)
    with pytest.raises(ValueError):
        r.ranking_metrics(("a","b"),{("a","b"):np.nan},set(),1,0)


def test_null_csv_label(tmp_path):
    path = tmp_path/"data.csv"
    path.write_text("strength,metric\nnull,\n",encoding="utf-8")
    a = r.read_csv(path)
    assert a.iloc[0].strength == "null" and pd.isna(a.iloc[0].metric)


@pytest.mark.parametrize("strength", ["moderate", "null"])
def test_mock_end_to_end_resume_and_corruption(tmp_path, monkeypatch, strength):
    spec = r.select_specs(strength=strength,task_id=49,realization=1)[0]
    names = tuple("abcdefghi")
    truth = {("a","b"),("c","d"),("e","f")} if strength == "moderate" else set()
    scores = {p: float(i+1) for i,p in enumerate(r.full_pair_universe(names))}
    anchor = {"n_isr_retained":1, "isr_pairs":"a::b", "mean_interaction_auprc":.2,
              "selection_f1":.3, "isr_f1":.5}
    monkeypatch.setattr(r,"read_frozen_anchor",lambda *a:(anchor,"anchor","evidence"))
    X = np.ones((12,9)); y=np.array([0,1]*6)
    monkeypatch.setattr(r,"prepare_realx",lambda *a:(X,y,X,y,names,truth,42,8,{}))
    monkeypatch.setattr(r,"get_device",lambda:torch.device("cpu"))
    monkeypatch.setattr(r,"seed_everything",lambda *a:None)
    monkeypatch.setattr(r,"version",lambda p:"0.9.5")
    calls=[]
    def h(**kwargs):
        calls.append("h"); return scores,1.,2.,3.,4.,5.
    def mlp(*a,**kw):
        calls.append("mlp"); return SimpleNamespace(model=None,scaler=None,training_seconds=2.,rss_start_mib=3.,rss_end_mib=4.,rss_peak_mib=5.,cuda_peak_allocated_mib=0.,best_epoch=2,best_val_auroc=.7)
    monkeypatch.setattr(r,"friedman_h_pair_scores",h)
    monkeypatch.setattr(r,"train_shared_sian_mlp",mlp)
    monkeypatch.setattr(r,"nid_pair_scores_from_model",lambda *a:scores)
    monkeypatch.setattr(r,"sian_fid_pair_scores",lambda **kw:(scores,2.,3.,4.,5.,0.))
    r.run_external_realx(spec,root=tmp_path/"new",frozen_root=tmp_path/"old")
    a,b,m = r.validate_outputs(spec,tmp_path/"new",anchor_hash="anchor",evidence_hash="evidence")
    assert len(a)==3 and len(b)==108
    r.run_external_realx(spec,root=tmp_path/"new",frozen_root=tmp_path/"old")
    assert calls == ["h","mlp"]
    path = r.paths_for(spec,tmp_path/"new")[1]
    path.write_text(path.read_text()+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="hash mismatch"):
        r.run_external_realx(spec,root=tmp_path/"new",frozen_root=tmp_path/"old")


def test_frozen_root_protected(tmp_path):
    with pytest.raises(ValueError,match="frozen"):
        r.run_external_realx(r.select_specs()[0], root=tmp_path/"frozen"/"new", frozen_root=tmp_path/"frozen")
