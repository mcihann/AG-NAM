from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_r11_comparison_tables.py"
spec = importlib.util.spec_from_file_location("r11_tables_module", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    synthetic = tmp_path / "results/synthetic"
    ext = tmp_path / "results/revision/external_interaction_baselines"
    all_agnam = []
    for sn, scenario in enumerate(module.SCENARIOS):
        for i in range(1, 21):
            seed = 10000 + 1000 * sn + i - 1
            frame = pd.DataFrame([
                dict(scenario=scenario, realization_number=i, dataset_seed=seed,
                     method=method, interaction_auprc=0.4 + j*0.1,
                     interaction_auroc=0.8, ndcg=0.7, f1_at_k=0.25,
                     f1_at_true_count=0.5, standalone_seconds=2.0)
                for j, method in enumerate(module.METHODS)
            ])
            p = ext / "core_synthetic" / scenario / f"realization_{i:02d}_summary.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(p, index=False)
            ar = pd.DataFrame([dict(scenario=scenario, realization_index=i-1,
                                    dataset_seed=seed, mean_interaction_auprc=0.3,
                                    selection_f1=0.2, isr_f1=0.4)])
            q = synthetic / scenario / f"realization_{i:02d}.csv"
            q.parent.mkdir(parents=True, exist_ok=True)
            ar.to_csv(q, index=False)
            all_agnam.append(ar)
    pub = synthetic / "publication/table_synthetic_structure.csv"
    pub.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(scenario=s, n_realizations=20, interaction_auprc_mean=0.3,
                       interaction_auprc_std=0, selection_f1_mean=0.2, isr_f1_mean=0.4,
                       false_positive_reduction_mean=0.6) for s in module.SCENARIOS]).to_csv(pub, index=False)

    for i in range(1, 6):
        seed = 10000 + i-1
        d = ext / "additive_groves" / "S1" / f"realization_{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(scenario="S1", realization_number=i, dataset_seed=seed,
                           method="additive_groves", interaction_auprc=.8,
                           interaction_auroc=.9, ndcg=.8, f1_at_k=.25,
                           f1_at_true_count=.5,
                           standalone_seconds=(float('nan') if i == 1 else 100.0))]
                     ).to_csv(d / "additive_groves_summary.csv", index=False)
        mani = dict(n_all_pairs=190, n_true_pairs=3, native_tp=3, native_fp=1,
                    native_fn=0, native_f1=6/7, n_tested_pairs=36,
                    n_untested_pairs=154)
        if i >= 2:
            mani["treeextra_training_parameters"] = dict(bagging_iterations={2:140,3:140,4:210,5:210}[i])
            mani["total_cpp_stage_seconds"] = 100.0
        (d / "additive_groves_manifest.json").write_text(json.dumps(mani), encoding="utf-8")
        pairs = [dict(feature_a=f"x{a}", feature_b=f"x{b}", score=float(a+b))
                 for a in range(1,21) for b in range(a+1,21)]
        pd.DataFrame(pairs).to_csv(d / "additive_groves_pair_scores.csv", index=False)
    return tmp_path


def test_end_to_end_coverage_and_separate_panels(repo: Path) -> None:
    tables, audit = module.collect(repo)
    assert audit["counts"] == {
        "full_standalone_rows": 240, "pilot_rows": 20,
        "agnam_realization_rows": 80, "treeextra_rows": 5,
    }
    full = tables["table_standalone_full_scope.csv"]
    pilot = tables["table_standalone_s1_paired_pilot.csv"]
    assert len(full) == 12 and set(full.n) == {20}
    assert len(pilot) == 4 and set(pilot.n) == {5}
    assert set(pilot.method) == set(module.PILOT_METHODS)
    assert (pilot.loc[pilot.method == "additive_groves", "runtime_measured_n"] == 4).all()
    markdown = module.render_manuscript_tables(tables)
    assert "AG-NAM repeated-discovery" in markdown
    assert "Extended S1 detector comparison" in markdown
    assert "n=5" in markdown
    assert "previous version" not in markdown.lower()
    assert "reviewer" not in markdown.lower()


def test_seed_mismatch_rejected(repo: Path) -> None:
    p = repo / "results/synthetic/S1/realization_02.csv"
    d = pd.read_csv(p)
    d["dataset_seed"] += 99
    d.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Dataset seed misalignment"):
        module.collect(repo)


def test_unfixed_bagging_rejected(repo: Path) -> None:
    p = repo / "results/revision/external_interaction_baselines/additive_groves/S1/realization_02/additive_groves_manifest.json"
    data = json.loads(p.read_text())
    data["treeextra_training_parameters"]["bagging_iterations"] = 90
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Uncorrected TreeExtra bagging"):
        module.collect(repo)


def test_duplicate_pair_rejected(repo: Path) -> None:
    p = repo / "results/revision/external_interaction_baselines/additive_groves/S1/realization_03/additive_groves_pair_scores.csv"
    df = pd.read_csv(p)
    df.loc[1, ["feature_a", "feature_b"]] = df.loc[0, ["feature_a", "feature_b"]].to_numpy()
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Duplicate feature pairs"):
        module.collect(repo)


def test_missing_baseline_rejected(repo: Path) -> None:
    p = repo / "results/revision/external_interaction_baselines/core_synthetic/S3/realization_13_summary.csv"
    p.unlink()
    with pytest.raises(ValueError, match="Missing source file"):
        module.collect(repo)


def test_ag_nam_publication_mean_mismatch_rejected(repo: Path) -> None:
    p = repo / "results/synthetic/publication/table_synthetic_structure.csv"
    df = pd.read_csv(p)
    df.loc[df.scenario == 'S4', 'interaction_auprc_mean'] = .9
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="publication mean"):
        module.collect(repo)


def test_write_outputs_and_overwrite_guard(repo: Path) -> None:
    destination = repo / "results/revision/r11_publication_tables"
    module.write_outputs(repo, destination)
    assert (destination / "r11_comparison_audit.json").is_file()
    assert (destination / "r11_manuscript_tables_draft.md").is_file()
    assert len(list(destination.glob("table_*.csv"))) == 5
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        module.write_outputs(repo, destination)
    module.write_outputs(repo, destination, overwrite=True)
