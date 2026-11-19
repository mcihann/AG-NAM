"""Small, deterministic tests for read-only R9-B analysis and fail-closed audit."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from agnam.revision.predictive_threshold_analysis import (
    CONFIG_IDS, REFERENCE, SCENARIOS, _bootstrap_deltas,
    analyze_predictive_threshold,
)


def _write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_fixture(tmp_path):
    root = tmp_path / "v1"
    for scenario in SCENARIOS:
        for n in range(1, 6):
            out = root / scenario / f"realization_{n:02d}"
            out.mkdir(parents=True)
            pair_sets = []
            models = {}
            records = []
            for ci, cid in enumerate(CONFIG_IDS):
                h = f"test_key_{ci}"
                pairs = [["a", "b"]] if ci % 2 else []
                value = 0.70 + 0.01 * (ci % 2)
                # Same reference across fixtures; zero-pair = Main.
                pair_sets.append({"config_id": cid, "pair_hash": h, "isr_pairs": pairs})
                models[h] = {"pairs": pairs, "auroc": value, "auprc": .5, "balanced_accuracy": .6, "f1": .4}
                records.append({"scenario": scenario, "realization_number": n, "config_id": cid, "pair_hash": h,
                                "n_isr_retained": len(pairs), "main_auroc_locked": .70, "agnam_auroc": value,
                                "delta_auroc_vs_main": value - .70, "auprc": .5, "balanced_accuracy": .6,
                                "f1": .4, "model_fit_cache_key": h})
            _write_json(out / "pair_sets.json", pair_sets)
            _write_json(out / "final_model_cache.json", {"source_hashes": {"original": "mock", "r4": "mock"},
                "pair_sets_sha256": _hash(out / "pair_sets.json"), "models": models})
            pd.DataFrame(records).to_csv(out / "predictive_summary.csv", index=False)
            _write_json(out / "manifest.json", {
                "scenario": scenario, "realization_number": n,
                "r4_recovery_parity": "9/9", "locked_primary_results_modified": False,
                "n_threshold_configs": 9, "n_unique_pair_sets": 9,
                "subset_definition": "S1-S4 realizations 1-5 (20 total), fixed before R9-B outcomes",
                "source_hashes": {"original": "mock", "r4": "mock"},
                "main_auroc": .70, "original_full_auroc": .70,
                "summary_sha256": _hash(out / "predictive_summary.csv"),
                "pair_sets_sha256": _hash(out / "pair_sets.json"),
                "model_cache_sha256": _hash(out / "final_model_cache.json"),
            })
    return root


def test_analysis_generates_complete_summary(tmp_path):
    root = _make_fixture(tmp_path)
    result = analyze_predictive_threshold(input_root=root, verify_sources=False, bootstrap_repeats=100)
    assert all(p.exists() for p in result.values())
    assert len(pd.read_csv(result["realization_summary_long.csv"])) == 180
    assert len(pd.read_csv(result["scenario_configuration_summary.csv"])) == 36
    assert len(pd.read_csv(result["configuration_summary.csv"])) == 9
    assert len(pd.read_csv(result["realization_audit.csv"])) == 20
    assert json.loads(result["manifest"].read_text())["total_final_model_fits"] == 180


def test_reference_difference_zero(tmp_path):
    root = _make_fixture(tmp_path)
    result = analyze_predictive_threshold(input_root=root, verify_sources=False, bootstrap_repeats=100)
    summary = pd.read_csv(result["paired_vs_reference.csv"]).set_index("config_id")
    ref = summary.loc[REFERENCE]
    assert ref.mean_paired_delta == 0
    assert ref.wins == 0 and ref.ties == 20 and ref.losses == 0


def test_corrupted_summary_fails(tmp_path):
    root = _make_fixture(tmp_path)
    target = root / "S2" / "realization_01" / "predictive_summary.csv"
    target.write_text(target.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        analyze_predictive_threshold(input_root=root, verify_sources=False, bootstrap_repeats=100)


def test_missing_realization_fails(tmp_path):
    root = _make_fixture(tmp_path)
    (root / "S3" / "realization_05" / "manifest.json").unlink()
    with pytest.raises(ValueError, match="Missing R9-B files"):
        analyze_predictive_threshold(input_root=root, verify_sources=False, bootstrap_repeats=100)


def test_model_value_mismatch_fails_even_if_file_hashes_updated(tmp_path):
    root = _make_fixture(tmp_path)
    out = root / "S1" / "realization_01"
    cache = json.loads((out / "final_model_cache.json").read_text())
    cache["models"]["test_key_0"]["auroc"] = 0.8
    _write_json(out / "final_model_cache.json", cache)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["model_cache_sha256"] = _hash(out / "final_model_cache.json")
    _write_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Cached auroc mismatch"):
        analyze_predictive_threshold(input_root=root, verify_sources=False, bootstrap_repeats=100)


def test_stratified_bootstrap_zero_for_constant_reference():
    rows = []
    for scenario in SCENARIOS:
        for n in range(1, 6):
            for cid in CONFIG_IDS:
                rows.append({"scenario": scenario, "realization_number": n, "config_id": cid, "paired_delta_vs_reference": 0.0})
    result = _bootstrap_deltas(pd.DataFrame(rows), n_bootstrap=100, seed=42)
    assert all(v == (0.0, 0.0) for v in result.values())
