"""Regression tests for TreeExtra log conversion, independent of training."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_revision_additive_groves_analysis.py"
MODULE_SPEC = importlib.util.spec_from_file_location("revision_additive_groves_analysis", SCRIPT)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)


def test_parses_treeextra_present_absent_and_small() -> None:
    log = "\n".join(
        (
            "Restricting interaction between x3 and x4",
            "Performance: 0.789146. 25.5423 standard deviations from the mean. Interaction is present.",
            "Restricting interaction between x4 and x8",
            "Performance: 0.838971. 3.99479 standard deviations from the mean. Small interaction is present.",
            "Restricting interaction between x1 and x2",
            "Performance: 0.855259. -3.0493 standard deviations from the mean. Interaction is absent.",
        )
    )
    parsed = MODULE._parse_log(log)
    assert len(parsed) == 3
    assert parsed[("x3", "x4")][0] == pytest.approx(25.5423)
    assert parsed[("x4", "x8")][1] == "small interaction is present"
    assert parsed[("x1", "x2")][0] == pytest.approx(-3.0493)


def test_rejects_duplicate_pairs_in_any_order() -> None:
    log = (
        "Restricting interaction between x3 and x4\n"
        "Performance: 0.8. 3.1 standard deviations from the mean. Interaction is present.\n"
        "Restricting interaction between x4 and x3\n"
        "Performance: 0.9. 2.1 standard deviations from the mean. Interaction is absent."
    )
    with pytest.raises(ValueError, match="Duplicate"):
        MODULE._parse_log(log)


def test_rejects_logs_without_completed_pair_tests() -> None:
    with pytest.raises(ValueError, match="No completed"):
        MODULE._parse_log("Iteration 1 out of 210")


def test_attribute_and_exclusion_parser(tmp_path: Path) -> None:
    p = tmp_path / "data.fs.attr"
    p.write_text(
        "class: 0,1 (class).\n"
        "x1: cont.\nx2: cont.\nx3: cont.\n"
        "contexts:\nx3 never.\n",
        encoding="utf-8",
    )
    assert MODULE._attribute_names(p) == ("x1", "x2", "x3")
    assert MODULE._excluded_features(p) == {"x3"}


def test_native_decision_f1_counts_small_positive() -> None:
    parsed = {
        ("x3", "x4"): (25.54, "interaction is present"),
        ("x5", "x6"): (3.69, "small interaction is present"),
        ("x4", "x8"): (3.99, "small interaction is present"),
        ("x8", "x9"): (7.52, "interaction is present"),
        ("x1", "x2"): (-3.04, "interaction is absent"),
    }
    truth = {("x3", "x4"), ("x5", "x6"), ("x8", "x9")}
    native = MODULE._native_counts(parsed, truth)
    assert native["native_tp"] == 3
    assert native["native_fp"] == 1
    assert native["native_fn"] == 0
    assert math.isclose(native["native_f1"], 6 / 7)
