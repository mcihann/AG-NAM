from dataclasses import dataclass

from agnam.revision.sensitivity import (
    build_sensitivity_configurations,
    candidate_k_levels,
)


def test_sensitivity_matrix_is_locked_and_complete():
    configs = build_sensitivity_configurations()

    assert len(configs) == 18

    threshold = [c for c in configs if c.family == "threshold_grid"]
    runs = [c for c in configs if c.family == "run_count"]
    k_rows = [c for c in configs if c.family == "candidate_k"]
    support = [c for c in configs if c.family == "reference_support"]

    assert len(threshold) == 9
    assert len(runs) == 3
    assert len(k_rows) == 3
    assert len(support) == 3

    assert {
        (c.selection_threshold, c.isr_threshold)
        for c in threshold
    } == {
        (s, i)
        for s in (0.4, 0.6, 0.8)
        for i in (0.4, 0.6, 0.8)
    }

    assert {c.n_discovery_runs for c in runs} == {3, 5, 10}
    assert {c.k_mode for c in k_rows} == {"half", "locked", "double"}
    assert {c.reference_size for c in support} == {128, 256, 512}


def test_candidate_k_levels():
    # S1: p=20 -> P=190; locked K=19.
    levels = candidate_k_levels(
        n_features=20,
        locked_k=19,
    )

    assert levels == {
        "half": 10,
        "locked": 19,
        "double": 38,
    }

    # Ensure the doubled value is capped by the interaction universe.
    levels_small = candidate_k_levels(
        n_features=4,
        locked_k=4,
    )
    assert levels_small["double"] == 6
