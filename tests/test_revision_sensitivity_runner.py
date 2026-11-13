from agnam.revision.sensitivity import build_sensitivity_configurations


def test_reference_rows_match_submitted_configuration():
    configs = build_sensitivity_configurations()

    reference_rows = [c for c in configs if c.is_reference]
    assert len(reference_rows) == 4

    for row in reference_rows:
        assert row.selection_threshold == 0.60
        assert row.isr_threshold == 0.60
        assert row.n_discovery_runs == 5
        assert row.k_mode == "locked"
        assert row.reference_size == 512
