from agnam.revision.isr_novelty_runner import _find_spec


def test_find_spec_uses_locked_schedule():
    protocol, spec = _find_spec("S1", 1)
    assert spec.dataset_seed == 10000
    assert spec.outer_split_seed == 20000
    assert spec.discovery_base_seed == 40000
    assert protocol.selection_threshold == 0.60
    assert protocol.isr_threshold == 0.60
