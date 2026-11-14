from agnam.revision.external_recovery_runner import METHODS


def test_core_external_methods_are_locked():
    assert METHODS == (
        "friedman_h_gbdt",
        "nid_shared_mlp",
        "sian_fid_shared_mlp",
    )
