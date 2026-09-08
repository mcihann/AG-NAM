from dataclasses import dataclass

import pytest

import agnam.benchmarking.realx_discovery as realx_discovery


@dataclass
class FakeScoreResult:
    ranked_interactions: tuple


@dataclass
class FakeRun:
    k: int
    score_result: FakeScoreResult


@dataclass
class FakeBaseResult:
    runs: tuple
    selection: object


def make_fake_result(
    *,
    n_runs: int = 5,
    n_ranked: int = 12,
    original_k: int = 5,
):
    ranking = tuple(
        range(
            n_ranked
        )
    )

    runs = tuple(
        FakeRun(
            k=original_k,
            score_result=(
                FakeScoreResult(
                    ranked_interactions=(
                        ranking
                    )
                )
            ),
        )
        for _ in range(
            n_runs
        )
    )

    return FakeBaseResult(
        runs=runs,
        selection="old",
    )


def test_validate_realx_candidate_k_accepts_locked_value():
    assert (
        realx_discovery
        .validate_realx_candidate_k(
            8,
            available_ranked_pairs=36,
        )
        == 8
    )


def test_validate_realx_candidate_k_rejects_excess():
    with pytest.raises(
        ValueError
    ):
        realx_discovery.validate_realx_candidate_k(
            37,
            available_ranked_pairs=36,
        )


def test_candidate_k_override_updates_all_runs(
    monkeypatch,
):
    base = make_fake_result(
        n_runs=5,
        n_ranked=12,
        original_k=5,
    )

    captured = {}

    def fake_summary(
        *,
        rankings,
        k,
        threshold,
    ):
        captured[
            "n_rankings"
        ] = len(
            rankings
        )

        captured[
            "k"
        ] = k

        captured[
            "threshold"
        ] = threshold

        return "new-selection"

    monkeypatch.setattr(
        realx_discovery
        .discovery,
        "summarize_selection_reproducibility",
        fake_summary,
    )

    updated = (
        realx_discovery
        .override_discovery_candidate_k(
            base,
            candidate_k=8,
            selection_threshold=0.60,
        )
    )

    assert (
        len(
            updated.runs
        )
        == 5
    )

    assert all(
        run.k == 8
        for run in (
            updated.runs
        )
    )

    assert (
        updated.selection
        == "new-selection"
    )

    assert (
        captured[
            "n_rankings"
        ]
        == 5
    )

    assert (
        captured[
            "k"
        ]
        == 8
    )

    assert (
        captured[
            "threshold"
        ]
        == 0.60
    )