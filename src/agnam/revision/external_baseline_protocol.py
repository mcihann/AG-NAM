from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence


REQUIRED_BASELINES = (
    "friedman_h",
    "nit",
    "additive_groves",
    "irf",
    "sian",
)

SECONDARY_BASELINES = (
    "nid",
)


@dataclass(frozen=True)
class ExternalBaselineSpec:
    name: str
    requested_by_reviewer: bool
    implementation_mode: str
    output_type: str
    isolated_environment: bool


BASELINE_SPECS = (
    ExternalBaselineSpec(
        name="friedman_h",
        requested_by_reviewer=True,
        implementation_mode="python_native",
        output_type="full_pair_ranking",
        isolated_environment=False,
    ),
    ExternalBaselineSpec(
        name="nit",
        requested_by_reviewer=True,
        implementation_mode="paper_reproduction",
        output_type="scored_pair_or_structure",
        isolated_environment=False,
    ),
    ExternalBaselineSpec(
        name="additive_groves",
        requested_by_reviewer=True,
        implementation_mode="external_executable",
        output_type="pairwise_interaction_tests",
        isolated_environment=True,
    ),
    ExternalBaselineSpec(
        name="irf",
        requested_by_reviewer=True,
        implementation_mode="authors_implementation",
        output_type="scored_interaction_sets",
        isolated_environment=True,
    ),
    ExternalBaselineSpec(
        name="sian",
        requested_by_reviewer=True,
        implementation_mode="authors_python_package",
        output_type="scored_interaction_sets",
        isolated_environment=False,
    ),
    ExternalBaselineSpec(
        name="nid",
        requested_by_reviewer=False,
        implementation_mode="paper_reference_python",
        output_type="full_pair_ranking",
        isolated_environment=False,
    ),
)


def unordered_pairs(feature_names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    names = tuple(str(name) for name in feature_names)
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique.")
    return tuple(
        (first, second)
        for first, second in combinations(names, 2)
    )


def project_scored_interaction_sets_to_pairs(
    feature_names: Sequence[str],
    scored_sets: Iterable[tuple[Iterable[str], float]],
    *,
    floor_score: float = 0.0,
) -> dict[tuple[str, str], float]:
    universe = unordered_pairs(feature_names)
    scores = {pair: float(floor_score) for pair in universe}
    allowed = set(feature_names)

    for raw_set, raw_score in scored_sets:
        members = tuple(sorted(set(str(x) for x in raw_set)))
        if len(members) < 2:
            continue
        unknown = set(members).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown features in interaction set: {sorted(unknown)}")

        score = float(raw_score)
        for pair in combinations(members, 2):
            pair = tuple(sorted(pair))
            scores[pair] = max(scores[pair], score)

    return scores


def validate_full_pair_scores(
    feature_names: Sequence[str],
    scores: Mapping[tuple[str, str], float],
) -> None:
    expected = set(unordered_pairs(feature_names))
    observed = {
        tuple(sorted((str(pair[0]), str(pair[1]))))
        for pair in scores
    }

    if expected != observed:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ValueError(
            f"Pair universe mismatch. Missing={missing[:5]}, extra={extra[:5]}"
        )


def deterministic_rank_pair_scores(
    scores: Mapping[tuple[str, str], float],
) -> tuple[tuple[str, str, float], ...]:
    normalized = [
        (min(str(a), str(b)), max(str(a), str(b)), float(score))
        for (a, b), score in scores.items()
    ]
    normalized.sort(key=lambda row: (-row[2], row[0], row[1]))
    return tuple(normalized)
