from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
import copy
import threading
import time

import numpy as np
import psutil
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import partial_dependence
from sklearn.metrics import (
    average_precision_score,
    ndcg_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


Pair = tuple[str, str]


@dataclass(frozen=True)
class PairRankingMetrics:
    interaction_auprc: float
    interaction_auroc: float
    ndcg: float
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    precision_at_true_count: float
    recall_at_true_count: float
    f1_at_true_count: float


@dataclass(frozen=True)
class SharedMLPTrainingResult:
    model: object
    scaler: StandardScaler
    best_epoch: int
    best_val_auroc: float
    training_seconds: float
    rss_start_mib: float
    rss_end_mib: float
    rss_peak_mib: float
    cuda_peak_allocated_mib: float


class PeakRSSMonitor:
    def __init__(self, interval_seconds: float = 0.02):
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread = None
        self.start_bytes = 0
        self.end_bytes = 0
        self.peak_bytes = 0

    def __enter__(self):
        process = psutil.Process()
        self.start_bytes = process.memory_info().rss
        self.peak_bytes = self.start_bytes

        def _worker():
            proc = psutil.Process()
            while not self._stop.is_set():
                try:
                    self.peak_bytes = max(
                        self.peak_bytes,
                        proc.memory_info().rss,
                    )
                except psutil.Error:
                    pass
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(
            target=_worker,
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.end_bytes = psutil.Process().memory_info().rss
        self.peak_bytes = max(self.peak_bytes, self.end_bytes)

    @staticmethod
    def _mib(value: int) -> float:
        return float(value) / (1024.0 ** 2)

    @property
    def start_mib(self) -> float:
        return self._mib(self.start_bytes)

    @property
    def end_mib(self) -> float:
        return self._mib(self.end_bytes)

    @property
    def peak_mib(self) -> float:
        return self._mib(self.peak_bytes)


def canonical_pair(a: str, b: str) -> Pair:
    a = str(a)
    b = str(b)
    if a == b:
        raise ValueError("An interaction pair requires two distinct features.")
    return (a, b) if a < b else (b, a)


def full_pair_universe(feature_names) -> tuple[Pair, ...]:
    names = tuple(str(x) for x in feature_names)
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique.")
    return tuple(
        canonical_pair(a, b)
        for a, b in combinations(names, 2)
    )


def deterministic_ranking(
    pair_scores: dict[Pair, float],
) -> tuple[tuple[str, str, float], ...]:
    rows = [
        (
            canonical_pair(a, b)[0],
            canonical_pair(a, b)[1],
            float(score),
        )
        for (a, b), score in pair_scores.items()
    ]
    rows.sort(key=lambda row: (-row[2], row[0], row[1]))
    return tuple(rows)


def _set_metrics(
    selected: set[Pair],
    truth: set[Pair],
) -> tuple[float, float, float]:
    tp = len(selected.intersection(truth))
    fp = len(selected.difference(truth))
    fn = len(truth.difference(selected))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return float(precision), float(recall), float(f1)


def evaluate_pair_ranking(
    *,
    feature_names,
    pair_scores: dict[Pair, float],
    true_pairs,
    candidate_k: int,
) -> PairRankingMetrics:
    universe = full_pair_universe(feature_names)
    expected = set(universe)

    normalized_scores = {
        canonical_pair(*pair): float(score)
        for pair, score in pair_scores.items()
    }

    if set(normalized_scores) != expected:
        missing = sorted(expected.difference(normalized_scores))
        extra = sorted(set(normalized_scores).difference(expected))
        raise ValueError(
            "Pair-score universe mismatch. "
            f"Missing={missing[:5]}, extra={extra[:5]}"
        )

    truth = {
        canonical_pair(*pair)
        for pair in true_pairs
    }

    ranking = deterministic_ranking(normalized_scores)
    ranked_pairs = [
        canonical_pair(a, b)
        for a, b, _ in ranking
    ]

    labels = np.asarray(
        [1 if pair in truth else 0 for pair in universe],
        dtype=np.int64,
    )
    scores = np.asarray(
        [normalized_scores[pair] for pair in universe],
        dtype=np.float64,
    )

    auprc = float(average_precision_score(labels, scores))
    auroc = float(roc_auc_score(labels, scores))
    ndcg = float(
        ndcg_score(
            labels.reshape(1, -1),
            scores.reshape(1, -1),
        )
    )

    k = min(int(candidate_k), len(ranked_pairs))
    selected_k = set(ranked_pairs[:k])
    p_k, r_k, f_k = _set_metrics(selected_k, truth)

    n_true = min(len(truth), len(ranked_pairs))
    selected_true_count = set(ranked_pairs[:n_true])
    p_t, r_t, f_t = _set_metrics(
        selected_true_count,
        truth,
    )

    return PairRankingMetrics(
        interaction_auprc=auprc,
        interaction_auroc=auroc,
        ndcg=ndcg,
        precision_at_k=p_k,
        recall_at_k=r_k,
        f1_at_k=f_k,
        precision_at_true_count=p_t,
        recall_at_true_count=r_t,
        f1_at_true_count=f_t,
    )


def _cuda_begin(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_end(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    torch.cuda.synchronize(device)
    return float(
        torch.cuda.max_memory_allocated(device)
        / (1024.0 ** 2)
    )


def train_shared_sian_mlp(
    X_fit,
    y_fit,
    X_val,
    y_val,
    *,
    seed: int,
    device: torch.device,
    hidden_sizes=(128, 64, 32),
    batch_size: int = 32,
    max_epochs: int = 300,
    learning_rate: float = 5e-3,
    l1_coefficient: float = 5e-5,
    patience: int = 20,
) -> SharedMLPTrainingResult:
    from sian.models.models import MLP

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(
        np.asarray(X_fit, dtype=np.float32)
    ).astype(np.float32)
    X_val_scaled = scaler.transform(
        np.asarray(X_val, dtype=np.float32)
    ).astype(np.float32)

    y_fit_arr = np.asarray(y_fit, dtype=np.float32).reshape(-1, 1)
    y_val_arr = np.asarray(y_val, dtype=np.float32).reshape(-1)

    sizes = [
        X_fit_scaled.shape[1],
        *[int(x) for x in hidden_sizes],
        1,
    ]
    model = MLP(sizes).to(device)

    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr=float(learning_rate),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    train_ds = TensorDataset(
        torch.from_numpy(X_fit_scaled),
        torch.from_numpy(y_fit_arr),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    X_val_tensor = torch.from_numpy(X_val_scaled).to(device)

    best_metric = -np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    _cuda_begin(device)
    started = time.perf_counter()

    with PeakRSSMonitor() as memory:
        for epoch in range(1, int(max_epochs) + 1):
            model.train()

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad(set_to_none=True)
                dnn_logits, gam_logits, _ = model(xb)
                logits = dnn_logits + gam_logits

                loss = loss_fn(logits, yb)
                if l1_coefficient > 0:
                    l1 = torch.zeros((), device=device)
                    for parameter in model.parameters():
                        l1 = l1 + parameter.abs().sum()
                    loss = loss + float(l1_coefficient) * l1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()

            model.eval()
            with torch.no_grad():
                dnn_val, gam_val, _ = model(X_val_tensor)
                logits_val = (dnn_val + gam_val)[:, 0]
                probabilities = torch.sigmoid(logits_val)
                probabilities = probabilities.detach().cpu().numpy()

            try:
                val_auroc = float(
                    roc_auc_score(y_val_arr, probabilities)
                )
            except ValueError:
                val_auroc = 0.0

            if val_auroc > best_metric + 1e-8:
                best_metric = val_auroc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= int(patience):
                break

        model.load_state_dict(best_state)
        model.eval()

    elapsed = time.perf_counter() - started
    cuda_peak = _cuda_end(device)

    # Current SIAN helper code expects this attribute in several places.
    model.device = device

    return SharedMLPTrainingResult(
        model=model,
        scaler=scaler,
        best_epoch=int(best_epoch),
        best_val_auroc=float(best_metric),
        training_seconds=float(elapsed),
        rss_start_mib=memory.start_mib,
        rss_end_mib=memory.end_mib,
        rss_peak_mib=memory.peak_mib,
        cuda_peak_allocated_mib=cuda_peak,
    )


def nid_pair_scores_from_model(
    model,
    feature_names,
) -> dict[Pair, float]:
    linears = list(model.dnn.hiddens)
    if len(linears) < 2:
        raise ValueError(
            "NID requires at least one hidden layer and one output layer."
        )

    weights = [
        layer.weight.detach().abs().cpu().numpy().astype(np.float64)
        for layer in linears
    ]

    first = weights[0]

    # Absolute downstream influence from each first-hidden unit to output.
    downstream = weights[-1]
    if downstream.shape[0] != 1:
        downstream = downstream.sum(axis=0, keepdims=True)

    influence = downstream.reshape(-1)

    for weight in reversed(weights[1:-1]):
        influence = influence @ weight

    if influence.shape[0] != first.shape[0]:
        raise ValueError(
            "Unexpected NID weight geometry: "
            f"influence={influence.shape}, first={first.shape}"
        )

    names = tuple(str(x) for x in feature_names)
    scores: dict[Pair, float] = {}

    for j, k in combinations(range(len(names)), 2):
        pair_evidence = np.minimum(
            first[:, j],
            first[:, k],
        )
        score = float(np.sum(influence * pair_evidence))
        scores[canonical_pair(names[j], names[k])] = score

    return scores


def sian_fid_pair_scores(
    *,
    model,
    scaler: StandardScaler,
    X_score,
    feature_names,
    seed: int,
    device: torch.device,
    agg_k: int = 32,
    explainer_batch_size: int = 512,
) -> tuple[dict[Pair, float], float, float, float, float]:
    from sian.interpret.basic_wrappers import MixedModelWrapperTorch
    from sian.interpret.explainer2 import (
        JamArchipelago,
        unmasked_FID_Hyperparameters,
    )

    X_scaled = scaler.transform(
        np.asarray(X_score, dtype=np.float32)
    ).astype(np.float32)

    rng = np.random.default_rng(int(seed))
    support_n = min(int(agg_k), len(X_scaled))
    support_idx = rng.choice(
        len(X_scaled),
        size=support_n,
        replace=False,
    )
    support = X_scaled[support_idx]

    names = tuple(str(x) for x in feature_names)
    p = len(names)

    grouped = {
        "D": p,
        "D0": p,
        "C": 2,
    }
    for index in range(p):
        grouped[index] = [index]

    fid_hypers = unmasked_FID_Hyperparameters(
        "triangle_marginal",
        "binary_classification",
        "old_arch_inter_score",
        [
            "inc_inter_score",
            "rem_inter_score",
            "old_arch_inter_score",
        ],
        device,
        grouped,
    )

    wrapped = MixedModelWrapperTorch(model, device)
    explainer = JamArchipelago(wrapped, fid_hypers)
    explainer.batch_size = int(explainer_batch_size)
    explainer.verbose = False

    pair_indices = tuple(combinations(range(p), 2))

    _cuda_begin(device)
    started = time.perf_counter()
    with PeakRSSMonitor() as memory:
        raw = explainer.get_explanation_scores(
            support,
            support_n,
            pair_indices,
        )
    elapsed = time.perf_counter() - started
    cuda_peak = _cuda_end(device)

    scores: dict[Pair, float] = {}
    for j, k in pair_indices:
        value = raw[(j, k)].get(
            "old_arch_inter_score",
            0.0,
        )
        if not np.isfinite(value):
            value = 0.0
        scores[canonical_pair(names[j], names[k])] = float(value)

    return (
        scores,
        float(elapsed),
        memory.start_mib,
        memory.end_mib,
        max(memory.peak_mib, memory.start_mib),
        float(cuda_peak),
    )


def _center(values):
    values = np.asarray(values, dtype=np.float64)
    return values - np.mean(values)


def friedman_h_pair_scores(
    *,
    X_fit,
    y_fit,
    X_score,
    feature_names,
    seed: int,
    grid_resolution: int = 10,
    allow_degenerate_grid_fallback: bool = False,
    grid_fallback_features: list[str] | None = None,
) -> tuple[
    dict[Pair, float],
    float,
    float,
    float,
    float,
]:
    X_fit = np.asarray(X_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit)
    X_score = np.asarray(X_score, dtype=np.float64)
    names = tuple(str(x) for x in feature_names)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=int(seed),
    )

    with PeakRSSMonitor() as memory:
        fit_started = time.perf_counter()
        model.fit(X_fit, y_fit)
        fit_seconds = time.perf_counter() - fit_started

        scoring_started = time.perf_counter()

        single = {}
        single_grid = {}
        fallback_indices = set()

        for j in range(len(names)):
            pd_kwargs = dict(
                response_method="decision_function",
                method="recursion",
                percentiles=(0.05, 0.95),
                grid_resolution=int(grid_resolution),
                kind="average",
            )
            try:
                pd_result = partial_dependence(model, X_score, [j], **pd_kwargs)
            except ValueError as exc:
                if (not allow_degenerate_grid_fallback
                        or "percentiles are too close" not in str(exc)):
                    raise
                # Only the affected axis changes. Use a deterministic subset of
                # observed scoring-support values, including both endpoints.
                # No labels, truth, or test-set information are involved.
                unique = np.unique(X_score[:, j])
                if not len(unique) or not np.isfinite(unique).all():
                    raise ValueError("Invalid scoring support for H grid") from exc
                indices = np.linspace(0, len(unique) - 1,
                                      min(len(unique), int(grid_resolution)), dtype=int)
                fallback_grid = unique[indices]
                pd_result = partial_dependence(
                    model, X_score, [j], custom_values={j: fallback_grid}, **pd_kwargs
                )
                fallback_indices.add(j)
                if grid_fallback_features is not None:
                    grid_fallback_features.append(names[j])
            single[j] = _center(pd_result["average"][0])
            single_grid[j] = np.asarray(
                pd_result["grid_values"][0],
                dtype=np.float64,
            )

        scores: dict[Pair, float] = {}

        for j, k in combinations(range(len(names)), 2):
            # Use exactly the single-axis grids for pairs involving a fallback.
            # Unaffected pairs retain the original sklearn call unchanged.
            pair_kwargs = {}
            pair_features = [(j, k)]
            if j in fallback_indices or k in fallback_indices:
                pair_kwargs["custom_values"] = {j: single_grid[j], k: single_grid[k]}
                # sklearn resolves custom_values against individual feature keys.
                pair_features = [j, k]
            pd_result = partial_dependence(
                model,
                X_score,
                pair_features,
                response_method="decision_function",
                method="recursion",
                percentiles=(0.05, 0.95),
                grid_resolution=int(grid_resolution),
                kind="average",
                **pair_kwargs,
            )

            surface = _center(pd_result["average"][0])
            grid_j = np.asarray(
                pd_result["grid_values"][0],
                dtype=np.float64,
            )
            grid_k = np.asarray(
                pd_result["grid_values"][1],
                dtype=np.float64,
            )

            main_j = np.interp(
                grid_j,
                single_grid[j],
                single[j],
            )
            main_k = np.interp(
                grid_k,
                single_grid[k],
                single[k],
            )

            residual = (
                surface
                - main_j[:, None]
                - main_k[None, :]
            )

            denominator = float(np.sum(surface ** 2))
            if denominator <= 1e-15:
                h_score = 0.0
            else:
                h2 = float(
                    np.sum(residual ** 2)
                    / denominator
                )
                h_score = sqrt(max(0.0, h2))

            scores[
                canonical_pair(names[j], names[k])
            ] = float(h_score)

        scoring_seconds = time.perf_counter() - scoring_started

    return (
        scores,
        float(fit_seconds),
        float(scoring_seconds),
        memory.start_mib,
        memory.end_mib,
        memory.peak_mib,
    )
