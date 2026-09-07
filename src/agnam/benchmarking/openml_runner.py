from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from agnam.benchmarking.external_baselines import (
    ExternalBaselineProtocol,
    build_external_baseline_schedule,
    catboost_locked_parameters,
    ebm_locked_parameters,
    external_interaction_budget,
)
from agnam.benchmarking.openml_data import (
    LoadedOpenMLTask,
    audit_loaded_openml_task,
    load_openml_task,
)
from agnam.benchmarking.openml_protocol import (
    OPENML_TASKS,
    OpenMLBenchmarkProtocol,
    OpenMLExecutionSpec,
    OpenMLTaskSpec,
    build_openml_schedule,
)
from agnam.benchmarking.synthetic_protocol import (
    candidate_count_from_feature_count,
)
from agnam.benchmarking.synthetic_runner import (
    SyntheticModelConfig,
    normalize_pairs,
    sample_random_pairs,
)
from agnam.data.preprocessing import (
    TabularPreprocessor,
)
from agnam.models.agnam import (
    AGNAM,
)
from agnam.models.main_effect_nam import (
    MainEffectNAM,
)
from agnam.training.agnam_trainer import (
    AGNAMTrainingConfig,
    compute_agnam_binary_metrics,
    predict_agnam_probabilities,
    train_agnam,
)
from agnam.training.discovery import (
    ReproducibleDiscoveryResult,
    run_reproducible_interaction_discovery,
)
from agnam.training.isr import (
    ISRConfig,
    ISRDiscoveryResult,
    evaluate_isr_for_discovery,
)
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
    compute_binary_metrics,
    predict_probabilities,
    train_main_effect_nam,
)
from agnam.training.proposer_trainer import (
    ProposerTrainingConfig,
)
from agnam.utils.reproducibility import (
    get_device,
    seed_everything,
)


Pair = tuple[str, str]


@dataclass(frozen=True)
class OpenMLModelEvaluation:
    auroc: float
    auprc: float
    balanced_accuracy: float
    f1: float

    best_iteration: int


@dataclass
class OpenMLFinalPredictionData:
    X_train_raw: pd.DataFrame
    X_validation_raw: pd.DataFrame
    X_test_raw: pd.DataFrame

    train_data: object
    validation_data: object
    test_data: object

    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray


@dataclass
class OpenMLBenchmarkRecord:
    task_id: int
    dataset_id: int
    dataset_name: str
    feature_type: str

    n_samples: int
    n_features: int
    n_numeric_features: int
    n_categorical_features: int

    n_development: int
    n_test: int

    positive_label: str
    negative_label: str

    development_positive_fraction: float
    test_positive_fraction: float

    missing_fraction: float

    candidate_k: int
    ebm_interaction_budget: int

    mean_pairwise_jaccard: float

    n_selection_stable: int
    n_isr_retained: int

    isr_sparsification: float

    main_auroc: float
    main_auprc: float
    main_balanced_accuracy: float
    main_f1: float

    agnam_auroc: float
    agnam_auprc: float
    agnam_balanced_accuracy: float
    agnam_f1: float

    random_pair_auroc: float
    random_pair_auprc: float

    no_isr_auroc: float
    no_isr_auprc: float

    single_run_auroc: float
    single_run_auprc: float

    ebm_auroc: float
    ebm_auprc: float
    ebm_balanced_accuracy: float
    ebm_f1: float

    catboost_auroc: float
    catboost_auprc: float
    catboost_balanced_accuracy: float
    catboost_f1: float

    delta_agnam_main_auroc: float
    delta_agnam_main_auprc: float

    delta_agnam_random_auroc: float
    delta_agnam_ebm_auroc: float
    delta_agnam_catboost_auroc: float

    max_decomposition_error: float

    main_best_iteration: int
    agnam_best_iteration: int
    ebm_best_iteration: int
    catboost_best_iteration: int

    runtime_seconds: float
    ebm_runtime_seconds: float
    catboost_runtime_seconds: float


@dataclass
class SingleOpenMLBenchmarkResult:
    record: OpenMLBenchmarkRecord

    selection_pairs: tuple[
        Pair,
        ...
    ]

    isr_pairs: tuple[
        Pair,
        ...
    ]

    random_pairs: tuple[
        Pair,
        ...
    ]

    single_run_pairs: tuple[
        Pair,
        ...
    ]

    discovery: (
        ReproducibleDiscoveryResult
    )

    isr: ISRDiscoveryResult


def get_openml_task_spec(
    task_id: int,
) -> OpenMLTaskSpec:
    matches = [
        task
        for task in OPENML_TASKS
        if task.task_id
        == task_id
    ]

    if len(
        matches
    ) != 1:
        raise ValueError(
            f"Expected exactly one locked task "
            f"with task_id={task_id}."
        )

    return matches[
        0
    ]


def get_openml_execution_spec(
    task_id: int,
    *,
    protocol: (
        OpenMLBenchmarkProtocol
        | None
    ) = None,
) -> OpenMLExecutionSpec:
    if protocol is None:
        protocol = (
            OpenMLBenchmarkProtocol()
        )

    matches = [
        specification
        for specification
        in build_openml_schedule(
            protocol
        )
        if specification.task_id
        == task_id
    ]

    if len(
        matches
    ) != 1:
        raise ValueError(
            f"Expected exactly one execution "
            f"specification for task_id={task_id}."
        )

    return matches[
        0
    ]


def get_external_execution_spec(
    task_id: int,
    *,
    protocol: (
        ExternalBaselineProtocol
        | None
    ) = None,
):
    if protocol is None:
        protocol = (
            ExternalBaselineProtocol()
        )

    matches = [
        specification
        for specification
        in build_external_baseline_schedule(
            protocol
        )
        if specification.task_id
        == task_id
    ]

    if len(
        matches
    ) != 1:
        raise ValueError(
            f"Expected exactly one external-baseline "
            f"specification for task_id={task_id}."
        )

    return matches[
        0
    ]


def baseline_feature_types(
    categorical_indicator,
) -> tuple[
    str,
    ...
]:
    """
    Explicit feature-type mapping for EBM.

    OpenML categorical:
        nominal

    OpenML numeric:
        continuous
    """
    return tuple(
        "nominal"
        if bool(
            is_categorical
        )
        else "continuous"
        for is_categorical
        in categorical_indicator
    )


def categorical_feature_indices(
    categorical_indicator,
) -> tuple[
    int,
    ...
]:
    return tuple(
        index
        for index, is_categorical
        in enumerate(
            categorical_indicator
        )
        if bool(
            is_categorical
        )
    )


def prepare_external_baseline_frame(
    X: pd.DataFrame,
    categorical_indicator,
    *,
    missing_category: str = (
        "__AGNAM_MISSING__"
    ),
) -> pd.DataFrame:
    """
    Build deterministic raw-data representation for EBM/CatBoost.

    Numeric:
        float64, NaN retained.

    Categorical:
        explicit missing token + string representation.
    """
    frame = X.copy()

    indicator = tuple(
        bool(
            value
        )
        for value
        in categorical_indicator
    )

    if len(
        indicator
    ) != frame.shape[
        1
    ]:
        raise ValueError(
            "categorical_indicator length does not "
            "match predictor count."
        )

    for index, column in enumerate(
        frame.columns
    ):
        if indicator[
            index
        ]:
            series = (
                frame[
                    column
                ]
                .astype(
                    object
                )
            )

            series = series.where(
                series.notna(),
                missing_category,
            )

            frame[
                column
            ] = series.map(
                str
            )

        else:
            frame[
                column
            ] = pd.to_numeric(
                frame[
                    column
                ],
                errors="coerce",
            ).astype(
                np.float64
            )

    return frame


def _build_discovery_configs(
    execution_spec: OpenMLExecutionSpec,
    model_config: SyntheticModelConfig,
):
    nam_config = (
        NAMTrainingConfig(
            learning_rate=(
                model_config
                .discovery_learning_rate
            ),
            weight_decay=(
                model_config
                .discovery_weight_decay
            ),
            batch_size=(
                model_config
                .discovery_batch_size
            ),
            max_epochs=(
                model_config
                .nam_discovery_max_epochs
            ),
            patience=(
                model_config
                .nam_discovery_patience
            ),
            seed=(
                execution_spec
                .discovery_base_seed
            ),
            deterministic=True,
        )
    )

    proposer_config = (
        ProposerTrainingConfig(
            learning_rate=(
                model_config
                .discovery_learning_rate
            ),
            weight_decay=(
                model_config
                .discovery_weight_decay
            ),
            batch_size=(
                model_config
                .discovery_batch_size
            ),
            max_epochs=(
                model_config
                .proposer_max_epochs
            ),
            patience=(
                model_config
                .proposer_patience
            ),
            seed=(
                execution_spec
                .discovery_base_seed
            ),
            deterministic=True,
        )
    )

    return (
        nam_config,
        proposer_config,
    )


def _build_isr_config(
    benchmark_protocol: (
        OpenMLBenchmarkProtocol
    ),
    model_config: (
        SyntheticModelConfig
    ),
) -> ISRConfig:
    return ISRConfig(
        reference_size=(
            benchmark_protocol
            .reference_size
        ),
        reference_seed=(
            benchmark_protocol
            .reference_seed
        ),
        isr_threshold=(
            benchmark_protocol
            .isr_threshold
        ),
        feature_embedding_dim=(
            model_config
            .interaction_embedding_dim
        ),
        hidden_dim=(
            model_config
            .interaction_hidden_dim
        ),
        depth=(
            model_config
            .interaction_depth
        ),
        dropout=(
            model_config
            .interaction_dropout
        ),
        learning_rate=(
            model_config
            .discovery_learning_rate
        ),
        weight_decay=(
            model_config
            .discovery_weight_decay
        ),
        batch_size=(
            model_config
            .discovery_batch_size
        ),
        max_epochs=(
            model_config
            .pairwise_max_epochs
        ),
        patience=(
            model_config
            .pairwise_patience
        ),
        min_delta=1e-5,
        surface_batch_size=8192,
    )


def prepare_final_prediction_data(
    loaded: LoadedOpenMLTask,
    *,
    final_split_seed: int,
    validation_fraction: float,
) -> OpenMLFinalPredictionData:
    X_development = (
        loaded
        .X_development
    )

    y_development = (
        loaded
        .y_development
    )

    X_test = (
        loaded
        .X_test
    )

    y_test = (
        loaded
        .y_test
    )

    indices = np.arange(
        len(
            X_development
        )
    )

    train_indices, validation_indices = (
        train_test_split(
            indices,
            test_size=(
                validation_fraction
            ),
            random_state=(
                final_split_seed
            ),
            stratify=(
                y_development
            ),
        )
    )

    X_train = (
        X_development
        .iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )

    X_validation = (
        X_development
        .iloc[
            validation_indices
        ]
        .reset_index(
            drop=True
        )
    )

    y_train = (
        y_development[
            train_indices
        ]
    )

    y_validation = (
        y_development[
            validation_indices
        ]
    )

    preprocessor = (
        TabularPreprocessor()
    )

    train_data = (
        preprocessor
        .fit_transform(
            X_train
        )
    )

    validation_data = (
        preprocessor
        .transform(
            X_validation
        )
    )

    test_data = (
        preprocessor
        .transform(
            X_test
        )
    )

    return OpenMLFinalPredictionData(
        X_train_raw=(
            X_train
        ),
        X_validation_raw=(
            X_validation
        ),
        X_test_raw=(
            X_test
        ),
        train_data=(
            train_data
        ),
        validation_data=(
            validation_data
        ),
        test_data=(
            test_data
        ),
        y_train=(
            np.asarray(
                y_train,
                dtype=np.int64,
            )
        ),
        y_validation=(
            np.asarray(
                y_validation,
                dtype=np.int64,
            )
        ),
        y_test=(
            np.asarray(
                y_test,
                dtype=np.int64,
            )
        ),
    )


def _train_main_nam(
    data: OpenMLFinalPredictionData,
    *,
    seed: int,
    model_config: SyntheticModelConfig,
    device: torch.device,
) -> OpenMLModelEvaluation:
    seed_everything(
        seed
    )

    model = MainEffectNAM(
        feature_specs=(
            data
            .train_data
            .feature_specs
        ),
        hidden_dim=(
            model_config
            .nam_hidden_dim
        ),
        depth=(
            model_config
            .nam_depth
        ),
        dropout=(
            model_config
            .nam_dropout
        ),
        categorical_embedding_dim=(
            model_config
            .categorical_embedding_dim
        ),
    )

    config = (
        NAMTrainingConfig(
            learning_rate=(
                model_config
                .final_learning_rate
            ),
            weight_decay=(
                model_config
                .final_weight_decay
            ),
            batch_size=(
                model_config
                .final_batch_size
            ),
            max_epochs=(
                model_config
                .final_max_epochs
            ),
            patience=(
                model_config
                .final_patience
            ),
            seed=(
                seed
            ),
            deterministic=True,
        )
    )

    result = (
        train_main_effect_nam(
            model=model,
            train_data=(
                data.train_data
            ),
            train_y=(
                data.y_train
            ),
            val_data=(
                data.validation_data
            ),
            val_y=(
                data.y_validation
            ),
            config=(
                config
            ),
            device=(
                device
            ),
        )
    )

    probabilities = (
        predict_probabilities(
            model=model,
            transformed=(
                data.test_data
            ),
            device=(
                device
            ),
        )
    )

    metrics = (
        compute_binary_metrics(
            y_true=(
                data.y_test
            ),
            probabilities=(
                probabilities
            ),
        )
    )

    return OpenMLModelEvaluation(
        auroc=float(
            metrics[
                "auroc"
            ]
        ),
        auprc=float(
            metrics[
                "auprc"
            ]
        ),
        balanced_accuracy=float(
            metrics[
                "balanced_accuracy"
            ]
        ),
        f1=float(
            metrics[
                "f1"
            ]
        ),
        best_iteration=int(
            result.best_epoch
            + 1
        ),
    )


def _train_agnam_model(
    data: OpenMLFinalPredictionData,
    interaction_pairs: tuple[
        Pair,
        ...
    ],
    *,
    seed: int,
    model_config: SyntheticModelConfig,
    device: torch.device,
) -> tuple[
    OpenMLModelEvaluation,
    float,
]:
    seed_everything(
        seed
    )

    model = AGNAM(
        feature_specs=(
            data
            .train_data
            .feature_specs
        ),
        interaction_pairs=(
            interaction_pairs
        ),
        main_hidden_dim=(
            model_config
            .nam_hidden_dim
        ),
        main_depth=(
            model_config
            .nam_depth
        ),
        main_dropout=(
            model_config
            .nam_dropout
        ),
        categorical_embedding_dim=(
            model_config
            .categorical_embedding_dim
        ),
        interaction_embedding_dim=(
            model_config
            .interaction_embedding_dim
        ),
        interaction_hidden_dim=(
            model_config
            .interaction_hidden_dim
        ),
        interaction_depth=(
            model_config
            .interaction_depth
        ),
        interaction_dropout=(
            model_config
            .interaction_dropout
        ),
    )

    config = (
        AGNAMTrainingConfig(
            learning_rate=(
                model_config
                .final_learning_rate
            ),
            weight_decay=(
                model_config
                .final_weight_decay
            ),
            batch_size=(
                model_config
                .final_batch_size
            ),
            max_epochs=(
                model_config
                .final_max_epochs
            ),
            patience=(
                model_config
                .final_patience
            ),
            seed=(
                seed
            ),
            deterministic=True,
        )
    )

    result = train_agnam(
        model=model,
        train_data=(
            data.train_data
        ),
        train_y=(
            data.y_train
        ),
        val_data=(
            data.validation_data
        ),
        val_y=(
            data.y_validation
        ),
        config=(
            config
        ),
        device=(
            device
        ),
    )

    probabilities = (
        predict_agnam_probabilities(
            model=model,
            transformed=(
                data.test_data
            ),
            device=(
                device
            ),
        )
    )

    metrics = (
        compute_agnam_binary_metrics(
            y_true=(
                data.y_test
            ),
            probabilities=(
                probabilities
            ),
        )
    )

    model.eval()

    with torch.no_grad():
        numeric = (
            torch.from_numpy(
                data
                .test_data
                .numeric
            )
            .to(
                device
            )
        )

        missing = (
            torch.from_numpy(
                data
                .test_data
                .numeric_missing
            )
            .to(
                device
            )
        )

        categorical = (
            torch.from_numpy(
                data
                .test_data
                .categorical
            )
            .to(
                device
            )
        )

        output = model(
            numeric=(
                numeric
            ),
            numeric_missing=(
                missing
            ),
            categorical=(
                categorical
            ),
        )

        reconstructed = (
            output.baseline
            + output
            .main_contributions
            .sum(
                dim=1
            )
            + output
            .interaction_contributions
            .sum(
                dim=1
            )
        )

        decomposition_error = float(
            torch.max(
                torch.abs(
                    output.logits
                    - reconstructed
                )
            )
            .detach()
            .cpu()
            .item()
        )

    evaluation = (
        OpenMLModelEvaluation(
            auroc=float(
                metrics[
                    "auroc"
                ]
            ),
            auprc=float(
                metrics[
                    "auprc"
                ]
            ),
            balanced_accuracy=float(
                metrics[
                    "balanced_accuracy"
                ]
            ),
            f1=float(
                metrics[
                    "f1"
                ]
            ),
            best_iteration=int(
                result.best_epoch
                + 1
            ),
        )
    )

    return (
        evaluation,
        decomposition_error,
    )


def _positive_probability(
    model,
    probabilities,
) -> np.ndarray:
    classes = np.asarray(
        model.classes_
    )

    matches = np.where(
        classes
        == 1
    )[0]

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            "Unable to identify positive-class "
            "probability column."
        )

    return np.asarray(
        probabilities[
            :,
            int(
                matches[
                    0
                ]
            ),
        ],
        dtype=np.float64,
    )


def _train_ebm(
    loaded: LoadedOpenMLTask,
    *,
    seed: int,
    baseline_protocol: (
        ExternalBaselineProtocol
    ),
) -> tuple[
    OpenMLModelEvaluation,
    float,
]:
    from interpret.glassbox import (
        ExplainableBoostingClassifier,
    )

    start = perf_counter()

    X_development = (
        prepare_external_baseline_frame(
            loaded
            .X_development,
            loaded
            .categorical_indicator,
        )
    )

    X_test = (
        prepare_external_baseline_frame(
            loaded
            .X_test,
            loaded
            .categorical_indicator,
        )
    )

    y_development = (
        loaded
        .y_development
    )

    y_test = (
        loaded
        .y_test
    )

    parameters = (
        ebm_locked_parameters(
            n_features=(
                X_development
                .shape[
                    1
                ]
            ),
            seed=(
                seed
            ),
            protocol=(
                baseline_protocol
            ),
        )
    )

    model = (
        ExplainableBoostingClassifier(
            feature_names=[
                str(
                    column
                )
                for column
                in X_development.columns
            ],
            feature_types=list(
                baseline_feature_types(
                    loaded
                    .categorical_indicator
                )
            ),
            **parameters,
        )
    )

    model.fit(
        X_development,
        y_development,
    )

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    positive_probabilities = (
        _positive_probability(
            model,
            probabilities,
        )
    )

    metrics = (
        compute_binary_metrics(
            y_true=(
                y_test
            ),
            probabilities=(
                positive_probabilities
            ),
        )
    )

    best_iteration = -1

    if hasattr(
        model,
        "best_iteration_",
    ):
        try:
            values = np.asarray(
                model
                .best_iteration_
            )

            if values.size > 0:
                best_iteration = int(
                    np.max(
                        values
                    )
                    + 1
                )

        except Exception:
            best_iteration = -1

    runtime = float(
        perf_counter()
        - start
    )

    return (
        OpenMLModelEvaluation(
            auroc=float(
                metrics[
                    "auroc"
                ]
            ),
            auprc=float(
                metrics[
                    "auprc"
                ]
            ),
            balanced_accuracy=float(
                metrics[
                    "balanced_accuracy"
                ]
            ),
            f1=float(
                metrics[
                    "f1"
                ]
            ),
            best_iteration=(
                best_iteration
            ),
        ),
        runtime,
    )


def _train_catboost(
    loaded: LoadedOpenMLTask,
    final_data: OpenMLFinalPredictionData,
    *,
    seed: int,
    baseline_protocol: (
        ExternalBaselineProtocol
    ),
) -> tuple[
    OpenMLModelEvaluation,
    float,
]:
    from catboost import (
        CatBoostClassifier,
        Pool,
    )

    start = perf_counter()

    X_train = (
        prepare_external_baseline_frame(
            final_data
            .X_train_raw,
            loaded
            .categorical_indicator,
        )
    )

    X_validation = (
        prepare_external_baseline_frame(
            final_data
            .X_validation_raw,
            loaded
            .categorical_indicator,
        )
    )

    X_test = (
        prepare_external_baseline_frame(
            final_data
            .X_test_raw,
            loaded
            .categorical_indicator,
        )
    )

    categorical_indices = list(
        categorical_feature_indices(
            loaded
            .categorical_indicator
        )
    )

    feature_names = [
        str(
            column
        )
        for column
        in X_train.columns
    ]

    train_pool = Pool(
        data=(
            X_train
        ),
        label=(
            final_data
            .y_train
        ),
        cat_features=(
            categorical_indices
        ),
        feature_names=(
            feature_names
        ),
    )

    validation_pool = Pool(
        data=(
            X_validation
        ),
        label=(
            final_data
            .y_validation
        ),
        cat_features=(
            categorical_indices
        ),
        feature_names=(
            feature_names
        ),
    )

    test_pool = Pool(
        data=(
            X_test
        ),
        cat_features=(
            categorical_indices
        ),
        feature_names=(
            feature_names
        ),
    )

    parameters = (
        catboost_locked_parameters(
            seed=(
                seed
            ),
            protocol=(
                baseline_protocol
            ),
        )
    )

    model = (
        CatBoostClassifier(
            **parameters
        )
    )

    model.fit(
        train_pool,
        eval_set=(
            validation_pool
        ),
        early_stopping_rounds=(
            baseline_protocol
            .catboost_early_stopping_rounds
        ),
        use_best_model=True,
        verbose=False,
    )

    probabilities = (
        model.predict_proba(
            test_pool
        )
    )

    positive_probabilities = (
        _positive_probability(
            model,
            probabilities,
        )
    )

    metrics = (
        compute_binary_metrics(
            y_true=(
                final_data
                .y_test
            ),
            probabilities=(
                positive_probabilities
            ),
        )
    )

    best_iteration = (
        model.get_best_iteration()
    )

    if best_iteration is None:
        best_iteration = -1

    best_iteration = int(
        best_iteration
    )

    if best_iteration >= 0:
        best_iteration += 1

    runtime = float(
        perf_counter()
        - start
    )

    return (
        OpenMLModelEvaluation(
            auroc=float(
                metrics[
                    "auroc"
                ]
            ),
            auprc=float(
                metrics[
                    "auprc"
                ]
            ),
            balanced_accuracy=float(
                metrics[
                    "balanced_accuracy"
                ]
            ),
            f1=float(
                metrics[
                    "f1"
                ]
            ),
            best_iteration=(
                best_iteration
            ),
        ),
        runtime,
    )


def build_real_world_pair_audit(
    result: SingleOpenMLBenchmarkResult,
) -> pd.DataFrame:
    selection_pairs = set(
        result.selection_pairs
    )

    isr_pairs = set(
        result.isr_pairs
    )

    random_pairs = set(
        result.random_pairs
    )

    single_run_pairs = set(
        result.single_run_pairs
    )

    all_pairs = set()

    all_pairs.update(
        selection_pairs
    )

    all_pairs.update(
        isr_pairs
    )

    all_pairs.update(
        random_pairs
    )

    all_pairs.update(
        single_run_pairs
    )

    selection_counts = {
        pair: 0
        for pair
        in all_pairs
    }

    for run in (
        result
        .discovery
        .runs
    ):
        for pair in (
            normalize_pairs(
                run.top_k_pairs
            )
        ):
            selection_counts[
                pair
            ] = (
                selection_counts
                .get(
                    pair,
                    0,
                )
                + 1
            )

    isr_score_lookup = {}

    for item in (
        result
        .isr
        .interactions
    ):
        pair = normalize_pairs(
            [
                (
                    item.feature_j,
                    item.feature_k,
                )
            ]
        )[0]

        score = getattr(
            item,
            "isr_score",
            np.nan,
        )

        try:
            score = float(
                score
            )

        except Exception:
            score = np.nan

        isr_score_lookup[
            pair
        ] = score

    rows = []

    n_runs = len(
        result
        .discovery
        .runs
    )

    for pair in sorted(
        all_pairs
    ):
        count = int(
            selection_counts
            .get(
                pair,
                0,
            )
        )

        rows.append(
            {
                "feature_j": (
                    pair[
                        0
                    ]
                ),
                "feature_k": (
                    pair[
                        1
                    ]
                ),
                "selection_count": (
                    count
                ),
                "selection_frequency": (
                    count
                    / n_runs
                    if n_runs
                    > 0
                    else np.nan
                ),
                "selection_stable": (
                    pair
                    in selection_pairs
                ),
                "isr_retained": (
                    pair
                    in isr_pairs
                ),
                "isr_score": (
                    isr_score_lookup
                    .get(
                        pair,
                        np.nan,
                    )
                ),
                "random_control": (
                    pair
                    in random_pairs
                ),
                "single_run_top_k": (
                    pair
                    in single_run_pairs
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def run_single_openml_benchmark(
    task_id: int,
    *,
    benchmark_protocol: (
        OpenMLBenchmarkProtocol
        | None
    ) = None,
    baseline_protocol: (
        ExternalBaselineProtocol
        | None
    ) = None,
    model_config: (
        SyntheticModelConfig
        | None
    ) = None,
    device: (
        torch.device
        | None
    ) = None,
    verbose: bool = True,
) -> SingleOpenMLBenchmarkResult:
    if benchmark_protocol is None:
        benchmark_protocol = (
            OpenMLBenchmarkProtocol()
        )

    if baseline_protocol is None:
        baseline_protocol = (
            ExternalBaselineProtocol()
        )

    if model_config is None:
        model_config = (
            SyntheticModelConfig()
        )

    if device is None:
        device = get_device()

    start_time = perf_counter()

    task_spec = (
        get_openml_task_spec(
            task_id
        )
    )

    execution_spec = (
        get_openml_execution_spec(
            task_id,
            protocol=(
                benchmark_protocol
            ),
        )
    )

    external_spec = (
        get_external_execution_spec(
            task_id,
            protocol=(
                baseline_protocol
            ),
        )
    )

    # =========================================================
    # LOAD + AUDIT
    # =========================================================

    loaded = (
        load_openml_task(
            task_spec,
            protocol=(
                benchmark_protocol
            ),
        )
    )

    audit = (
        audit_loaded_openml_task(
            loaded
        )
    )

    if not audit.audit_pass:
        raise RuntimeError(
            f"Task {task_id} failed the locked "
            "OpenML audit."
        )

    if verbose:
        print(
            "========================================"
        )

        print(
            "CANONICAL OPENML BENCHMARK"
        )

        print(
            "========================================"
        )

        print(
            f"Task: "
            f"{task_id}"
        )

        print(
            f"Dataset: "
            f"{loaded.actual_dataset_name}"
        )

        print(
            f"Feature type: "
            f"{audit.feature_type}"
        )

        print(
            f"Samples: "
            f"{audit.n_samples}"
        )

        print(
            f"Predictors: "
            f"{audit.n_predictors}"
        )

        print(
            f"Development / test: "
            f"{audit.n_development} / "
            f"{audit.n_test}"
        )

        print(
            f"Positive label: "
            f"{audit.positive_label}"
        )

        print(
            f"Device: "
            f"{device}"
        )

    X_development = (
        loaded
        .X_development
    )

    y_development = (
        loaded
        .y_development
    )

    # =========================================================
    # DISCOVERY
    # =========================================================

    if verbose:
        print(
            "\n=== INTERACTION DISCOVERY ==="
        )

    (
        nam_discovery_config,
        proposer_config,
    ) = _build_discovery_configs(
        execution_spec,
        model_config,
    )

    discovery = (
        run_reproducible_interaction_discovery(
            X=(
                X_development
            ),
            y=(
                y_development
            ),
            n_runs=(
                benchmark_protocol
                .n_discovery_runs
            ),
            base_seed=(
                execution_spec
                .discovery_base_seed
            ),
            selection_threshold=(
                benchmark_protocol
                .selection_threshold
            ),
            crossfit_n_splits=(
                benchmark_protocol
                .residual_crossfit_folds
            ),
            main_early_stop_fraction=0.20,
            nam_hidden_dim=(
                model_config
                .nam_hidden_dim
            ),
            nam_depth=(
                model_config
                .nam_depth
            ),
            nam_dropout=(
                model_config
                .nam_dropout
            ),
            categorical_embedding_dim=(
                model_config
                .categorical_embedding_dim
            ),
            proposer_d_model=(
                model_config
                .proposer_d_model
            ),
            proposer_n_heads=(
                model_config
                .proposer_n_heads
            ),
            proposer_n_layers=(
                model_config
                .proposer_n_layers
            ),
            proposer_dropout=(
                model_config
                .proposer_dropout
            ),
            nam_config=(
                nam_discovery_config
            ),
            proposer_config=(
                proposer_config
            ),
            device=(
                device
            ),
        )
    )

    selection_pairs = (
        normalize_pairs(
            (
                (
                    item.feature_j,
                    item.feature_k,
                )
                for item
                in (
                    discovery
                    .selection
                    .accepted_interactions
                )
            )
        )
    )

    # =========================================================
    # ISR
    # =========================================================

    if verbose:
        print(
            "\n=== ISR ==="
        )

    isr_result = (
        evaluate_isr_for_discovery(
            X=(
                X_development
            ),
            discovery=(
                discovery
            ),
            config=(
                _build_isr_config(
                    benchmark_protocol,
                    model_config,
                )
            ),
            device=(
                device
            ),
            verbose=(
                verbose
            ),
        )
    )

    isr_pairs = (
        normalize_pairs(
            (
                (
                    item.feature_j,
                    item.feature_k,
                )
                for item
                in (
                    isr_result
                    .retained_interactions
                )
            )
        )
    )

    feature_names = tuple(
        str(
            column
        )
        for column
        in X_development.columns
    )

    random_pairs = (
        sample_random_pairs(
            feature_names=(
                feature_names
            ),
            n_pairs=(
                len(
                    isr_pairs
                )
            ),
            seed=(
                execution_spec
                .random_pair_seed
            ),
        )
    )

    single_run_pairs = (
        normalize_pairs(
            discovery
            .runs[
                0
            ]
            .top_k_pairs
        )
    )

    # =========================================================
    # FINAL TRAIN / VALIDATION
    # =========================================================

    final_data = (
        prepare_final_prediction_data(
            loaded,
            final_split_seed=(
                execution_spec
                .final_split_seed
            ),
            validation_fraction=(
                benchmark_protocol
                .final_validation_fraction
            ),
        )
    )

    # =========================================================
    # NEURAL MODELS
    # =========================================================

    if verbose:
        print(
            "\n=== FINAL PREDICTIVE MODELS ==="
        )

        print(
            "1/7 Main-Effect NAM"
        )

    main = (
        _train_main_nam(
            final_data,
            seed=(
                execution_spec
                .final_model_seed
            ),
            model_config=(
                model_config
            ),
            device=(
                device
            ),
        )
    )

    if verbose:
        print(
            "2/7 Full AG-NAM"
        )

    agnam, decomposition_error = (
        _train_agnam_model(
            final_data,
            isr_pairs,
            seed=(
                execution_spec
                .final_model_seed
            ),
            model_config=(
                model_config
            ),
            device=(
                device
            ),
        )
    )

    if verbose:
        print(
            "3/7 Random-Pair NAM"
        )

    random_model, _ = (
        _train_agnam_model(
            final_data,
            random_pairs,
            seed=(
                execution_spec
                .final_model_seed
            ),
            model_config=(
                model_config
            ),
            device=(
                device
            ),
        )
    )

    if verbose:
        print(
            "4/7 AG-NAM without ISR"
        )

    no_isr, _ = (
        _train_agnam_model(
            final_data,
            selection_pairs,
            seed=(
                execution_spec
                .final_model_seed
            ),
            model_config=(
                model_config
            ),
            device=(
                device
            ),
        )
    )

    if verbose:
        print(
            "5/7 Single-Run AG-NAM"
        )

    single_run, _ = (
        _train_agnam_model(
            final_data,
            single_run_pairs,
            seed=(
                execution_spec
                .final_model_seed
            ),
            model_config=(
                model_config
            ),
            device=(
                device
            ),
        )
    )

    # =========================================================
    # EXTERNAL BASELINES
    # =========================================================

    if verbose:
        print(
            "6/7 Explainable Boosting Machine"
        )

    ebm, ebm_runtime = (
        _train_ebm(
            loaded,
            seed=(
                external_spec
                .ebm_seed
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )

    if verbose:
        print(
            "7/7 CatBoost"
        )

    catboost, catboost_runtime = (
        _train_catboost(
            loaded,
            final_data,
            seed=(
                external_spec
                .catboost_seed
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )

    # =========================================================
    # RECORD
    # =========================================================

    if len(
        selection_pairs
    ) > 0:
        isr_sparsification = float(
            1.0
            - (
                len(
                    isr_pairs
                )
                / len(
                    selection_pairs
                )
            )
        )

    else:
        isr_sparsification = (
            np.nan
        )

    candidate_k = (
        candidate_count_from_feature_count(
            audit
            .n_predictors
        )
    )

    ebm_budget = (
        external_interaction_budget(
            audit
            .n_predictors,
            protocol=(
                baseline_protocol
            ),
        )
    )

    runtime_seconds = float(
        perf_counter()
        - start_time
    )

    record = (
        OpenMLBenchmarkRecord(
            task_id=(
                task_id
            ),
            dataset_id=(
                loaded
                .dataset_id
            ),
            dataset_name=(
                loaded
                .actual_dataset_name
            ),
            feature_type=(
                audit
                .feature_type
            ),
            n_samples=(
                audit
                .n_samples
            ),
            n_features=(
                audit
                .n_predictors
            ),
            n_numeric_features=(
                audit
                .n_numeric_predictors
            ),
            n_categorical_features=(
                audit
                .n_categorical_predictors
            ),
            n_development=(
                audit
                .n_development
            ),
            n_test=(
                audit
                .n_test
            ),
            positive_label=(
                audit
                .positive_label
            ),
            negative_label=(
                audit
                .negative_label
            ),
            development_positive_fraction=(
                audit
                .development_positive_fraction
            ),
            test_positive_fraction=(
                audit
                .test_positive_fraction
            ),
            missing_fraction=(
                audit
                .missing_fraction
            ),
            candidate_k=(
                candidate_k
            ),
            ebm_interaction_budget=(
                ebm_budget
            ),
            mean_pairwise_jaccard=float(
                discovery
                .selection
                .mean_pairwise_jaccard
            ),
            n_selection_stable=(
                len(
                    selection_pairs
                )
            ),
            n_isr_retained=(
                len(
                    isr_pairs
                )
            ),
            isr_sparsification=(
                isr_sparsification
            ),
            main_auroc=(
                main.auroc
            ),
            main_auprc=(
                main.auprc
            ),
            main_balanced_accuracy=(
                main
                .balanced_accuracy
            ),
            main_f1=(
                main.f1
            ),
            agnam_auroc=(
                agnam.auroc
            ),
            agnam_auprc=(
                agnam.auprc
            ),
            agnam_balanced_accuracy=(
                agnam
                .balanced_accuracy
            ),
            agnam_f1=(
                agnam.f1
            ),
            random_pair_auroc=(
                random_model
                .auroc
            ),
            random_pair_auprc=(
                random_model
                .auprc
            ),
            no_isr_auroc=(
                no_isr
                .auroc
            ),
            no_isr_auprc=(
                no_isr
                .auprc
            ),
            single_run_auroc=(
                single_run
                .auroc
            ),
            single_run_auprc=(
                single_run
                .auprc
            ),
            ebm_auroc=(
                ebm.auroc
            ),
            ebm_auprc=(
                ebm.auprc
            ),
            ebm_balanced_accuracy=(
                ebm
                .balanced_accuracy
            ),
            ebm_f1=(
                ebm.f1
            ),
            catboost_auroc=(
                catboost
                .auroc
            ),
            catboost_auprc=(
                catboost
                .auprc
            ),
            catboost_balanced_accuracy=(
                catboost
                .balanced_accuracy
            ),
            catboost_f1=(
                catboost
                .f1
            ),
            delta_agnam_main_auroc=(
                agnam.auroc
                - main.auroc
            ),
            delta_agnam_main_auprc=(
                agnam.auprc
                - main.auprc
            ),
            delta_agnam_random_auroc=(
                agnam.auroc
                - random_model.auroc
            ),
            delta_agnam_ebm_auroc=(
                agnam.auroc
                - ebm.auroc
            ),
            delta_agnam_catboost_auroc=(
                agnam.auroc
                - catboost.auroc
            ),
            max_decomposition_error=(
                decomposition_error
            ),
            main_best_iteration=(
                main
                .best_iteration
            ),
            agnam_best_iteration=(
                agnam
                .best_iteration
            ),
            ebm_best_iteration=(
                ebm
                .best_iteration
            ),
            catboost_best_iteration=(
                catboost
                .best_iteration
            ),
            runtime_seconds=(
                runtime_seconds
            ),
            ebm_runtime_seconds=(
                ebm_runtime
            ),
            catboost_runtime_seconds=(
                catboost_runtime
            ),
        )
    )

    result = (
        SingleOpenMLBenchmarkResult(
            record=(
                record
            ),
            selection_pairs=(
                selection_pairs
            ),
            isr_pairs=(
                isr_pairs
            ),
            random_pairs=(
                random_pairs
            ),
            single_run_pairs=(
                single_run_pairs
            ),
            discovery=(
                discovery
            ),
            isr=(
                isr_result
            ),
        )
    )

    if verbose:
        print(
            "\n========================================"
        )

        print(
            "OPENML REALIZATION SUMMARY"
        )

        print(
            "========================================"
        )

        print(
            f"Selection-stable interactions: "
            f"{len(selection_pairs)}"
        )

        print(
            f"ISR-retained interactions: "
            f"{len(isr_pairs)}"
        )

        print(
            f"ISR sparsification: "
            f"{isr_sparsification:.4f}"
            if np.isfinite(
                isr_sparsification
            )
            else (
                "ISR sparsification: undefined"
            )
        )

        print(
            "\nTEST AUROC"
        )

        print(
            f"Main NAM:       "
            f"{main.auroc:.4f}"
        )

        print(
            f"Full AG-NAM:    "
            f"{agnam.auroc:.4f}"
        )

        print(
            f"Random Pair:    "
            f"{random_model.auroc:.4f}"
        )

        print(
            f"No ISR:         "
            f"{no_isr.auroc:.4f}"
        )

        print(
            f"Single Run:     "
            f"{single_run.auroc:.4f}"
        )

        print(
            f"EBM:            "
            f"{ebm.auroc:.4f}"
        )

        print(
            f"CatBoost:       "
            f"{catboost.auroc:.4f}"
        )

        print(
            "\nAG-NAM DELTAS"
        )

        print(
            f"vs Main NAM:    "
            f"{record.delta_agnam_main_auroc:+.4f}"
        )

        print(
            f"vs Random Pair: "
            f"{record.delta_agnam_random_auroc:+.4f}"
        )

        print(
            f"vs EBM:         "
            f"{record.delta_agnam_ebm_auroc:+.4f}"
        )

        print(
            f"vs CatBoost:    "
            f"{record.delta_agnam_catboost_auroc:+.4f}"
        )

        print(
            "\nADDITIVE AUDIT"
        )

        print(
            f"Maximum decomposition error: "
            f"{decomposition_error:.12e}"
        )

        print(
            f"\nRuntime (s): "
            f"{runtime_seconds:.1f}"
        )

    return result


def save_single_openml_result(
    result: SingleOpenMLBenchmarkResult,
    *,
    output_root: str | Path = (
        "results/openml/single"
    ),
) -> tuple[
    Path,
    Path,
]:
    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    task_id = (
        result
        .record
        .task_id
    )

    record_path = (
        output_root
        / (
            f"task_{task_id}.csv"
        )
    )

    pair_path = (
        output_root
        / (
            f"task_{task_id}_pairs.csv"
        )
    )

    if record_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing "
            f"single-task result: {record_path}"
        )

    if pair_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing "
            f"pair audit: {pair_path}"
        )

    pd.DataFrame(
        [
            asdict(
                result.record
            )
        ]
    ).to_csv(
        record_path,
        index=False,
    )

    build_real_world_pair_audit(
        result
    ).to_csv(
        pair_path,
        index=False,
    )

    return (
        record_path,
        pair_path,
    )