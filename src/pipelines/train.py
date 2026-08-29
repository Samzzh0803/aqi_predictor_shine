"""Day 4 and Day 5 training pipeline: honest selection, final test, SHAP, and registry."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

from src.data.open_meteo import OpenMeteoClientError
from src.feature_store import create_feature_view, load_feature_view, load_features, load_targets
from src.features.build_targets import TARGET_COLUMNS
from src.models import MODEL_NAME, RegisteredModelVersion, get_champion, register_model_version

LOGGER = logging.getLogger(__name__)

KEY_COLUMNS: Final[list[str]] = ["city_id", "event_time", "latitude", "longitude"]
ARTIFACTS_DIR: Final[Path] = Path("data") / "metrics"
MODEL_COMPARISON_CSV: Final[Path] = ARTIFACTS_DIR / "day4_model_comparison.csv"
MODEL_COMPARISON_MD: Final[Path] = ARTIFACTS_DIR / "day4_model_comparison.md"
DAY5_SUMMARY_JSON: Final[Path] = ARTIFACTS_DIR / "day5_summary.json"
ROLLING_VALIDATION_CSV: Final[Path] = ARTIFACTS_DIR / "day5_rolling_validation.csv"
ROLLING_VALIDATION_SUMMARY_CSV: Final[Path] = ARTIFACTS_DIR / "day5_rolling_validation_summary.csv"
SHAP_DIR: Final[Path] = ARTIFACTS_DIR / "shap"
BASELINE_MODELS: Final[set[str]] = {"persistence", "seasonal_naive"}
CANDIDATE_MODELS: Final[tuple[str, ...]] = (
    "ridge",
    "random_forest",
    "hist_gradient_boosting",
    "tensorflow_mlp",
)
TREE_MODELS: Final[set[str]] = {"random_forest", "hist_gradient_boosting"}
TF_RANDOM_SEED: Final[int] = 42


@dataclass(frozen=True)
class DatasetSplit:
    """Chronological train/validation/test split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class ModelRunResult:
    """Configuration, metrics, and fitted models for one training run."""

    model_name: str
    fitted_models: dict[str, Any]
    best_params: dict[str, dict[str, Any]]
    metrics: dict[str, float]


@dataclass(frozen=True)
class RollingValidationArtifacts:
    """Raw and summarized rolling-origin validation outputs."""

    raw: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class Day5Artifacts:
    """Complete Day 5 outputs for local execution."""

    comparison: pd.DataFrame
    rolling_validation_raw: pd.DataFrame
    rolling_validation_summary: pd.DataFrame
    top_two_models: list[str]
    champion_name: str
    champion_result: ModelRunResult
    feature_columns: list[str]
    data_start: str
    data_end: str
    shap_artifact_paths: dict[str, str]
    registered_version: RegisteredModelVersion | None


@dataclass(frozen=True)
class DailyTrainingDecision:
    """Decision summary for the automated daily training job."""

    candidate_champion_name: str
    candidate_metrics: dict[str, float]
    incumbent_version: int | None
    incumbent_metrics: dict[str, float] | None
    promoted: bool
    registered_version: RegisteredModelVersion | None


def load_training_frame() -> pd.DataFrame:
    """Load the materialized training frame from the local feature view or rebuild it."""

    try:
        frame = load_feature_view()
    except OpenMeteoClientError:
        frame = create_feature_view(features=load_features(), targets=load_targets())

    frame = frame.sort_values(["city_id", "event_time"]).reset_index(drop=True)
    _validate_training_frame(frame)
    return frame


def prepare_modeling_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with incomplete features or targets before model splitting."""

    feature_columns = get_model_feature_columns(frame)
    modeling_frame = frame.dropna(subset=[*feature_columns, *TARGET_COLUMNS]).reset_index(drop=True)
    if modeling_frame.empty:
        raise OpenMeteoClientError("No fully-observed rows remain after dropping incomplete features")
    return modeling_frame


def chronological_split(
    frame: pd.DataFrame,
    test_fraction: float = 0.15,
    val_fraction: float = 0.15,
) -> DatasetSplit:
    """Split the timeline chronologically into train, validation, and final test."""

    if not 0 < test_fraction < 1:
        raise OpenMeteoClientError("test_fraction must be between 0 and 1")
    if not 0 < val_fraction < 1:
        raise OpenMeteoClientError("val_fraction must be between 0 and 1")
    if frame.empty:
        raise OpenMeteoClientError("Cannot split an empty training frame")
    if not frame["event_time"].is_monotonic_increasing:
        raise OpenMeteoClientError("Training frame must be sorted chronologically before splitting")

    total_rows = len(frame)
    test_size = max(1, int(round(total_rows * test_fraction)))
    pretest_size = total_rows - test_size
    if pretest_size < 2:
        raise OpenMeteoClientError("Not enough rows left before the final test split")

    val_size = max(1, int(round(pretest_size * val_fraction)))
    train_size = pretest_size - val_size
    if train_size < 1:
        raise OpenMeteoClientError("Not enough rows left for training after validation split")

    train = frame.iloc[:train_size].reset_index(drop=True)
    val = frame.iloc[train_size:pretest_size].reset_index(drop=True)
    test = frame.iloc[pretest_size:].reset_index(drop=True)
    return DatasetSplit(train=train, val=val, test=test)


def get_model_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the exact ordered model feature columns for training."""

    excluded = set(KEY_COLUMNS + TARGET_COLUMNS)
    feature_columns = [column for column in frame.columns if column not in excluded]
    if not feature_columns:
        raise OpenMeteoClientError("Training frame does not contain any usable feature columns")
    return feature_columns


def evaluate_day4_models() -> pd.DataFrame:
    """Run Day 4 baselines and candidate models, then persist the evaluation table."""

    frame = prepare_modeling_frame(load_training_frame())
    split = chronological_split(frame)
    results = run_day4_model_suite(split=split, feature_columns=get_model_feature_columns(frame))
    comparison = _comparison_from_results(results)
    _persist_model_comparison(comparison)
    return comparison


def run_day5_pipeline(register_in_local_registry: bool = True) -> Day5Artifacts:
    """Run Day 5 with untouched final test, validation-based selection, SHAP, and registry."""

    frame = prepare_modeling_frame(load_training_frame())
    split = chronological_split(frame)
    feature_columns = get_model_feature_columns(frame)
    pretest_frame = pd.concat([split.train, split.val], ignore_index=True)

    results = run_day5_model_suite(split=split, feature_columns=feature_columns)
    rolling_validation = compute_rolling_validation(
        frame=pretest_frame,
        feature_columns=feature_columns,
        model_results=results,
    )
    comparison = build_day5_comparison(results=results, rolling_summary=rolling_validation.summary)
    _persist_model_comparison(comparison)

    champion_name = select_champion_name(comparison)
    champion_result = next(result for result in results if result.model_name == champion_name)
    shap_artifact_paths = generate_shap_artifacts(
        model_result=champion_result,
        source_frame=pretest_frame,
        feature_columns=feature_columns,
    )

    registered_version = None
    if register_in_local_registry:
        champion_metrics = champion_result.metrics.copy()
        champion_row = comparison.loc[comparison["model"] == champion_name].iloc[0]
        champion_metrics["selection_mae_mean"] = float(champion_row["selection_mae_mean"])
        champion_metrics["selection_mae_std"] = float(champion_row["selection_mae_std"])
        registered_version = register_model_version(
            model_name=MODEL_NAME,
            model_type=champion_name,
            fitted_models=champion_result.fitted_models,
            metrics=champion_metrics,
            feature_list=feature_columns,
            data_start=frame["event_time"].min().isoformat(),
            data_end=frame["event_time"].max().isoformat(),
            shap_artifact_paths=shap_artifact_paths,
        )

    _persist_day5_summary(
        comparison=comparison,
        top_two_models=get_top_two_model_names(comparison),
        champion_name=champion_name,
        rolling_validation=rolling_validation,
        shap_artifact_paths=shap_artifact_paths,
        registered_version=registered_version,
    )
    return Day5Artifacts(
        comparison=comparison,
        rolling_validation_raw=rolling_validation.raw,
        rolling_validation_summary=rolling_validation.summary,
        top_two_models=get_top_two_model_names(comparison),
        champion_name=champion_name,
        champion_result=champion_result,
        feature_columns=feature_columns,
        data_start=frame["event_time"].min().isoformat(),
        data_end=frame["event_time"].max().isoformat(),
        shap_artifact_paths=shap_artifact_paths,
        registered_version=registered_version,
    )


def run_daily_training_job() -> DailyTrainingDecision:
    """Train a fresh candidate and register it only if it beats the incumbent."""

    incumbent = _safe_get_existing_champion()
    artifacts = run_day5_pipeline(register_in_local_registry=False)
    champion_row = artifacts.comparison.loc[
        artifacts.comparison["model"] == artifacts.champion_name
    ].iloc[0]
    candidate_metrics = {
        **artifacts.champion_result.metrics.copy(),
        "selection_mae_mean": float(champion_row["selection_mae_mean"]),
        "selection_mae_std": float(champion_row["selection_mae_std"]),
    }
    promoted = _should_promote_candidate(candidate_metrics, incumbent)
    registered_version = None

    if promoted:
        registered_version = register_model_version(
            model_name=MODEL_NAME,
            model_type=artifacts.champion_name,
            fitted_models=artifacts.champion_result.fitted_models,
            metrics=candidate_metrics,
            feature_list=artifacts.feature_columns,
            data_start=artifacts.data_start,
            data_end=artifacts.data_end,
            shap_artifact_paths=artifacts.shap_artifact_paths,
        )

    LOGGER.info(
        "Daily training decision",
        extra={
            "candidate_champion_name": artifacts.champion_name,
            "candidate_selection_mae_mean": candidate_metrics["selection_mae_mean"],
            "incumbent_version": None if incumbent is None else incumbent.version,
            "incumbent_selection_mae_mean": None
            if incumbent is None
            else float(incumbent.metrics.get("selection_mae_mean", incumbent.metrics["mae_mean"])),
            "promoted": promoted,
            "registered_version": None if registered_version is None else registered_version.version,
        },
    )
    return DailyTrainingDecision(
        candidate_champion_name=artifacts.champion_name,
        candidate_metrics=candidate_metrics,
        incumbent_version=None if incumbent is None else incumbent.version,
        incumbent_metrics=None if incumbent is None else incumbent.metrics,
        promoted=promoted,
        registered_version=registered_version,
    )


def run_day4_model_suite(split: DatasetSplit, feature_columns: list[str]) -> list[ModelRunResult]:
    """Run the Day 4 baselines and classical models."""

    return [
        evaluate_baseline("persistence", split.test, "us_aqi"),
        evaluate_baseline("seasonal_naive", split.test, "aqi_mean_24h"),
        train_ridge_models(split=split, feature_columns=feature_columns),
        train_random_forest_models(split=split, feature_columns=feature_columns),
        train_hist_gradient_boosting_models(split=split, feature_columns=feature_columns),
    ]


def run_day5_model_suite(split: DatasetSplit, feature_columns: list[str]) -> list[ModelRunResult]:
    """Run Day 4 models plus the Day 5 TensorFlow MLP."""

    results = run_day4_model_suite(split=split, feature_columns=feature_columns)
    results.append(train_tensorflow_mlp(split=split, feature_columns=feature_columns))
    return results


def day4_gate_passes(comparison: pd.DataFrame) -> bool:
    """Return whether any non-baseline model beats persistence on mean MAE."""

    persistence_rows = comparison.loc[comparison["model"] == "persistence", "mae_mean"]
    if persistence_rows.empty:
        raise OpenMeteoClientError("Comparison table is missing the persistence baseline row")
    persistence_mae = float(persistence_rows.iloc[0])
    candidate_rows = comparison.loc[~comparison["model"].isin(BASELINE_MODELS), "mae_mean"]
    if candidate_rows.empty:
        raise OpenMeteoClientError("Comparison table does not include any ML model rows")
    return bool((candidate_rows < persistence_mae).any())


def evaluate_baseline(model_name: str, source_frame: pd.DataFrame, prediction_column: str) -> ModelRunResult:
    """Score a naive baseline using one existing feature column for all horizons."""

    predictions = {
        target: source_frame[prediction_column].astype(float).to_numpy() for target in TARGET_COLUMNS
    }
    metrics = _compute_metrics(source_frame=source_frame, predictions=predictions)
    return ModelRunResult(model_name=model_name, fitted_models={}, best_params={}, metrics=metrics)


def train_ridge_models(split: DatasetSplit, feature_columns: list[str]) -> ModelRunResult:
    """Train one Ridge model per horizon using validation selection."""

    alphas = [0.1, 1.0, 10.0, 100.0]
    best_params: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}
    train_sample = _sample_chronologically(split.train, max_rows=12000)
    pretest_sample = _sample_chronologically(
        pd.concat([split.train, split.val], ignore_index=True),
        max_rows=15000,
    )

    for target in TARGET_COLUMNS:
        best_alpha = None
        best_score = float("inf")
        for alpha in alphas:
            pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
            pipeline.fit(train_sample[feature_columns], train_sample[target])
            val_predictions = pipeline.predict(split.val[feature_columns])
            val_mae = mean_absolute_error(split.val[target], val_predictions)
            if val_mae < best_score:
                best_score = val_mae
                best_alpha = alpha

        assert best_alpha is not None
        final_pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))])
        final_pipeline.fit(pretest_sample[feature_columns], pretest_sample[target])
        fitted_models[target] = final_pipeline
        best_params[target] = {"alpha": best_alpha}

    predictions = {target: fitted_models[target].predict(split.test[feature_columns]) for target in TARGET_COLUMNS}
    return ModelRunResult("ridge", fitted_models, best_params, _compute_metrics(split.test, predictions))


def train_random_forest_models(split: DatasetSplit, feature_columns: list[str]) -> ModelRunResult:
    """Train one RandomForest model per horizon with small time-series search."""

    param_space = {
        "n_estimators": [200],
        "max_depth": [15, 25],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.7],
    }
    best_params: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}
    train_sample = _sample_chronologically(split.train, max_rows=8000)
    pretest_sample = _sample_chronologically(
        pd.concat([split.train, split.val], ignore_index=True),
        max_rows=10000,
    )

    for target in TARGET_COLUMNS:
        search = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=1),
            param_distributions=param_space,
            n_iter=2,
            scoring="neg_mean_absolute_error",
            cv=TimeSeriesSplit(n_splits=2),
            random_state=42,
            n_jobs=1,
            refit=True,
        )
        search.fit(train_sample[feature_columns], train_sample[target])
        selected_params = search.best_params_
        final_model = RandomForestRegressor(random_state=42, n_jobs=1, **selected_params)
        final_model.fit(pretest_sample[feature_columns], pretest_sample[target])
        fitted_models[target] = final_model
        best_params[target] = selected_params

    predictions = {target: fitted_models[target].predict(split.test[feature_columns]) for target in TARGET_COLUMNS}
    return ModelRunResult("random_forest", fitted_models, best_params, _compute_metrics(split.test, predictions))


def train_hist_gradient_boosting_models(split: DatasetSplit, feature_columns: list[str]) -> ModelRunResult:
    """Train one HistGradientBoosting model per horizon using validation selection."""

    candidates = [
        {"max_depth": None, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 20},
        {"max_depth": 8, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 20},
        {"max_depth": 12, "learning_rate": 0.03, "max_iter": 250, "min_samples_leaf": 30},
    ]
    best_params: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}
    train_sample = _sample_chronologically(split.train, max_rows=12000)
    pretest_sample = _sample_chronologically(
        pd.concat([split.train, split.val], ignore_index=True),
        max_rows=15000,
    )

    for target in TARGET_COLUMNS:
        best_candidate = None
        best_score = float("inf")
        for params in candidates:
            estimator = HistGradientBoostingRegressor(random_state=42, **params)
            estimator.fit(train_sample[feature_columns], train_sample[target])
            val_predictions = estimator.predict(split.val[feature_columns])
            val_mae = mean_absolute_error(split.val[target], val_predictions)
            if val_mae < best_score:
                best_score = val_mae
                best_candidate = params

        assert best_candidate is not None
        final_model = HistGradientBoostingRegressor(random_state=42, **best_candidate)
        final_model.fit(pretest_sample[feature_columns], pretest_sample[target])
        fitted_models[target] = final_model
        best_params[target] = best_candidate

    predictions = {target: fitted_models[target].predict(split.test[feature_columns]) for target in TARGET_COLUMNS}
    return ModelRunResult(
        "hist_gradient_boosting",
        fitted_models,
        best_params,
        _compute_metrics(split.test, predictions),
    )


def train_tensorflow_mlp(split: DatasetSplit, feature_columns: list[str]) -> ModelRunResult:
    """Train the Day 5 TensorFlow MLP with a Dense(3) output head."""

    tf.keras.utils.set_random_seed(TF_RANDOM_SEED)

    train_sample = _sample_chronologically(split.train, max_rows=12000)
    val_sample = _sample_chronologically(split.val, max_rows=3000)
    train_val_frame = pd.concat([split.train, split.val], ignore_index=True)
    pretest_sample = _sample_chronologically(train_val_frame, max_rows=15000)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_sample[feature_columns]).astype("float32")
    x_val = scaler.transform(val_sample[feature_columns]).astype("float32")
    y_train = train_sample[TARGET_COLUMNS].to_numpy(dtype="float32")
    y_val = val_sample[TARGET_COLUMNS].to_numpy(dtype="float32")

    warmup_model = _build_mlp_model(input_dim=len(feature_columns))
    training_callbacks = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    history = warmup_model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=128,
        verbose=0,
        callbacks=training_callbacks,
    )

    final_scaler = StandardScaler()
    x_pretest = final_scaler.fit_transform(pretest_sample[feature_columns]).astype("float32")
    x_test = final_scaler.transform(split.test[feature_columns]).astype("float32")
    y_pretest = pretest_sample[TARGET_COLUMNS].to_numpy(dtype="float32")

    final_model = _build_mlp_model(input_dim=len(feature_columns))
    final_model.fit(
        x_pretest,
        y_pretest,
        validation_split=0.1,
        epochs=max(1, len(history.history.get("loss", []))),
        batch_size=128,
        verbose=0,
        callbacks=training_callbacks,
    )

    predictions_matrix = final_model.predict(x_test, verbose=0)
    predictions = {
        "target_aqi_day1": predictions_matrix[:, 0],
        "target_aqi_day2": predictions_matrix[:, 1],
        "target_aqi_day3": predictions_matrix[:, 2],
    }
    return ModelRunResult(
        model_name="tensorflow_mlp",
        fitted_models={"all_horizons": {"model": final_model, "scaler": final_scaler}},
        best_params={"all_horizons": {"epochs": len(history.history.get("loss", [])), "batch_size": 128}},
        metrics=_compute_metrics(split.test, predictions),
    )


def compute_rolling_validation(
    frame: pd.DataFrame,
    feature_columns: list[str],
    model_results: list[ModelRunResult],
    n_splits: int = 4,
) -> RollingValidationArtifacts:
    """Run rolling-origin validation on the pre-test timeline for all candidate models."""

    model_lookup = {result.model_name: result for result in model_results}
    rows: list[dict[str, Any]] = []
    cv = TimeSeriesSplit(n_splits=n_splits)

    for model_name in CANDIDATE_MODELS:
        template_result = model_lookup[model_name]
        for fold_number, (train_idx, test_idx) in enumerate(cv.split(frame), start=1):
            fold_train = frame.iloc[train_idx].reset_index(drop=True)
            fold_test = frame.iloc[test_idx].reset_index(drop=True)
            fold_val_size = max(1, int(round(len(fold_train) * 0.15)))
            fold_train_part = fold_train.iloc[:-fold_val_size].reset_index(drop=True)
            fold_val_part = fold_train.iloc[-fold_val_size:].reset_index(drop=True)
            split = DatasetSplit(train=fold_train_part, val=fold_val_part, test=fold_test)
            refit_result = refit_model_result(
                model_name=model_name,
                split=split,
                feature_columns=feature_columns,
                template_result=template_result,
            )
            rows.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "mae_day1": refit_result.metrics["mae_day1"],
                    "mae_day2": refit_result.metrics["mae_day2"],
                    "mae_day3": refit_result.metrics["mae_day3"],
                    "mae_mean": refit_result.metrics["mae_mean"],
                }
            )

    raw = pd.DataFrame(rows)
    summary = _build_rolling_validation_summary(raw)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(ROLLING_VALIDATION_CSV, index=False)
    summary.to_csv(ROLLING_VALIDATION_SUMMARY_CSV, index=False)
    return RollingValidationArtifacts(raw=raw, summary=summary)


def refit_model_result(
    model_name: str,
    split: DatasetSplit,
    feature_columns: list[str],
    template_result: ModelRunResult | None = None,
) -> ModelRunResult:
    """Refit a model family for rolling validation using existing best params when available."""

    if model_name == "ridge":
        return train_ridge_models(split, feature_columns) if template_result is None else _train_fixed_ridge(
            split,
            feature_columns,
            template_result.best_params,
        )
    if model_name == "random_forest":
        return train_random_forest_models(split, feature_columns) if template_result is None else _train_fixed_random_forest(
            split,
            feature_columns,
            template_result.best_params,
        )
    if model_name == "hist_gradient_boosting":
        return train_hist_gradient_boosting_models(split, feature_columns) if template_result is None else _train_fixed_hist_gradient_boosting(
            split,
            feature_columns,
            template_result.best_params,
        )
    if model_name == "tensorflow_mlp":
        return train_tensorflow_mlp(split, feature_columns)
    raise OpenMeteoClientError(f"Unsupported model for refit: {model_name}")


def build_day5_comparison(results: list[ModelRunResult], rolling_summary: pd.DataFrame) -> pd.DataFrame:
    """Combine untouched final-test metrics with validation-based selection metrics."""

    comparison = _comparison_from_results(results)
    comparison = comparison.merge(rolling_summary, on="model", how="left")
    return comparison.sort_values(
        ["selection_mae_mean", "selection_mae_std", "mae_mean", "rmse_mean"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def select_champion_name(comparison: pd.DataFrame) -> str:
    """Select the overall champion from all candidate models using validation MAE."""

    candidate_rows = comparison.loc[comparison["model"].isin(CANDIDATE_MODELS)].sort_values(
        ["selection_mae_mean", "selection_mae_std", "mae_mean", "rmse_mean"],
        kind="stable",
    )
    if candidate_rows.empty:
        raise OpenMeteoClientError("Comparison table does not include any candidate models")
    if candidate_rows["selection_mae_mean"].isna().any():
        raise OpenMeteoClientError("Champion selection requires validation MAE for every candidate model")
    return str(candidate_rows.iloc[0]["model"])


def get_top_two_model_names(comparison: pd.DataFrame) -> list[str]:
    """Return the top two candidate models by validation MAE."""

    top_two = (
        comparison.loc[comparison["model"].isin(CANDIDATE_MODELS)]
        .sort_values(["selection_mae_mean", "selection_mae_std", "mae_mean", "rmse_mean"], kind="stable")
        .head(2)["model"]
        .tolist()
    )
    if len(top_two) < 2:
        raise OpenMeteoClientError("Need at least two candidate models for Day 5 validation summary")
    return top_two


def generate_shap_artifacts(
    model_result: ModelRunResult,
    source_frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path = SHAP_DIR,
) -> dict[str, str]:
    """Generate SHAP artifacts for the champion model, using a model-specific explainer."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    background_frame = _sample_chronologically(source_frame, max_rows=200)
    evaluation_frame = _sample_chronologically(source_frame, max_rows=100)
    artifact_paths: dict[str, str] = {}

    if model_result.model_name in TREE_MODELS:
        for horizon_index, horizon in enumerate(TARGET_COLUMNS):
            model = model_result.fitted_models[horizon]
            background_features = background_frame[feature_columns]
            evaluation_features = evaluation_frame[feature_columns]
            explainer = shap.TreeExplainer(model)
            shap_values = _extract_horizon_shap_values(
                explainer.shap_values(evaluation_features),
                horizon_index=horizon_index,
            )
            _write_shap_outputs(
                shap_values=shap_values,
                feature_frame=evaluation_features,
                model_name=model_result.model_name,
                horizon=horizon,
                output_dir=output_dir,
                artifact_paths=artifact_paths,
                plt=plt,
            )
        return artifact_paths

    if model_result.model_name == "ridge":
        for horizon_index, horizon in enumerate(TARGET_COLUMNS):
            pipeline = model_result.fitted_models[horizon]
            scaler = pipeline.named_steps["scaler"]
            estimator = pipeline.named_steps["ridge"]
            background_features = pd.DataFrame(
                scaler.transform(background_frame[feature_columns]),
                columns=feature_columns,
            )
            evaluation_features = pd.DataFrame(
                scaler.transform(evaluation_frame[feature_columns]),
                columns=feature_columns,
            )
            explainer = shap.LinearExplainer(estimator, background_features)
            shap_values = _extract_horizon_shap_values(
                explainer.shap_values(evaluation_features),
                horizon_index=horizon_index,
            )
            _write_shap_outputs(
                shap_values=shap_values,
                feature_frame=evaluation_features,
                model_name=model_result.model_name,
                horizon=horizon,
                output_dir=output_dir,
                artifact_paths=artifact_paths,
                plt=plt,
            )
        return artifact_paths

    if model_result.model_name == "tensorflow_mlp":
        model_bundle = model_result.fitted_models["all_horizons"]
        scaler = model_bundle["scaler"]
        model = model_bundle["model"]
        background_features = scaler.transform(background_frame[feature_columns]).astype("float32")
        evaluation_features = scaler.transform(evaluation_frame[feature_columns]).astype("float32")
        explainer = shap.GradientExplainer(model, background_features)
        shap_values = explainer.shap_values(evaluation_features)
        for horizon_index, horizon in enumerate(TARGET_COLUMNS):
            horizon_values = _extract_horizon_shap_values(shap_values, horizon_index=horizon_index)
            feature_frame = pd.DataFrame(evaluation_features, columns=feature_columns)
            _write_shap_outputs(
                shap_values=horizon_values,
                feature_frame=feature_frame,
                model_name=model_result.model_name,
                horizon=horizon,
                output_dir=output_dir,
                artifact_paths=artifact_paths,
                plt=plt,
            )
        return artifact_paths

    raise OpenMeteoClientError(f"Unsupported SHAP model type: {model_result.model_name}")


def _build_mlp_model(input_dim: int) -> keras.Model:
    """Build and compile the Day 5 MLP architecture."""

    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(3),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(), loss="mae")
    return model


def _train_fixed_ridge(
    split: DatasetSplit,
    feature_columns: list[str],
    best_params: dict[str, dict[str, Any]],
) -> ModelRunResult:
    fitted_models: dict[str, Any] = {}
    pretest = pd.concat([split.train, split.val], ignore_index=True)
    for target in TARGET_COLUMNS:
        alpha = float(best_params[target]["alpha"])
        pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        pipeline.fit(pretest[feature_columns], pretest[target])
        fitted_models[target] = pipeline
    predictions = {target: fitted_models[target].predict(split.test[feature_columns]) for target in TARGET_COLUMNS}
    return ModelRunResult("ridge", fitted_models, best_params, _compute_metrics(split.test, predictions))


def _train_fixed_random_forest(
    split: DatasetSplit,
    feature_columns: list[str],
    best_params: dict[str, dict[str, Any]],
) -> ModelRunResult:
    fitted_models: dict[str, Any] = {}
    pretest = pd.concat([split.train, split.val], ignore_index=True)
    for target in TARGET_COLUMNS:
        estimator = RandomForestRegressor(random_state=42, n_jobs=1, **best_params[target])
        estimator.fit(pretest[feature_columns], pretest[target])
        fitted_models[target] = estimator
    predictions = {target: fitted_models[target].predict(split.test[feature_columns]) for target in TARGET_COLUMNS}
    return ModelRunResult("random_forest", fitted_models, best_params, _compute_metrics(split.test, predictions))


def _train_fixed_hist_gradient_boosting(
    split: DatasetSplit,
    feature_columns: list[str],
    best_params: dict[str, dict[str, Any]],
) -> ModelRunResult:
    fitted_models: dict[str, Any] = {}
    pretest = pd.concat([split.train, split.val], ignore_index=True)
    for target in TARGET_COLUMNS:
        estimator = HistGradientBoostingRegressor(random_state=42, **best_params[target])
        estimator.fit(pretest[feature_columns], pretest[target])
        fitted_models[target] = estimator
    predictions = {target: fitted_models[target].predict(split.test[feature_columns]) for target in TARGET_COLUMNS}
    return ModelRunResult(
        "hist_gradient_boosting",
        fitted_models,
        best_params,
        _compute_metrics(split.test, predictions),
    )


def _compute_metrics(source_frame: pd.DataFrame, predictions: dict[str, Any]) -> dict[str, float]:
    mae_scores: dict[str, float] = {}
    rmse_scores: dict[str, float] = {}
    r2_scores: dict[str, float] = {}

    for index, target in enumerate(TARGET_COLUMNS, start=1):
        y_true = source_frame[target].astype(float)
        y_pred = pd.Series(predictions[target], index=source_frame.index, dtype="float64")
        mae_scores[f"mae_day{index}"] = float(mean_absolute_error(y_true, y_pred))
        rmse_scores[f"rmse_day{index}"] = float(mean_squared_error(y_true, y_pred) ** 0.5)
        r2_scores[f"r2_day{index}"] = float(r2_score(y_true, y_pred))

    return {
        "mae_day1": mae_scores["mae_day1"],
        "mae_day2": mae_scores["mae_day2"],
        "mae_day3": mae_scores["mae_day3"],
        "mae_mean": sum(mae_scores.values()) / len(mae_scores),
        "rmse_mean": sum(rmse_scores.values()) / len(rmse_scores),
        "r2_mean": sum(r2_scores.values()) / len(r2_scores),
    }


def _comparison_from_results(results: list[ModelRunResult]) -> pd.DataFrame:
    rows = [
        {
            "model": result.model_name,
            "mae_day1": result.metrics["mae_day1"],
            "mae_day2": result.metrics["mae_day2"],
            "mae_day3": result.metrics["mae_day3"],
            "mae_mean": result.metrics["mae_mean"],
            "rmse_mean": result.metrics["rmse_mean"],
            "r2_mean": result.metrics["r2_mean"],
        }
        for result in results
    ]
    comparison = pd.DataFrame(rows)
    persistence_row = comparison.loc[comparison["model"] == "persistence"]
    ordered = comparison.loc[comparison["model"] != "persistence"].sort_values(
        ["mae_mean", "rmse_mean"],
        kind="stable",
    )
    return pd.concat([persistence_row, ordered], ignore_index=True)


def _build_rolling_validation_summary(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby("model", sort=False).agg(
        selection_mae_mean=("mae_mean", "mean"),
        selection_mae_std=("mae_mean", "std"),
        selection_mae_day1_mean=("mae_day1", "mean"),
        selection_mae_day1_std=("mae_day1", "std"),
        selection_mae_day2_mean=("mae_day2", "mean"),
        selection_mae_day2_std=("mae_day2", "std"),
        selection_mae_day3_mean=("mae_day3", "mean"),
        selection_mae_day3_std=("mae_day3", "std"),
    )
    return grouped.reset_index()


def _persist_model_comparison(comparison: pd.DataFrame) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(MODEL_COMPARISON_CSV, index=False)
    MODEL_COMPARISON_MD.write_text(_to_markdown_table(comparison), encoding="utf-8")


def _persist_day5_summary(
    *,
    comparison: pd.DataFrame,
    top_two_models: list[str],
    champion_name: str,
    rolling_validation: RollingValidationArtifacts,
    shap_artifact_paths: dict[str, str],
    registered_version: RegisteredModelVersion | None,
) -> None:
    DAY5_SUMMARY_JSON.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "top_two_models": top_two_models,
                "champion_name": champion_name,
                "comparison_rows": comparison.to_dict(orient="records"),
                "rolling_validation_rows": rolling_validation.raw.to_dict(orient="records"),
                "rolling_validation_summary_rows": rolling_validation.summary.to_dict(orient="records"),
                "shap_artifact_paths": shap_artifact_paths,
                "registered_version": None if registered_version is None else registered_version.version,
                "champion_version": None if registered_version is None else get_champion().version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _to_markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = []
    for index, header in enumerate(headers):
        cell_widths = [len(row[index]) for row in rows] if rows else [0]
        widths.append(max(len(header), max(cell_widths)))

    def _format_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [_format_row(headers), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines) + "\n"


def _sample_chronologically(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    sampled = frame.iloc[pd.Index(np.linspace(0, len(frame) - 1, num=max_rows, dtype=int)).unique()]
    return sampled.reset_index(drop=True)


def _extract_horizon_shap_values(raw_values: Any, horizon_index: int) -> np.ndarray:
    if isinstance(raw_values, list):
        values = raw_values[horizon_index]
    else:
        values = np.asarray(raw_values)
        if values.ndim == 3:
            values = values[:, :, horizon_index]
    return np.asarray(values, dtype="float64")


def _write_shap_outputs(
    *,
    shap_values: np.ndarray,
    feature_frame: pd.DataFrame,
    model_name: str,
    horizon: str,
    output_dir: Path,
    artifact_paths: dict[str, str],
    plt: Any,
) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_frame.columns.tolist(),
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance_path = output_dir / f"{model_name}_{horizon}_importance.csv"
    importance.to_csv(importance_path, index=False)
    artifact_paths[f"{horizon}_importance_csv"] = str(importance_path)

    summary_path = output_dir / f"{model_name}_{horizon}_summary.png"
    plt.figure()
    shap.summary_plot(shap_values, feature_frame, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()
    artifact_paths[f"{horizon}_summary_png"] = str(summary_path)

    top10_path = output_dir / f"{model_name}_{horizon}_top10.png"
    plt.figure()
    shap.summary_plot(shap_values, feature_frame, plot_type="bar", max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(top10_path, bbox_inches="tight")
    plt.close()
    artifact_paths[f"{horizon}_top10_png"] = str(top10_path)


def _validate_training_frame(frame: pd.DataFrame) -> None:
    required_columns = set(KEY_COLUMNS + TARGET_COLUMNS)
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise OpenMeteoClientError(
            f"Training frame is missing required columns for Day 4/5: {missing_columns}"
        )
    if frame.duplicated(subset=["city_id", "event_time"]).any():
        raise OpenMeteoClientError("Training frame contains duplicate (city_id, event_time) keys")


def _safe_get_existing_champion() -> RegisteredModelVersion | None:
    try:
        return get_champion()
    except OpenMeteoClientError:
        return None


def _should_promote_candidate(
    candidate_metrics: dict[str, float],
    incumbent: RegisteredModelVersion | None,
) -> bool:
    if incumbent is None:
        return True
    incumbent_selection_mae = float(
        incumbent.metrics.get("selection_mae_mean", incumbent.metrics["mae_mean"])
    )
    incumbent_selection_std = float(incumbent.metrics.get("selection_mae_std", 0.0))
    return (
        float(candidate_metrics["selection_mae_mean"]),
        float(candidate_metrics["selection_mae_std"]),
        float(candidate_metrics["mae_mean"]),
    ) < (
        incumbent_selection_mae,
        incumbent_selection_std,
        float(incumbent.metrics["mae_mean"]),
    )


if __name__ == "__main__":
    artifacts = run_day5_pipeline(register_in_local_registry=True)
    LOGGER.info(
        "Day 5 pipeline completed",
        extra={
            "champion_name": artifacts.champion_name,
            "top_two_models": artifacts.top_two_models,
        },
    )
