"""Day 5 tests for champion selection, rolling validation, and SHAP generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.registry import get_champion as get_registry_champion
from src.models.registry import register_model_version as real_register_model_version
from src.pipelines.train import (
    ModelRunResult,
    RollingValidationArtifacts,
    _build_mlp_model,
    build_day5_comparison,
    compute_rolling_validation,
    generate_shap_artifacts,
    get_top_two_model_names,
    run_day5_pipeline,
    select_champion_name,
)


def test_get_top_two_model_names_uses_validation_metric() -> None:
    comparison = pd.DataFrame(
        {
            "model": ["persistence", "seasonal_naive", "ridge", "hist_gradient_boosting", "random_forest"],
            "selection_mae_mean": [np.nan, np.nan, 8.0, 8.5, 9.0],
            "selection_mae_std": [np.nan, np.nan, 0.5, 0.4, 0.6],
            "mae_mean": [10.0, 9.0, 8.4, 8.6, 9.1],
            "rmse_mean": [11.0, 10.0, 9.0, 9.2, 9.8],
        }
    )

    assert get_top_two_model_names(comparison) == ["ridge", "hist_gradient_boosting"]


def test_build_day5_comparison_merges_validation_summary() -> None:
    results = [
        _mock_model_result("persistence", mae_mean=10.0),
        _mock_model_result("ridge", mae_mean=8.0),
    ]
    rolling_summary = pd.DataFrame(
        {
            "model": ["ridge"],
            "selection_mae_mean": [7.5],
            "selection_mae_std": [0.3],
            "selection_mae_day1_mean": [7.0],
            "selection_mae_day1_std": [0.2],
            "selection_mae_day2_mean": [7.5],
            "selection_mae_day2_std": [0.3],
            "selection_mae_day3_mean": [8.0],
            "selection_mae_day3_std": [0.4],
        }
    )

    comparison = build_day5_comparison(results, rolling_summary)

    assert "selection_mae_mean" in comparison.columns
    assert float(comparison.loc[comparison["model"] == "ridge", "selection_mae_mean"].iloc[0]) == 7.5


def test_compute_rolling_validation_writes_mean_and_std(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "city_id": ["lahore"] * 20,
            "event_time": pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC"),
            "latitude": [31.5] * 20,
            "longitude": [74.3] * 20,
            "us_aqi": np.arange(20),
            "aqi_mean_24h": np.arange(20),
            "feature_x": np.arange(20),
            "target_aqi_day1": np.arange(20),
            "target_aqi_day2": np.arange(20),
            "target_aqi_day3": np.arange(20),
        }
    )
    model_results = [
        _mock_model_result("ridge", 8.0),
        _mock_tree_result("random_forest", 8.5),
        _mock_hist_result(8.3),
        _mock_mlp_result(8.7),
    ]
    monkeypatch.setattr("src.pipelines.train.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("src.pipelines.train.ROLLING_VALIDATION_CSV", tmp_path / "rolling.csv")
    monkeypatch.setattr(
        "src.pipelines.train.ROLLING_VALIDATION_SUMMARY_CSV",
        tmp_path / "rolling_summary.csv",
    )

    artifacts = compute_rolling_validation(
        frame=frame,
        feature_columns=["us_aqi", "aqi_mean_24h", "feature_x"],
        model_results=model_results,
        n_splits=2,
    )

    assert (tmp_path / "rolling.csv").exists()
    assert (tmp_path / "rolling_summary.csv").exists()
    assert set(artifacts.summary["model"]) == {
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
        "tensorflow_mlp",
    }
    assert "selection_mae_mean" in artifacts.summary.columns
    assert "selection_mae_std" in artifacts.summary.columns


def test_run_day5_pipeline_registers_ridge_when_validation_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _run_day5_pipeline_for_champion(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        comparison_models=[
            _mock_model_result("persistence", 11.0),
            _mock_model_result("seasonal_naive", 10.0),
            _mock_model_result("ridge", 8.0),
            _mock_tree_result("hist_gradient_boosting", 8.5),
            _mock_tree_result("random_forest", 9.0),
            _mock_mlp_result(9.5),
        ],
        selection_rows=[
            {"model": "ridge", "selection_mae_mean": 7.0, "selection_mae_std": 0.2},
            {"model": "hist_gradient_boosting", "selection_mae_mean": 7.5, "selection_mae_std": 0.3},
            {"model": "random_forest", "selection_mae_mean": 8.0, "selection_mae_std": 0.4},
            {"model": "tensorflow_mlp", "selection_mae_mean": 8.2, "selection_mae_std": 0.5},
        ],
    )

    assert artifacts.registered_version is not None
    assert artifacts.registered_version.model_type == "ridge"
    assert artifacts.registered_version.metrics["selection_mae_mean"] == 7.0
    assert artifacts.registered_version.metrics["selection_mae_mean"] == artifacts.comparison["selection_mae_mean"].min()


def test_run_day5_pipeline_registers_hgb_when_validation_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _run_day5_pipeline_for_champion(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        comparison_models=[
            _mock_model_result("persistence", 11.0),
            _mock_model_result("seasonal_naive", 10.0),
            _mock_model_result("ridge", 8.4),
            _mock_tree_result("hist_gradient_boosting", 8.5),
            _mock_tree_result("random_forest", 8.8),
            _mock_mlp_result(9.1),
        ],
        selection_rows=[
            {"model": "ridge", "selection_mae_mean": 7.3, "selection_mae_std": 0.3},
            {"model": "hist_gradient_boosting", "selection_mae_mean": 7.1, "selection_mae_std": 0.2},
            {"model": "random_forest", "selection_mae_mean": 7.7, "selection_mae_std": 0.4},
            {"model": "tensorflow_mlp", "selection_mae_mean": 8.2, "selection_mae_std": 0.5},
        ],
    )

    assert artifacts.registered_version is not None
    assert artifacts.registered_version.model_type == "hist_gradient_boosting"


def test_run_day5_pipeline_registers_mlp_when_validation_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _run_day5_pipeline_for_champion(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        comparison_models=[
            _mock_model_result("persistence", 11.0),
            _mock_model_result("seasonal_naive", 10.0),
            _mock_model_result("ridge", 8.2),
            _mock_tree_result("hist_gradient_boosting", 8.3),
            _mock_tree_result("random_forest", 8.5),
            _mock_mlp_result(8.8),
        ],
        selection_rows=[
            {"model": "ridge", "selection_mae_mean": 7.4, "selection_mae_std": 0.3},
            {"model": "hist_gradient_boosting", "selection_mae_mean": 7.2, "selection_mae_std": 0.2},
            {"model": "random_forest", "selection_mae_mean": 7.5, "selection_mae_std": 0.4},
            {"model": "tensorflow_mlp", "selection_mae_mean": 7.1, "selection_mae_std": 0.2},
        ],
    )

    assert artifacts.registered_version is not None
    assert artifacts.registered_version.model_type == "tensorflow_mlp"


def test_select_champion_name_uses_lowest_validation_mae() -> None:
    comparison = pd.DataFrame(
        {
            "model": ["ridge", "hist_gradient_boosting", "random_forest", "tensorflow_mlp"],
            "selection_mae_mean": [7.5, 7.2, 7.8, 7.4],
            "selection_mae_std": [0.3, 0.4, 0.2, 0.1],
            "mae_mean": [8.0, 8.1, 8.2, 8.3],
            "rmse_mean": [9.0, 9.1, 9.2, 9.3],
        }
    )

    assert select_champion_name(comparison) == "hist_gradient_boosting"
    assert comparison["selection_mae_mean"].min() == 7.2


def test_generate_shap_artifacts_smoke_for_ridge(tmp_path: Path) -> None:
    feature_columns = ["us_aqi", "aqi_mean_24h", "feature_x"]
    feature_frame = pd.DataFrame(
        {
            "us_aqi": np.linspace(10, 20, 12),
            "aqi_mean_24h": np.linspace(9, 19, 12),
            "feature_x": np.linspace(1, 2, 12),
        }
    )
    source_frame = pd.DataFrame(
        {
            "city_id": ["lahore"] * 12,
            "event_time": pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC"),
            "latitude": [31.5] * 12,
            "longitude": [74.3] * 12,
            "us_aqi": feature_frame["us_aqi"],
            "aqi_mean_24h": feature_frame["aqi_mean_24h"],
            "feature_x": feature_frame["feature_x"],
            "target_aqi_day1": np.linspace(11, 21, 12),
            "target_aqi_day2": np.linspace(12, 22, 12),
            "target_aqi_day3": np.linspace(13, 23, 12),
        }
    )

    fitted_models = {}
    for target in ["target_aqi_day1", "target_aqi_day2", "target_aqi_day3"]:
        pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        pipeline.fit(feature_frame, source_frame[target])
        fitted_models[target] = pipeline

    result = ModelRunResult(
        model_name="ridge",
        fitted_models=fitted_models,
        best_params={
            "target_aqi_day1": {"alpha": 1.0},
            "target_aqi_day2": {"alpha": 1.0},
            "target_aqi_day3": {"alpha": 1.0},
        },
        metrics={"mae_day1": 1.0, "mae_day2": 1.0, "mae_day3": 1.0, "mae_mean": 1.0, "rmse_mean": 1.0, "r2_mean": 0.5},
    )

    artifact_paths = generate_shap_artifacts(
        model_result=result,
        source_frame=source_frame,
        feature_columns=feature_columns,
        output_dir=tmp_path,
    )

    assert Path(artifact_paths["target_aqi_day1_summary_png"]).exists()
    assert Path(artifact_paths["target_aqi_day1_importance_csv"]).exists()


def _run_day5_pipeline_for_champion(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    comparison_models: list[ModelRunResult],
    selection_rows: list[dict[str, float]],
):
    frame = pd.DataFrame(
        {
            "city_id": ["lahore"] * 12,
            "event_time": pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC"),
            "latitude": [31.5] * 12,
            "longitude": [74.3] * 12,
            "us_aqi": np.linspace(10, 21, 12),
            "aqi_mean_24h": np.linspace(9, 20, 12),
            "feature_x": np.linspace(1, 12, 12),
            "target_aqi_day1": np.linspace(11, 22, 12),
            "target_aqi_day2": np.linspace(12, 23, 12),
            "target_aqi_day3": np.linspace(13, 24, 12),
        }
    )
    selection_summary = pd.DataFrame(selection_rows)
    selection_summary["selection_mae_day1_mean"] = selection_summary["selection_mae_mean"]
    selection_summary["selection_mae_day1_std"] = selection_summary["selection_mae_std"]
    selection_summary["selection_mae_day2_mean"] = selection_summary["selection_mae_mean"]
    selection_summary["selection_mae_day2_std"] = selection_summary["selection_mae_std"]
    selection_summary["selection_mae_day3_mean"] = selection_summary["selection_mae_mean"]
    selection_summary["selection_mae_day3_std"] = selection_summary["selection_mae_std"]

    monkeypatch.setattr("src.pipelines.train.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("src.pipelines.train.MODEL_COMPARISON_CSV", tmp_path / "comparison.csv")
    monkeypatch.setattr("src.pipelines.train.MODEL_COMPARISON_MD", tmp_path / "comparison.md")
    monkeypatch.setattr("src.pipelines.train.DAY5_SUMMARY_JSON", tmp_path / "summary.json")
    monkeypatch.setattr("src.pipelines.train.ROLLING_VALIDATION_CSV", tmp_path / "rolling.csv")
    monkeypatch.setattr("src.pipelines.train.ROLLING_VALIDATION_SUMMARY_CSV", tmp_path / "rolling_summary.csv")
    monkeypatch.setattr("src.pipelines.train.SHAP_DIR", tmp_path / "shap")
    monkeypatch.setattr("src.pipelines.train.load_training_frame", lambda: frame)
    monkeypatch.setattr("src.pipelines.train.run_day5_model_suite", lambda split, feature_columns: comparison_models)
    monkeypatch.setattr(
        "src.pipelines.train.compute_rolling_validation",
        lambda frame, feature_columns, model_results, n_splits=4: RollingValidationArtifacts(
            raw=selection_summary.assign(fold=1),
            summary=selection_summary,
        ),
    )
    monkeypatch.setattr(
        "src.pipelines.train.generate_shap_artifacts",
        lambda model_result, source_frame, feature_columns, output_dir=Path("unused"): {"summary": str(tmp_path / "champion.png")},
    )
    monkeypatch.setattr(
        "src.pipelines.train.register_model_version",
        lambda **kwargs: real_register_model_version(**kwargs, registry_root=tmp_path / "registry"),
    )
    monkeypatch.setattr(
        "src.pipelines.train.get_champion",
        lambda model_name="pearls_aqi_forecaster": get_registry_champion(model_name, registry_root=tmp_path / "registry"),
    )

    return run_day5_pipeline(register_in_local_registry=True)


def _mock_model_result(model_name: str, mae_mean: float) -> ModelRunResult:
    return ModelRunResult(
        model_name=model_name,
        fitted_models={},
        best_params={
            "target_aqi_day1": {"alpha": 1.0},
            "target_aqi_day2": {"alpha": 1.0},
            "target_aqi_day3": {"alpha": 1.0},
        },
        metrics={
            "mae_day1": mae_mean,
            "mae_day2": mae_mean,
            "mae_day3": mae_mean,
            "mae_mean": mae_mean,
            "rmse_mean": mae_mean + 1.0,
            "r2_mean": 0.5,
        },
    )


def _mock_tree_result(model_name: str, mae_mean: float) -> ModelRunResult:
    return ModelRunResult(
        model_name=model_name,
        fitted_models={
            "target_aqi_day1": Ridge(),
            "target_aqi_day2": Ridge(),
            "target_aqi_day3": Ridge(),
        },
        best_params={
            "target_aqi_day1": {"max_depth": 8},
            "target_aqi_day2": {"max_depth": 8},
            "target_aqi_day3": {"max_depth": 8},
        },
        metrics={
            "mae_day1": mae_mean,
            "mae_day2": mae_mean,
            "mae_day3": mae_mean,
            "mae_mean": mae_mean,
            "rmse_mean": mae_mean + 1.0,
            "r2_mean": 0.5,
        },
    )


def _mock_hist_result(mae_mean: float) -> ModelRunResult:
    return ModelRunResult(
        model_name="hist_gradient_boosting",
        fitted_models={
            "target_aqi_day1": Ridge(),
            "target_aqi_day2": Ridge(),
            "target_aqi_day3": Ridge(),
        },
        best_params={
            "target_aqi_day1": {"max_depth": 8, "learning_rate": 0.05, "max_iter": 50, "min_samples_leaf": 20},
            "target_aqi_day2": {"max_depth": 8, "learning_rate": 0.05, "max_iter": 50, "min_samples_leaf": 20},
            "target_aqi_day3": {"max_depth": 8, "learning_rate": 0.05, "max_iter": 50, "min_samples_leaf": 20},
        },
        metrics={
            "mae_day1": mae_mean,
            "mae_day2": mae_mean,
            "mae_day3": mae_mean,
            "mae_mean": mae_mean,
            "rmse_mean": mae_mean + 1.0,
            "r2_mean": 0.5,
        },
    )


def _mock_mlp_result(mae_mean: float) -> ModelRunResult:
    scaler = StandardScaler().fit(np.array([[0.0], [1.0], [2.0]]))
    model = _build_mlp_model(input_dim=1)
    return ModelRunResult(
        model_name="tensorflow_mlp",
        fitted_models={"all_horizons": {"model": model, "scaler": scaler}},
        best_params={"all_horizons": {"epochs": 1, "batch_size": 2}},
        metrics={
            "mae_day1": mae_mean,
            "mae_day2": mae_mean,
            "mae_day3": mae_mean,
            "mae_mean": mae_mean,
            "rmse_mean": mae_mean + 1.0,
            "r2_mean": 0.5,
        },
    )
