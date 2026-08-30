"""Day 4 tests for chronological splitting and training data preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.feature_store import store as store_module
from src.feature_store.store import (
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    insert_features,
    insert_targets,
)
from src.features.build_features import build_features
from src.features.build_targets import build_targets
from src.pipelines.train import (
    chronological_split,
    day4_gate_passes,
    evaluate_baseline,
    evaluate_day4_models,
    get_model_feature_columns,
    load_training_frame,
    prepare_modeling_frame,
)
from tests.test_feature_store import FakeFeatureStore


@pytest.fixture()
def fake_feature_store(monkeypatch: pytest.MonkeyPatch) -> FakeFeatureStore:
    fs = FakeFeatureStore()
    monkeypatch.setattr(store_module, "_get_feature_store", lambda: fs)
    monkeypatch.setattr(
        "src.feature_store.store.load_city_config",
        lambda: type(
            "City",
            (),
            {
                "city_id": "lahore",
                "name": "Lahore",
                "latitude": 31.5497,
                "longitude": 74.3436,
                "timezone": "Asia/Karachi",
            },
        )(),
    )
    return fs


def _load_raw_frame() -> pd.DataFrame:
    return (
        pd.read_parquet("data/raw/aqi_weather_2022-08-01_2026-08-24.parquet")
        .sort_values("event_time")
        .reset_index(drop=True)
    )


def test_chronological_split_keeps_time_order() -> None:
    raw = _load_raw_frame().iloc[96:500].reset_index(drop=True)
    features = build_features(raw)
    targets = build_targets(raw)
    feature_view = features.merge(targets, on=["city_id", "event_time"], how="inner")

    split = chronological_split(feature_view)

    assert split.train["event_time"].max() < split.val["event_time"].min()
    assert split.val["event_time"].max() < split.test["event_time"].min()
    assert len(split.test) == round(len(feature_view) * 0.15)


def test_load_training_frame_rebuilds_feature_view_when_missing(
    fake_feature_store: FakeFeatureStore,
) -> None:
    raw = _load_raw_frame().iloc[96:500].reset_index(drop=True)
    insert_features(build_features(raw))
    insert_targets(build_targets(raw))

    training_frame = load_training_frame()

    assert (FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION) in fake_feature_store._feature_views
    assert not training_frame.empty
    assert training_frame["event_time"].is_monotonic_increasing


def test_baselines_use_expected_columns() -> None:
    raw = _load_raw_frame().iloc[96:500].reset_index(drop=True)
    feature_view = build_features(raw).merge(
        build_targets(raw),
        on=["city_id", "event_time"],
        how="inner",
    )
    split = chronological_split(feature_view)

    persistence = evaluate_baseline("persistence", split.test, "us_aqi")
    seasonal = evaluate_baseline("seasonal_naive", split.test, "aqi_mean_24h")

    assert persistence.metrics["mae_mean"] >= 0
    assert seasonal.metrics["mae_mean"] >= 0


def test_get_model_feature_columns_excludes_targets_and_keys() -> None:
    raw = _load_raw_frame().iloc[96:500].reset_index(drop=True)
    feature_view = build_features(raw).merge(
        build_targets(raw),
        on=["city_id", "event_time"],
        how="inner",
    )

    feature_columns = get_model_feature_columns(feature_view)

    assert "target_aqi_day1" not in feature_columns
    assert "event_time" not in feature_columns
    assert "us_aqi" in feature_columns


def test_prepare_modeling_frame_drops_rows_with_incomplete_features() -> None:
    raw = _load_raw_frame().iloc[96:500].reset_index(drop=True)
    feature_view = build_features(raw).merge(
        build_targets(raw),
        on=["city_id", "event_time"],
        how="inner",
    )

    modeling_frame = prepare_modeling_frame(feature_view)

    feature_columns = get_model_feature_columns(modeling_frame)
    assert not modeling_frame[feature_columns].isna().any().any()


def test_evaluate_day4_models_persists_csv_and_markdown_without_tabulate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "city_id": ["lahore"] * 6,
            "event_time": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"),
            "latitude": [31.5497] * 6,
            "longitude": [74.3436] * 6,
            "us_aqi": [100.0, 102.0, 101.0, 104.0, 103.0, 105.0],
            "aqi_mean_24h": [99.0, 101.0, 100.0, 103.0, 102.0, 104.0],
            "feature_x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target_aqi_day1": [101.0, 103.0, 102.0, 105.0, 104.0, 106.0],
            "target_aqi_day2": [102.0, 104.0, 103.0, 106.0, 105.0, 107.0],
            "target_aqi_day3": [103.0, 105.0, 104.0, 107.0, 106.0, 108.0],
        }
    )
    monkeypatch.setattr("src.pipelines.train.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("src.pipelines.train.MODEL_COMPARISON_CSV", tmp_path / "comparison.csv")
    monkeypatch.setattr("src.pipelines.train.MODEL_COMPARISON_MD", tmp_path / "comparison.md")
    monkeypatch.setattr("src.pipelines.train.load_training_frame", lambda: frame)
    monkeypatch.setattr(
        "src.pipelines.train.train_ridge_models",
        lambda split, feature_columns: _mock_model_result("ridge", 1.0),
    )
    monkeypatch.setattr(
        "src.pipelines.train.train_random_forest_models",
        lambda split, feature_columns: _mock_model_result("random_forest", 2.0),
    )
    monkeypatch.setattr(
        "src.pipelines.train.train_hist_gradient_boosting_models",
        lambda split, feature_columns: _mock_model_result("hist_gradient_boosting", 3.0),
    )

    comparison = evaluate_day4_models()

    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "comparison.md").exists()
    markdown = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "| model" in markdown
    assert comparison.iloc[0]["model"] == "persistence"


def test_day4_gate_ignores_seasonal_naive_when_ml_models_do_not_win() -> None:
    comparison = pd.DataFrame(
        {
            "model": ["persistence", "seasonal_naive", "ridge", "random_forest"],
            "mae_mean": [10.0, 9.0, 10.5, 11.0],
        }
    )

    assert not day4_gate_passes(comparison)


def test_day4_gate_passes_when_real_model_beats_persistence() -> None:
    comparison = pd.DataFrame(
        {
            "model": ["persistence", "seasonal_naive", "ridge", "random_forest"],
            "mae_mean": [10.0, 9.5, 8.5, 11.0],
        }
    )

    assert day4_gate_passes(comparison)


def _mock_model_result(model_name: str, mae_mean: float):
    from src.pipelines.train import ModelRunResult

    return ModelRunResult(
        model_name=model_name,
        fitted_models={},
        best_params={},
        metrics={
            "mae_day1": mae_mean,
            "mae_day2": mae_mean,
            "mae_day3": mae_mean,
            "mae_mean": mae_mean,
            "rmse_mean": mae_mean + 1.0,
            "r2_mean": 0.5,
        },
    )
