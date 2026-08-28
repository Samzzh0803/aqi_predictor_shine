"""Day 6 tests for local champion inference and AQI category helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.data.open_meteo import OpenMeteoClientError
from src.inference.aqi import aqi_alert_level, aqi_category
from src.inference.predictor import predict_next_3_days
from src.models.registry import RegisteredModelVersion


def test_aqi_category_boundaries() -> None:
    assert aqi_category(50) == "Good"
    assert aqi_category(51) == "Moderate"
    assert aqi_category(100) == "Moderate"
    assert aqi_category(101) == "Unhealthy for Sensitive Groups"
    assert aqi_category(150) == "Unhealthy for Sensitive Groups"
    assert aqi_category(151) == "Unhealthy"
    assert aqi_category(200) == "Unhealthy"
    assert aqi_category(201) == "Very Unhealthy"
    assert aqi_category(300) == "Very Unhealthy"
    assert aqi_category(301) == "Hazardous"
    assert aqi_category(500) == "Hazardous"


def test_aqi_alert_boundaries() -> None:
    assert aqi_alert_level(150) == "none"
    assert aqi_alert_level(151) == "warning"
    assert aqi_alert_level(200) == "warning"
    assert aqi_alert_level(201) == "critical"
    assert aqi_alert_level(300) == "critical"
    assert aqi_alert_level(301) == "hazardous"
    assert aqi_alert_level(500) == "hazardous"


def test_predict_next_3_days_loads_local_champion_and_clips_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_list = ["feature_a", "feature_b"]
    model_paths = {}
    for index, target in enumerate(["target_aqi_day1", "target_aqi_day2", "target_aqi_day3"], start=1):
        model = Ridge()
        model.coef_ = np.array([0.0, 0.0])
        model.intercept_ = 600.0 if index == 1 else (-10.0 if index == 2 else 175.0)
        model.n_features_in_ = 2
        artifact_path = tmp_path / f"{target}.joblib"
        joblib.dump(model, artifact_path)
        model_paths[target] = str(artifact_path)

    champion = RegisteredModelVersion(
        model_name="pearls_aqi_forecaster",
        version=7,
        model_type="ridge",
        metrics={"mae_mean": 1.0, "selection_mae_mean": 1.0},
        feature_list=feature_list,
        trained_at="2026-08-27T00:00:00+00:00",
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-08-27T00:00:00+00:00",
        artifact_paths=model_paths,
    )
    latest_features = pd.DataFrame(
        [
            {
                "city_id": "lahore",
                "event_time": pd.Timestamp(datetime.now(UTC)),
                "latitude": 31.5497,
                "longitude": 74.3436,
                "us_aqi": 123.0,
                "feature_a": 1.0,
                "feature_b": 2.0,
            }
        ]
    )

    monkeypatch.setattr("src.inference.predictor.get_champion", lambda: champion)
    monkeypatch.setattr("src.inference.predictor.load_features", lambda: latest_features)
    monkeypatch.setattr("src.inference.predictor._configured_city_name", lambda: "Lahore")
    monkeypatch.setattr(
        "src.inference.predictor.load_registered_models",
        lambda version: {name: joblib.load(path) for name, path in version.artifact_paths.items()},
    )

    prediction = predict_next_3_days()

    assert prediction.model_version == 7
    assert prediction.current_aqi == 123.0
    assert [point.aqi for point in prediction.forecast] == [500.0, 0.0, 175.0]
    assert [point.alert for point in prediction.forecast] == ["hazardous", "none", "warning"]


def test_predict_next_3_days_rejects_stale_features(monkeypatch: pytest.MonkeyPatch) -> None:
    champion = RegisteredModelVersion(
        model_name="pearls_aqi_forecaster",
        version=1,
        model_type="ridge",
        metrics={"mae_mean": 1.0, "selection_mae_mean": 1.0},
        feature_list=["feature_a"],
        trained_at="2026-08-27T00:00:00+00:00",
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-08-27T00:00:00+00:00",
        artifact_paths={},
    )
    stale_features = pd.DataFrame(
        [
            {
                "city_id": "lahore",
                "event_time": pd.Timestamp("2026-08-24T00:00:00+00:00"),
                "latitude": 31.5497,
                "longitude": 74.3436,
                "us_aqi": 123.0,
                "feature_a": 1.0,
            }
        ]
    )

    monkeypatch.setattr("src.inference.predictor.get_champion", lambda: champion)
    monkeypatch.setattr("src.inference.predictor.load_features", lambda: stale_features)
    monkeypatch.setattr("src.inference.predictor._configured_city_name", lambda: "Lahore")

    with pytest.raises(OpenMeteoClientError, match="stale"):
        predict_next_3_days()


def test_predict_next_3_days_rejects_missing_registered_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    champion = RegisteredModelVersion(
        model_name="pearls_aqi_forecaster",
        version=1,
        model_type="ridge",
        metrics={"mae_mean": 1.0, "selection_mae_mean": 1.0},
        feature_list=["feature_a", "feature_b"],
        trained_at="2026-08-27T00:00:00+00:00",
        data_start="2026-01-01T00:00:00+00:00",
        data_end="2026-08-27T00:00:00+00:00",
        artifact_paths={},
    )
    latest_features = pd.DataFrame(
        [
            {
                "city_id": "lahore",
                "event_time": pd.Timestamp(datetime.now(UTC)),
                "latitude": 31.5497,
                "longitude": 74.3436,
                "us_aqi": 123.0,
                "feature_a": 1.0,
            }
        ]
    )

    monkeypatch.setattr("src.inference.predictor.get_champion", lambda: champion)
    monkeypatch.setattr("src.inference.predictor.load_features", lambda: latest_features)
    monkeypatch.setattr("src.inference.predictor._configured_city_name", lambda: "Lahore")

    with pytest.raises(OpenMeteoClientError, match="missing registered feature columns"):
        predict_next_3_days()
