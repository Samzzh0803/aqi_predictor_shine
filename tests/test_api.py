"""Day 6 API tests with TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.open_meteo import OpenMeteoClientError
from src.inference.predictor import PredictionArtifacts, PredictionPoint
from src.models.registry import RegisteredModelVersion


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_forecast_endpoint_returns_predictions(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.api.main.predict_next_3_days",
        lambda: PredictionArtifacts(
            city="Lahore",
            generated_at="2026-08-27T01:00:00+00:00",
            model_version=4,
            model_type="ridge",
            current_aqi=123.0,
            forecast=[
                PredictionPoint("day_1", 137.0, "Unhealthy for Sensitive Groups", "none"),
                PredictionPoint("day_2", 149.0, "Unhealthy for Sensitive Groups", "none"),
                PredictionPoint("day_3", 161.0, "Unhealthy", "warning"),
            ],
        ),
    )

    response = client.get("/forecast")

    assert response.status_code == 200
    assert response.json()["forecast"][2]["alert"] == "warning"


def test_predict_scenario_endpoint_forwards_overrides_and_returns_predictions(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}

    def fake_predict_scenario(overrides: dict[str, float]) -> PredictionArtifacts:
        captured.update(overrides)
        return PredictionArtifacts(
            city="Karachi",
            generated_at="2026-08-30T01:00:00+00:00",
            model_version=5,
            model_type="hist_gradient_boosting",
            current_aqi=61.0,
            forecast=[
                PredictionPoint("day_1", 90.0, "Moderate", "none"),
                PredictionPoint("day_2", 95.0, "Moderate", "none"),
                PredictionPoint("day_3", 99.0, "Moderate", "none"),
            ],
        )

    monkeypatch.setattr("src.api.main.predict_scenario", fake_predict_scenario)

    response = client.post("/predict-scenario", json={"pm2_5": 200.0, "temperature_2m": None})

    assert response.status_code == 200
    assert response.json()["forecast"][0]["aqi"] == 90.0
    # None-valued fields must be dropped, not forwarded as an override of None
    assert captured == {"pm2_5": 200.0}


def test_predict_scenario_endpoint_returns_503_for_invalid_override(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.api.main.predict_scenario",
        lambda overrides: (_ for _ in ()).throw(OpenMeteoClientError("Cannot override these columns: ['us_aqi']")),
    )

    response = client.post("/predict-scenario", json={"pm2_5": 10.0})

    assert response.status_code == 503
    assert "Cannot override" in response.json()["detail"]


def test_forecast_endpoint_returns_503_for_stale_or_missing_features(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.api.main.predict_next_3_days",
        lambda: (_ for _ in ()).throw(OpenMeteoClientError("Latest features are stale")),
    )

    response = client.get("/forecast")

    assert response.status_code == 503
    assert response.json()["detail"] == "Latest features are stale"


def test_model_info_endpoint_returns_champion_metadata(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.api.main.get_champion",
        lambda: RegisteredModelVersion(
            model_name="pearls_aqi_forecaster",
            version=3,
            model_type="ridge",
            metrics={"mae_mean": 17.0, "selection_mae_mean": 18.0},
            feature_list=["a", "b"],
            trained_at="2026-08-27T00:00:00+00:00",
            data_start="2026-01-01T00:00:00+00:00",
            data_end="2026-08-27T00:00:00+00:00",
            artifact_paths={"target_aqi_day1": "model.joblib"},
        ),
    )

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["feature_count"] == 2


def test_history_endpoint_returns_recent_points(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "city_id": ["lahore", "lahore", "lahore"],
            "event_time": pd.to_datetime(
                [
                    "2026-08-24T00:00:00+00:00",
                    "2026-08-25T00:00:00+00:00",
                    "2026-08-26T00:00:00+00:00",
                ],
                utc=True,
            ),
            "us_aqi": [100.0, 110.0, 120.0],
        }
    )
    monkeypatch.setattr("src.api.main.load_features", lambda: frame)
    monkeypatch.setattr("src.api.main.get_config", lambda: {"city": {"name": "Lahore"}})

    response = client.get("/history", params={"days": 1})

    assert response.status_code == 200
    assert response.json()["city"] == "Lahore"
    assert len(response.json()["points"]) == 2


def test_model_info_endpoint_returns_503_when_registry_empty(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.api.main.get_champion",
        lambda: (_ for _ in ()).throw(OpenMeteoClientError("No registered models found")),
    )

    response = client.get("/model-info")

    assert response.status_code == 503
    assert response.json()["detail"] == "No registered models found"


def test_history_endpoint_returns_503_when_feature_store_read_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.api.main.load_features",
        lambda: (_ for _ in ()).throw(OpenMeteoClientError("duplicate keys in feature store")),
    )

    response = client.get("/history")

    assert response.status_code == 503
    assert response.json()["detail"] == "duplicate keys in feature store"


def test_model_info_endpoint_returns_503_for_unexpected_registry_exception(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.api.main.get_champion",
        lambda: (_ for _ in ()).throw(RuntimeError("registry download failed")),
    )

    response = client.get("/model-info")

    assert response.status_code == 503
    assert "Model registry backend failed" in response.json()["detail"]
