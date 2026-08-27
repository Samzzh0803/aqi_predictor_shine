"""Champion-model inference using the local fallback registry and feature store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from src.config import get_config
from src.data.open_meteo import OpenMeteoClientError
from src.feature_store import load_features
from src.inference.aqi import aqi_alert_level, aqi_category
from src.models.registry import RegisteredModelVersion, get_champion, load_registered_models


@dataclass(frozen=True)
class PredictionPoint:
    """One horizon forecast with AQI category and alert level."""

    horizon: str
    aqi: float
    category: str
    alert: str


@dataclass(frozen=True)
class PredictionArtifacts:
    """Complete Day 6 forecast payload."""

    city: str
    generated_at: str
    model_version: int
    model_type: str
    current_aqi: float
    forecast: list[PredictionPoint]


def predict_next_3_days(
    city: str | None = None,
    *,
    max_feature_age_hours: int = 48,
) -> PredictionArtifacts:
    """Load the champion, validate feature freshness/order, and forecast three horizons."""

    champion = get_champion()
    latest_features = load_latest_feature_row(max_feature_age_hours=max_feature_age_hours)
    requested_city = city or _configured_city_name()
    if requested_city != _configured_city_name():
        raise OpenMeteoClientError(
            f"Only the configured city is supported in v1: expected {_configured_city_name()}, got {requested_city}"
        )

    feature_vector = _build_ordered_feature_vector(latest_features, champion.feature_list)
    predictions = _predict_horizons(champion=champion, feature_vector=feature_vector)
    forecast = [
        PredictionPoint(
            horizon=f"day_{index}",
            aqi=value,
            category=aqi_category(value),
            alert=aqi_alert_level(value),
        )
        for index, value in enumerate(predictions, start=1)
    ]
    return PredictionArtifacts(
        city=requested_city,
        generated_at=datetime.now(UTC).isoformat(),
        model_version=champion.version,
        model_type=champion.model_type,
        current_aqi=float(latest_features["us_aqi"]),
        forecast=forecast,
    )


def load_latest_feature_row(*, max_feature_age_hours: int = 48) -> pd.Series:
    """Return the most recent feature row and reject stale data."""

    features = load_features().sort_values(["city_id", "event_time"]).reset_index(drop=True)
    if features.empty:
        raise OpenMeteoClientError("Feature store is empty; run backfill or hourly feature generation first")
    latest = features.iloc[-1]
    latest_event_time = latest["event_time"]
    if not isinstance(latest_event_time, pd.Timestamp) or latest_event_time.tzinfo is None:
        raise OpenMeteoClientError("Latest feature row has a non-timezone-aware event_time")
    now_utc = datetime.now(UTC)
    age = now_utc - latest_event_time.to_pydatetime()
    if age > timedelta(hours=max_feature_age_hours):
        raise OpenMeteoClientError(
            "Latest features are stale: "
            f"event_time={latest_event_time.isoformat()} is older than {max_feature_age_hours} hours"
        )
    return latest


def _build_ordered_feature_vector(latest_features: pd.Series, feature_list: list[str]) -> pd.DataFrame:
    missing = [column for column in feature_list if column not in latest_features.index]
    if missing:
        raise OpenMeteoClientError(f"Latest feature row is missing registered feature columns: {missing}")

    feature_vector = latest_features.loc[feature_list]
    if feature_vector.index.tolist() != feature_list:
        raise OpenMeteoClientError("Inference feature order does not match the registered feature list")

    ordered_frame = pd.DataFrame([feature_vector.to_dict()], columns=feature_list)
    if ordered_frame.isna().any(axis=None):
        missing_columns = ordered_frame.columns[ordered_frame.isna().any()].tolist()
        raise OpenMeteoClientError(
            f"Latest feature row contains null values in required model inputs: {missing_columns}"
        )
    return ordered_frame.astype(float)


def _predict_horizons(champion: RegisteredModelVersion, feature_vector: pd.DataFrame) -> list[float]:
    loaded_models = load_registered_models(champion)

    if champion.model_type == "tensorflow_mlp":
        bundle = loaded_models["all_horizons"]
        scaled = bundle["scaler"].transform(feature_vector).astype("float32")
        matrix = bundle["model"].predict(scaled, verbose=0)
        if matrix.shape[1] != 3:
            raise OpenMeteoClientError("TensorFlow champion must predict exactly three horizons")
        return [_clip_prediction(value) for value in matrix[0].tolist()]

    ordered_targets = ["target_aqi_day1", "target_aqi_day2", "target_aqi_day3"]
    predictions: list[float] = []
    for target in ordered_targets:
        if target not in loaded_models:
            raise OpenMeteoClientError(f"Registered champion is missing a fitted artifact for {target}")
        prediction = float(loaded_models[target].predict(feature_vector)[0])
        predictions.append(_clip_prediction(prediction))
    return predictions


def _clip_prediction(value: float) -> float:
    return float(np.clip(value, 0.0, 500.0))


def _configured_city_name() -> str:
    config = get_config()
    city = config.get("city", {})
    name = city.get("name")
    if not name:
        raise OpenMeteoClientError("Configured city name is missing from config/config.yaml")
    return str(name)
