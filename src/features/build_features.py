"""Canonical feature engineering for AQI forecasting."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd

from src.config import get_config
from src.data.open_meteo import OpenMeteoClientError

RAW_AIR_QUALITY_COLUMNS: Final[list[str]] = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
]

RAW_WEATHER_COLUMNS: Final[list[str]] = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
]

KEY_COLUMNS: Final[list[str]] = ["city_id", "latitude", "longitude", "event_time"]

ENGINEERED_FEATURE_COLUMNS: Final[list[str]] = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "wind_dir_sin",
    "wind_dir_cos",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "aqi_lag_48h",
    "aqi_lag_72h",
    "pm25_lag_6h",
    "pm25_lag_24h",
    "aqi_mean_3h",
    "aqi_mean_6h",
    "aqi_mean_12h",
    "aqi_mean_24h",
    "aqi_mean_72h",
    "aqi_std_24h",
    "aqi_std_72h",
    "pm25_mean_24h",
    "temperature_mean_24h",
    "humidity_mean_24h",
    "wind_mean_24h",
    "aqi_change_6h",
    "aqi_change_24h",
    "pm25_change_24h",
]

FEATURE_COLUMNS: Final[list[str]] = [
    *KEY_COLUMNS,
    *RAW_AIR_QUALITY_COLUMNS,
    *RAW_WEATHER_COLUMNS,
    *ENGINEERED_FEATURE_COLUMNS,
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the locked, leakage-safe feature set for the AQI model."""

    _validate_input_frame(df)

    features = df.copy()
    features = features.sort_values(["city_id", "event_time"]).reset_index(drop=True)

    grouped = features.groupby("city_id", group_keys=False)
    local_event_time = features["event_time"].dt.tz_convert(_get_city_timezone())

    hour = local_event_time.dt.hour
    day_of_week = local_event_time.dt.dayofweek
    month = local_event_time.dt.month
    features["hour_sin"] = np.sin(2 * math.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * math.pi * hour / 24)
    features["dow_sin"] = np.sin(2 * math.pi * day_of_week / 7)
    features["dow_cos"] = np.cos(2 * math.pi * day_of_week / 7)
    features["month_sin"] = np.sin(2 * math.pi * (month - 1) / 12)
    features["month_cos"] = np.cos(2 * math.pi * (month - 1) / 12)
    features["is_weekend"] = (day_of_week >= 5).astype(int)

    wind_radians = np.deg2rad(features["wind_direction_10m"])
    features["wind_dir_sin"] = np.sin(wind_radians)
    features["wind_dir_cos"] = np.cos(wind_radians)

    aqi_shifted = grouped["us_aqi"].shift(1)
    pm25_shifted = grouped["pm2_5"].shift(1)
    temp_shifted = grouped["temperature_2m"].shift(1)
    humidity_shifted = grouped["relative_humidity_2m"].shift(1)
    wind_shifted = grouped["wind_speed_10m"].shift(1)

    for lag in [3, 6, 12, 24, 48, 72]:
        features[f"aqi_lag_{lag}h"] = grouped["us_aqi"].shift(lag)

    for lag in [6, 24]:
        features[f"pm25_lag_{lag}h"] = grouped["pm2_5"].shift(lag)

    for window in [3, 6, 12, 24, 72]:
        features[f"aqi_mean_{window}h"] = (
            aqi_shifted.groupby(features["city_id"]).rolling(window=window, min_periods=window).mean().reset_index(level=0, drop=True)
        )

    for window in [24, 72]:
        features[f"aqi_std_{window}h"] = (
            aqi_shifted.groupby(features["city_id"]).rolling(window=window, min_periods=window).std().reset_index(level=0, drop=True)
        )

    features["pm25_mean_24h"] = (
        pm25_shifted.groupby(features["city_id"]).rolling(window=24, min_periods=24).mean().reset_index(level=0, drop=True)
    )
    features["temperature_mean_24h"] = (
        temp_shifted.groupby(features["city_id"]).rolling(window=24, min_periods=24).mean().reset_index(level=0, drop=True)
    )
    features["humidity_mean_24h"] = (
        humidity_shifted.groupby(features["city_id"]).rolling(window=24, min_periods=24).mean().reset_index(level=0, drop=True)
    )
    features["wind_mean_24h"] = (
        wind_shifted.groupby(features["city_id"]).rolling(window=24, min_periods=24).mean().reset_index(level=0, drop=True)
    )

    features["aqi_change_6h"] = features["us_aqi"] - grouped["us_aqi"].shift(6)
    features["aqi_change_24h"] = features["us_aqi"] - grouped["us_aqi"].shift(24)
    features["pm25_change_24h"] = features["pm2_5"] - grouped["pm2_5"].shift(24)

    return features[FEATURE_COLUMNS]


def get_feature_columns() -> list[str]:
    """Return the exact ordered feature columns used across the project."""

    return FEATURE_COLUMNS.copy()


def _get_city_timezone() -> str:
    config = get_config()
    city = config.get("city", {})
    timezone = city.get("timezone")
    if not timezone:
        raise OpenMeteoClientError("City timezone is missing from config/config.yaml")
    return str(timezone)


def _validate_input_frame(df: pd.DataFrame) -> None:
    required_columns = set(KEY_COLUMNS + RAW_AIR_QUALITY_COLUMNS + RAW_WEATHER_COLUMNS)
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise OpenMeteoClientError(f"Input frame is missing required columns for feature building: {missing_columns}")

    if not df["event_time"].is_monotonic_increasing:
        raise OpenMeteoClientError("Input frame must be sorted chronologically before feature building")
    if df["event_time"].dt.tz is None:
        raise OpenMeteoClientError("event_time must be timezone-aware UTC before feature building")
