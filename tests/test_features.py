"""Day 2 tests for feature engineering and leakage safety."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.data.open_meteo import OpenMeteoClientError
from src.features.build_features import ENGINEERED_FEATURE_COLUMNS, build_features


def _load_raw_frame() -> pd.DataFrame:
    return pd.read_parquet("data/raw/aqi_weather_2022-08-01_2026-08-24.parquet").sort_values("event_time").reset_index(drop=True)


def test_build_features_outputs_locked_columns() -> None:
    frame = _load_raw_frame()
    features = build_features(frame)

    for column in ENGINEERED_FEATURE_COLUMNS:
        assert column in features.columns, f"Missing locked feature: {column}"


def test_build_features_raises_on_unsorted_input() -> None:
    frame = _load_raw_frame()
    shuffled = frame.sample(frac=1, random_state=0).reset_index(drop=True)
    with pytest.raises(OpenMeteoClientError, match="sorted chronologically"):
        build_features(shuffled)


def test_build_features_uses_local_time_for_calendar_features() -> None:
    frame = _load_raw_frame().head(1)
    features = build_features(frame)

    # 2022-08-01 00:00:00Z is 05:00 local time in Asia/Karachi.
    expected_hour_sin = math.sin(2 * math.pi * 5 / 24)
    expected_hour_cos = math.cos(2 * math.pi * 5 / 24)

    assert features.loc[0, "hour_sin"] == expected_hour_sin
    assert features.loc[0, "hour_cos"] == expected_hour_cos


def test_random_sample_features_match_truncated_recomputation() -> None:
    frame = _load_raw_frame()
    features = build_features(frame)

    eligible = features.dropna(subset=["aqi_mean_72h", "aqi_lag_72h"]).reset_index(drop=True)
    sample_positions = np.linspace(0, len(eligible) - 1, num=5, dtype=int)
    columns_to_compare = [
        "aqi_lag_72h",
        "aqi_mean_24h",
        "aqi_mean_72h",
        "aqi_std_72h",
        "pm25_mean_24h",
        "temperature_mean_24h",
        "aqi_change_24h",
    ]

    for position in sample_positions:
        event_time = eligible.loc[position, "event_time"]
        truncated = frame.loc[frame["event_time"] <= event_time].copy()
        recomputed = build_features(truncated)
        recomputed_row = (
            recomputed.loc[recomputed["event_time"] == event_time, columns_to_compare]
            .iloc[0]
            .astype(float)
        )
        full_row = eligible.loc[position, columns_to_compare].astype(float)
        pd.testing.assert_series_equal(full_row, recomputed_row, check_names=False)
