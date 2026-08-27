"""Day 2 tests for target construction and leakage checks."""

from __future__ import annotations

import pandas as pd

from src.features.build_features import build_features
from src.features.build_targets import TARGET_COLUMNS, build_targets


def _load_raw_frame() -> pd.DataFrame:
    return pd.read_parquet("data/raw/aqi_weather_2022-08-01_2026-08-24.parquet").sort_values("event_time").reset_index(drop=True)


def test_build_targets_creates_expected_columns_only() -> None:
    frame = _load_raw_frame()
    targets = build_targets(frame)

    assert list(targets.columns) == ["city_id", "event_time", *TARGET_COLUMNS]


def test_build_targets_drops_incomplete_trailing_rows() -> None:
    frame = _load_raw_frame()
    targets = build_targets(frame)

    assert targets["event_time"].max() == frame["event_time"].iloc[-73]
    expected_first_valid = frame.loc[frame["us_aqi"].notna(), "event_time"].min()
    assert targets["event_time"].min() == expected_first_valid - pd.Timedelta(hours=1)


def test_feature_target_leakage_guards_hold() -> None:
    frame = _load_raw_frame()
    features = build_features(frame)
    targets = build_targets(frame)
    joined = features.merge(targets, on=["city_id", "event_time"], how="inner")

    feature_columns = [column for column in features.columns if column not in {"city_id", "latitude", "longitude", "event_time"}]
    assert not any(column in TARGET_COLUMNS for column in feature_columns)

    correlations = joined[feature_columns + TARGET_COLUMNS].corr(numeric_only=True)
    for target_column in TARGET_COLUMNS:
        target_corr = correlations.loc[feature_columns, target_column].abs().fillna(0.0)
        assert (target_corr < 0.999).all(), f"Found near-perfect correlation with {target_column}: {target_corr[target_corr >= 0.999].to_dict()}"
