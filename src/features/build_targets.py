"""Target construction for 24/48/72-hour AQI forecasting horizons."""

from __future__ import annotations

from typing import Final

import pandas as pd

from src.data.open_meteo import OpenMeteoClientError

TARGET_COLUMNS: Final[list[str]] = [
    "target_aqi_day1",
    "target_aqi_day2",
    "target_aqi_day3",
]


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Build forward-looking AQI targets and drop rows with incomplete horizons."""

    _validate_input_frame(df)

    targets = df[["city_id", "event_time", "us_aqi"]].copy()
    grouped = targets.groupby("city_id", group_keys=False)["us_aqi"]

    targets["target_aqi_day1"] = grouped.transform(
        lambda s: _forward_window_mean(s, start_offset=1, end_offset=24)
    )
    targets["target_aqi_day2"] = grouped.transform(
        lambda s: _forward_window_mean(s, start_offset=25, end_offset=48)
    )
    targets["target_aqi_day3"] = grouped.transform(
        lambda s: _forward_window_mean(s, start_offset=49, end_offset=72)
    )

    targets = targets.drop(columns=["us_aqi"]).dropna().reset_index(drop=True)
    return targets


def _forward_window_mean(series: pd.Series, start_offset: int, end_offset: int) -> pd.Series:
    shifted_values = [series.shift(-offset) for offset in range(start_offset, end_offset + 1)]
    return pd.concat(shifted_values, axis=1).mean(axis=1, skipna=False)


def _validate_input_frame(df: pd.DataFrame) -> None:
    required_columns = {"city_id", "event_time", "us_aqi"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise OpenMeteoClientError(f"Input frame is missing required columns for target building: {missing_columns}")

    if not df["event_time"].is_monotonic_increasing:
        raise OpenMeteoClientError("Input frame must be sorted chronologically before target building")
