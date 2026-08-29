"""Day 8 hourly feature-store refresh pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from src.data.open_meteo import (
    OpenMeteoClientError,
    fetch_air_quality,
    fetch_weather,
    load_city_config,
    merge_air_quality_and_weather,
)
from src.feature_store import (
    FEATURES_PRIMARY_KEYS,
    TARGETS_PRIMARY_KEYS,
    create_feature_view,
    insert_features,
    insert_targets,
    load_features,
    load_targets,
    verify_feature_group,
)
from src.features.build_features import build_features
from src.features.build_targets import build_targets

LOGGER = logging.getLogger(__name__)

LOOKBACK_HOURS = 72
FETCH_BUFFER_HOURS = 6


@dataclass(frozen=True)
class HourlyArtifacts:
    """Verification details from one hourly refresh run."""

    raw_frame: pd.DataFrame
    inserted_features: pd.DataFrame
    inserted_targets: pd.DataFrame
    feature_view: pd.DataFrame
    features_summary: dict[str, object]
    targets_summary: dict[str, object]


def run_hourly_refresh(now_utc: datetime | None = None) -> HourlyArtifacts:
    """Append new features and backfill newly-eligible targets into Hopsworks."""

    now_utc = now_utc or datetime.now(UTC)
    existing_features = load_features()
    if existing_features.empty:
        raise OpenMeteoClientError(
            "Feature store is empty; run the historical backfill before the hourly pipeline"
        )

    existing_targets = load_targets()
    latest_feature_time = _latest_event_time(existing_features, "features")
    latest_target_time = _latest_event_time(existing_targets, "targets", allow_empty=True)

    raw_frame = _fetch_incremental_raw_frame(
        fetch_start=_compute_fetch_start(latest_feature_time, latest_target_time),
        fetch_end=now_utc,
    )
    features = build_features(raw_frame)
    targets = build_targets(raw_frame)

    new_features = features.loc[features["event_time"] > latest_feature_time].reset_index(drop=True)
    new_targets = _select_new_targets(
        targets=targets,
        latest_target_time=latest_target_time,
        latest_available_event_time=raw_frame["event_time"].max(),
    )

    inserted_features = load_features() if new_features.empty else insert_features(new_features)
    inserted_targets = load_targets() if new_targets.empty else insert_targets(new_targets)
    feature_view = create_feature_view()

    return HourlyArtifacts(
        raw_frame=raw_frame,
        inserted_features=inserted_features,
        inserted_targets=inserted_targets,
        feature_view=feature_view,
        features_summary=verify_feature_group(inserted_features, FEATURES_PRIMARY_KEYS),
        targets_summary=verify_feature_group(inserted_targets, TARGETS_PRIMARY_KEYS),
    )


def _fetch_incremental_raw_frame(fetch_start: datetime, fetch_end: datetime) -> pd.DataFrame:
    city = load_city_config()
    start_date = fetch_start.date().isoformat()
    end_date = fetch_end.date().isoformat()

    try:
        air_quality = fetch_air_quality(start=start_date, end=end_date, city=city)
    except OpenMeteoClientError as exc:
        raise OpenMeteoClientError(f"Hourly refresh failed while fetching air quality: {exc}") from exc
    try:
        weather = fetch_weather(start=start_date, end=end_date, city=city)
    except OpenMeteoClientError as exc:
        raise OpenMeteoClientError(f"Hourly refresh failed while fetching weather: {exc}") from exc

    merged = merge_air_quality_and_weather(air_quality, weather)
    merged = merged.sort_values(["city_id", "event_time"]).reset_index(drop=True)
    cutoff = pd.Timestamp(fetch_end)
    merged = merged.loc[merged["event_time"] <= cutoff].reset_index(drop=True)
    if merged.empty:
        raise OpenMeteoClientError(
            f"Hourly refresh source frame is empty for requested range {start_date} to {end_date}"
        )
    return merged


def _compute_fetch_start(
    latest_feature_time: pd.Timestamp,
    latest_target_time: pd.Timestamp | None,
) -> datetime:
    candidate_starts = [
        latest_feature_time.to_pydatetime() - pd.Timedelta(hours=LOOKBACK_HOURS + FETCH_BUFFER_HOURS),
    ]
    if latest_target_time is not None:
        candidate_starts.append(latest_target_time.to_pydatetime())
    return min(candidate_starts)


def _latest_event_time(
    frame: pd.DataFrame,
    label: str,
    *,
    allow_empty: bool = False,
) -> pd.Timestamp | None:
    if frame.empty:
        if allow_empty:
            return None
        raise OpenMeteoClientError(f"Feature store {label} group is empty")
    latest = frame["event_time"].max()
    if not isinstance(latest, pd.Timestamp) or latest.tzinfo is None:
        raise OpenMeteoClientError(f"Feature store {label} group has a non-timezone-aware event_time")
    return latest


def _select_new_targets(
    *,
    targets: pd.DataFrame,
    latest_target_time: pd.Timestamp | None,
    latest_available_event_time: pd.Timestamp,
) -> pd.DataFrame:
    eligible_cutoff = latest_available_event_time - pd.Timedelta(hours=LOOKBACK_HOURS)
    selected = targets.loc[targets["event_time"] <= eligible_cutoff].copy()
    if latest_target_time is not None:
        selected = selected.loc[selected["event_time"] > latest_target_time]
    return selected.reset_index(drop=True)


if __name__ == "__main__":
    artifacts = run_hourly_refresh()
    LOGGER.info(
        "Hourly refresh completed",
        extra={
            "features_rows": artifacts.features_summary["row_count"],
            "targets_rows": artifacts.targets_summary["row_count"],
        },
    )
