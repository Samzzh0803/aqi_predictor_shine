"""Day 3 backfill pipeline for local Parquet feature-store fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.open_meteo import (
    OpenMeteoClientError,
    fetch_air_quality,
    fetch_weather,
    load_city_config,
    merge_air_quality_and_weather,
)
from src.feature_store.store import (
    FEATURES_PRIMARY_KEYS,
    TARGETS_PRIMARY_KEYS,
    create_feature_view,
    insert_features,
    insert_targets,
    load_feature_view,
    load_features,
    load_targets,
    verify_feature_group,
)
from src.features.build_features import build_features
from src.features.build_targets import build_targets

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillArtifacts:
    """Returned datasets and verification summaries from a backfill run."""

    raw_frame: pd.DataFrame
    features: pd.DataFrame
    targets: pd.DataFrame
    feature_view: pd.DataFrame
    features_summary: dict[str, object]
    targets_summary: dict[str, object]


def run_backfill(
    start_date: str,
    end_date: str,
    source_frame: pd.DataFrame | None = None,
    raw_cache_path: Path | None = None,
) -> BackfillArtifacts:
    """Run the Day 3 pipeline end to end and persist local feature-store artifacts."""

    raw_frame = _load_or_fetch_raw_frame(
        start_date=start_date,
        end_date=end_date,
        source_frame=source_frame,
        raw_cache_path=raw_cache_path,
    )
    features = build_features(raw_frame)
    targets = build_targets(raw_frame)

    insert_features(features)
    insert_targets(targets)

    stored_features = load_features()
    stored_targets = load_targets()
    create_feature_view(stored_features, stored_targets)
    reloaded_feature_view = load_feature_view()

    return BackfillArtifacts(
        raw_frame=raw_frame,
        features=stored_features,
        targets=stored_targets,
        feature_view=reloaded_feature_view,
        features_summary=verify_feature_group(stored_features, FEATURES_PRIMARY_KEYS),
        targets_summary=verify_feature_group(stored_targets, TARGETS_PRIMARY_KEYS),
    )


def validate_backfill_source_frame(frame: pd.DataFrame) -> None:
    """Reject unusable raw backfill inputs before feature engineering begins."""

    critical_columns = {
        "us_aqi": "AQI signal",
        "temperature_2m": "weather signal",
        "relative_humidity_2m": "weather signal",
        "wind_speed_10m": "weather signal",
    }
    all_null_columns = [
        column for column in critical_columns if column in frame.columns and frame[column].isna().all()
    ]
    if all_null_columns:
        raise OpenMeteoClientError(
            "Backfill source frame contains unusable all-null columns: "
            + ", ".join(sorted(all_null_columns))
        )


def _load_or_fetch_raw_frame(
    start_date: str,
    end_date: str,
    source_frame: pd.DataFrame | None,
    raw_cache_path: Path | None,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(
        microseconds=1
    )

    if source_frame is not None:
        frame = source_frame.copy()
    elif raw_cache_path is not None and raw_cache_path.exists():
        frame = pd.read_parquet(raw_cache_path)
    else:
        city = load_city_config()
        try:
            air_quality = fetch_air_quality(start=start_date, end=end_date, city=city)
        except OpenMeteoClientError as exc:
            raise OpenMeteoClientError(f"Backfill failed while fetching air quality: {exc}") from exc
        try:
            weather = fetch_weather(start=start_date, end=end_date, city=city)
        except OpenMeteoClientError as exc:
            raise OpenMeteoClientError(f"Backfill failed while fetching weather: {exc}") from exc
        frame = merge_air_quality_and_weather(air_quality, weather)

        if raw_cache_path is not None:
            raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(raw_cache_path, index=False)

    frame = frame.sort_values(["city_id", "event_time"]).reset_index(drop=True)
    frame = frame.loc[frame["event_time"].between(start_ts, end_ts)].reset_index(drop=True)
    if frame.empty:
        raise OpenMeteoClientError(
            f"Backfill source frame is empty for requested range {start_date} to {end_date}"
        )
    validate_backfill_source_frame(frame)
    return frame


if __name__ == "__main__":
    DEFAULT_START_DATE = "2022-08-01"
    DEFAULT_END_DATE = pd.Timestamp.utcnow().date().isoformat()
    raw_cache_candidates = sorted(Path("data/raw").glob("aqi_weather_*.parquet"))
    default_raw_cache = raw_cache_candidates[-1] if raw_cache_candidates else None

    artifacts = run_backfill(
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
        raw_cache_path=default_raw_cache,
    )
    LOGGER.info(
        "Backfill completed",
        extra={
            "features_rows": artifacts.features_summary["row_count"],
            "targets_rows": artifacts.targets_summary["row_count"],
        },
    )
