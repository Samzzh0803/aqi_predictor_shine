"""Local Parquet feature-store fallback with Hopsworks-compatible interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DatetimeTZDtype

from src.data.open_meteo import OpenMeteoClientError

FEATURES_PRIMARY_KEYS: Final[list[str]] = ["city_id", "event_time"]
TARGETS_PRIMARY_KEYS: Final[list[str]] = ["city_id", "event_time"]

_FEATURE_STORE_ROOT = Path("data") / "feature_store"
_FEATURES_PATH = _FEATURE_STORE_ROOT / "aqi_features_v1.parquet"
_TARGETS_PATH = _FEATURE_STORE_ROOT / "aqi_targets_v1.parquet"
_FEATURE_VIEW_PATH = _FEATURE_STORE_ROOT / "aqi_fv_v1.parquet"


def insert_features(frame: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """Upsert feature rows into the local fallback store."""

    return _upsert_frame(
        frame=frame,
        path=_FEATURES_PATH if path is None else path,
        key_columns=FEATURES_PRIMARY_KEYS,
    )


def insert_targets(frame: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """Upsert target rows into the local fallback store."""

    return _upsert_frame(
        frame=frame,
        path=_TARGETS_PATH if path is None else path,
        key_columns=TARGETS_PRIMARY_KEYS,
    )


def load_features(path: Path | None = None) -> pd.DataFrame:
    """Load the local features feature group."""

    return _load_frame(_FEATURES_PATH if path is None else path)


def load_targets(path: Path | None = None) -> pd.DataFrame:
    """Load the local targets feature group."""

    return _load_frame(_TARGETS_PATH if path is None else path)


def create_feature_view(
    features: pd.DataFrame | None = None,
    targets: pd.DataFrame | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """Create and persist the Day 3 feature view join."""

    features = load_features() if features is None else features.copy()
    targets = load_targets() if targets is None else targets.copy()
    feature_view = features.merge(
        targets,
        on=["city_id", "event_time"],
        how="inner",
        validate="one_to_one",
    )
    feature_view = feature_view.sort_values(["city_id", "event_time"]).reset_index(drop=True)
    output_path = _FEATURE_VIEW_PATH if path is None else path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_view.to_parquet(output_path, index=False)
    return feature_view


def load_feature_view(path: Path | None = None) -> pd.DataFrame:
    """Load the materialized local feature view."""

    return _load_frame(_FEATURE_VIEW_PATH if path is None else path)


def verify_feature_group(frame: pd.DataFrame, key_columns: list[str]) -> dict[str, object]:
    """Return the Day 3 verification summary for a feature group."""

    missing = sorted(set(key_columns).difference(frame.columns))
    if missing:
        raise OpenMeteoClientError(f"Feature-store frame is missing key columns: {missing}")
    if "event_time" not in frame.columns:
        raise OpenMeteoClientError("Feature-store frame must include event_time")
    if not isinstance(frame["event_time"].dtype, DatetimeTZDtype):
        raise OpenMeteoClientError("Feature-store event_time must be timezone-aware UTC")
    duplicate_count = int(frame.duplicated(subset=key_columns).sum())
    return {
        "row_count": int(len(frame)),
        "date_min": frame["event_time"].min(),
        "date_max": frame["event_time"].max(),
        "null_counts": frame.isna().sum().to_dict(),
        "duplicate_count": duplicate_count,
        "head": frame.head(),
        "tail": frame.tail(),
    }


def _upsert_frame(frame: pd.DataFrame, path: Path, key_columns: list[str]) -> pd.DataFrame:
    _validate_frame(frame=frame, key_columns=key_columns)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.read_parquet(path) if path.exists() else frame.iloc[0:0].copy()
    if not existing.empty:
        _validate_frame(frame=existing, key_columns=key_columns)

    combined = pd.concat([existing, frame], ignore_index=True)
    combined = combined.sort_values(key_columns).drop_duplicates(subset=key_columns, keep="last")
    combined = combined.reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return combined


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise OpenMeteoClientError(f"Expected feature-store artifact is missing: {path}")

    frame = pd.read_parquet(path)
    _validate_frame(frame=frame, key_columns=["city_id", "event_time"])
    return frame


def _validate_frame(frame: pd.DataFrame, key_columns: list[str]) -> None:
    missing = sorted(set(key_columns).difference(frame.columns))
    if missing:
        raise OpenMeteoClientError(f"Feature-store frame is missing key columns: {missing}")
    if "event_time" not in frame.columns:
        raise OpenMeteoClientError("Feature-store frame must include event_time")
    if not isinstance(frame["event_time"].dtype, DatetimeTZDtype):
        raise OpenMeteoClientError("Feature-store event_time must be timezone-aware UTC")
    if frame.duplicated(subset=key_columns).any():
        raise OpenMeteoClientError(
            f"Feature-store frame contains duplicate keys for {key_columns}"
        )
