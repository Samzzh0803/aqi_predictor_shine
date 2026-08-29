"""Hopsworks-backed Feature Store for the AQI forecaster."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

import pandas as pd
from pandas import DatetimeTZDtype

from src.data.open_meteo import OpenMeteoClientError
from src.features.build_targets import TARGET_COLUMNS

FEATURES_PRIMARY_KEYS: Final[list[str]] = ["city_id", "event_time"]
TARGETS_PRIMARY_KEYS: Final[list[str]] = ["city_id", "event_time"]

FEATURES_FG_NAME: Final[str] = "aqi_features_v1"
TARGETS_FG_NAME: Final[str] = "aqi_targets_v1"
FEATURE_VIEW_NAME: Final[str] = "aqi_fv_v1"
FEATURE_GROUP_VERSION: Final[int] = 1
FEATURE_VIEW_VERSION: Final[int] = 1
CERT_FOLDER = str(Path("data") / ".hopsworks_certs")

_feature_store_cache: Any = None


def insert_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Upsert feature rows into the `aqi_features_v1` feature group."""

    return _insert(
        frame,
        key_columns=FEATURES_PRIMARY_KEYS,
        fg_name=FEATURES_FG_NAME,
        description="AQI predictor engineered feature rows.",
    )


def insert_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Upsert target rows into the `aqi_targets_v1` feature group."""

    return _insert(
        frame,
        key_columns=TARGETS_PRIMARY_KEYS,
        fg_name=TARGETS_FG_NAME,
        description="AQI predictor forward-looking target rows.",
    )


def load_features() -> pd.DataFrame:
    """Load the `aqi_features_v1` feature group."""

    return _load(FEATURES_FG_NAME, FEATURES_PRIMARY_KEYS)


def load_targets() -> pd.DataFrame:
    """Load the `aqi_targets_v1` feature group."""

    return _load(TARGETS_FG_NAME, TARGETS_PRIMARY_KEYS)


def create_feature_view(
    features: pd.DataFrame | None = None,
    targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create/reuse the `aqi_fv_v1` Feature View joining both groups, and materialize it.

    `features`/`targets` are accepted for call-site compatibility with the Day 3 local
    fallback's signature (`backfill.py` passes them positionally) but are not used: a
    Hopsworks Feature View is always built from the feature groups themselves, not from
    arbitrary in-memory frames.
    """

    del features, targets
    fs = _get_feature_store()
    features_fg = _get_feature_group(fs, FEATURES_FG_NAME)
    targets_fg = _get_feature_group(fs, TARGETS_FG_NAME)
    # `on` must name actual primary-key columns on both sides; event_time is not a
    # primary key, so it isn't listed here.
    # Select only the target columns from the right side: city_id/event_time are
    # already on the left, and selecting them again triggers hsfs's ambiguous-column
    # auto-prefixing (e.g. "aqi_targets_v1_target_aqi_day1").
    query = features_fg.select_all().join(targets_fg.select(TARGET_COLUMNS), on=["city_id"])
    fs.get_or_create_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION, query=query)
    return _build_training_frame()


def load_feature_view() -> pd.DataFrame:
    """Load the existing `aqi_fv_v1` Feature View, or raise if it hasn't been created yet."""

    fs = _get_feature_store()
    try:
        fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    except Exception as exc:  # noqa: BLE001 - hsfs raises its own exception types for "not found"
        raise OpenMeteoClientError(f"Feature View {FEATURE_VIEW_NAME} does not exist yet") from exc
    return _build_training_frame()


def _build_training_frame() -> pd.DataFrame:
    """Join features and targets directly rather than trusting the Feature View's own read path.

    hsfs Feature Views default to a point-in-time-correct join (matching each row to the
    most recent available value at or before its event_time), which silently reuses stale
    target values for the trailing rows build_targets() already dropped as incomplete --
    a real leakage risk. An exact (city_id, event_time) inner merge of the feature groups'
    own verified frames is the correct, testable semantics for this project's exact-grid
    feature/target rows.
    """

    frame = load_features().merge(
        load_targets(), on=["city_id", "event_time"], how="inner", validate="one_to_one"
    )
    return frame.sort_values(["city_id", "event_time"]).reset_index(drop=True)


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


# hsfs column type -> pandas dtype, for conforming an incoming frame to whatever a
# feature group's schema already locked in (see _conform_to_schema).
_HSFS_TYPE_TO_PANDAS: Final[dict[str, str]] = {
    "bigint": "int64",
    "int": "int32",
    "double": "float64",
    "float": "float32",
    "boolean": "bool",
    "string": "object",
}


def _insert(frame: pd.DataFrame, *, key_columns: list[str], fg_name: str, description: str) -> pd.DataFrame:
    _validate_frame(frame=frame, key_columns=key_columns)
    fs = _get_feature_store()
    fg = _get_feature_group(fs, fg_name, description=description)
    frame = _conform_to_schema(frame, fg)
    # The Python engine buffers inserts through Kafka before materializing to Hudi.
    # hsfs's default kafka_timeout (6s) for the metadata round-trip is too short for
    # this free-tier cluster's network path even though the broker is reachable and
    # SSL handshake succeeds -- it just takes longer than 6s. TASKS.md's own "the free
    # tier is not fast" applies here too.
    fg.insert(frame, wait=True, write_options={"kafka_timeout": 60})
    return _load(fg_name, key_columns)


def _conform_to_schema(frame: pd.DataFrame, fg: Any) -> pd.DataFrame:
    """Cast columns to match the feature group's already-registered schema.

    A feature group's column types are fixed by whatever pandas happened to infer on
    its first insert -- e.g. a whole-history backfill where relative_humidity_2m never
    once had a fractional or null reading infers as 'bigint', while us_aqi (which did,
    somewhere in ~4 years of data) infers as 'double'. A later incremental insert's own
    pandas inference depends on that batch's actual values and can easily disagree
    (a live fetch of exactly-integer AQI readings would infer 'bigint' where the
    registered schema expects 'double', or vice versa for a fractional humidity
    reading against a 'bigint' schema). Conform to the registered schema rather than
    guessing at the ingestion layer -- for a brand-new feature group (no schema
    registered yet), fg.columns is empty and this is a no-op; the first insert's own
    dtypes define the schema, same as before.
    """

    frame = frame.copy()
    for column in fg.columns:
        target_dtype = _HSFS_TYPE_TO_PANDAS.get(column.type.lower())
        if target_dtype is None or column.name not in frame.columns:
            continue
        frame[column.name] = frame[column.name].astype(target_dtype)
    return frame


def _load(fg_name: str, key_columns: list[str]) -> pd.DataFrame:
    fs = _get_feature_store()
    fg = _get_feature_group(fs, fg_name)
    frame = fg.read()
    if frame.empty:
        return frame
    frame = frame.sort_values(key_columns).reset_index(drop=True)
    _validate_frame(frame=frame, key_columns=key_columns)
    return frame


def _get_feature_group(fs: Any, name: str, *, description: str = "") -> Any:
    return fs.get_or_create_feature_group(
        name=name,
        version=FEATURE_GROUP_VERSION,
        primary_key=["city_id"],
        event_time="event_time",
        online_enabled=False,
        # DELTA (the hsfs default) needs the optional `delta`/`deltalake` package;
        # HUDI needs no extra dependency and is what ARCHITECTURE.md assumes for
        # offline primary_key + event_time uniqueness.
        time_travel_format="HUDI",
        # The free-tier statistics-computation call after each Hudi write is flaky
        # (observed: HTTP 500 "Transaction marked for rollback" on the metadata
        # service) and we don't use per-feature-group statistics, so skip it.
        statistics_config=False,
        description=description,
    )


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


def _get_feature_store() -> Any:
    """Log in to Hopsworks (once per process) and return the project's feature store."""

    global _feature_store_cache
    if _feature_store_cache is not None:
        return _feature_store_cache

    from dotenv import load_dotenv

    load_dotenv()

    import hopsworks
    import hopsworks_common.connection as hw_connection

    def _patched_provide_project(self: Any, name: str | None = None) -> None:
        from hopsworks_common import client

        _client = client._get_instance()
        if name:
            self._project = name
            if _client._is_external():
                _client._provide_project(name)
        if _client._project_name:
            self._project = _client._project_name
        if not self._project:
            return

        from hsfs import engine

        engine._get_instance()
        if self._variable_api._get_data_science_profile_enabled():
            try:
                self._model_serving_api._load_default_configuration()
            except Exception:  # noqa: BLE001 - free-tier keys lack SERVING scope; this is expected
                pass

    hw_connection.Connection._provide_project = _patched_provide_project

    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT")
    if not api_key or not project_name:
        raise OpenMeteoClientError(
            "HOPSWORKS_API_KEY and HOPSWORKS_PROJECT must be set to use the feature store"
        )

    project = hopsworks.login(
        api_key_value=api_key,
        project=project_name,
        cert_folder=CERT_FOLDER,
    )
    _feature_store_cache = project.get_feature_store()
    return _feature_store_cache
