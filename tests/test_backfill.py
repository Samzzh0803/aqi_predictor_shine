"""Day 3 tests for local feature-store fallback and backfill pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.open_meteo import OpenMeteoClientError
from src.feature_store import store
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
from src.pipelines.backfill import run_backfill


def _load_raw_frame() -> pd.DataFrame:
    return (
        pd.read_parquet("data/raw/aqi_weather_2022-08-01_2026-08-24.parquet")
        .sort_values("event_time")
        .reset_index(drop=True)
    )


@pytest.fixture()
def local_store_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    feature_store_root = tmp_path / "feature_store"
    features_path = feature_store_root / "aqi_features_v1.parquet"
    targets_path = feature_store_root / "aqi_targets_v1.parquet"
    feature_view_path = feature_store_root / "aqi_fv_v1.parquet"

    monkeypatch.setattr(store, "_FEATURE_STORE_ROOT", feature_store_root)
    monkeypatch.setattr(store, "_FEATURES_PATH", features_path)
    monkeypatch.setattr(store, "_TARGETS_PATH", targets_path)
    monkeypatch.setattr(store, "_FEATURE_VIEW_PATH", feature_view_path)

    return {
        "features": features_path,
        "targets": targets_path,
        "feature_view": feature_view_path,
    }


def test_insert_functions_upsert_without_duplicate_keys(local_store_paths: dict[str, Path]) -> None:
    raw = _load_raw_frame().iloc[96:240].reset_index(drop=True)
    features = build_features(raw)
    targets = build_targets(raw)

    updated_overlap = features.iloc[40:].copy()
    overlap_event_time = updated_overlap.iloc[0]["event_time"]
    updated_overlap.loc[updated_overlap.index[0], "us_aqi"] = 999.0

    first_insert = insert_features(features.iloc[:80], path=local_store_paths["features"])
    second_insert = insert_features(updated_overlap, path=local_store_paths["features"])
    insert_targets(targets.iloc[:20], path=local_store_paths["targets"])
    insert_targets(targets, path=local_store_paths["targets"])
    stored_features = load_features(path=local_store_paths["features"])

    assert len(second_insert) == len(features)
    assert not second_insert.duplicated(subset=FEATURES_PRIMARY_KEYS).any()
    assert len(load_targets(path=local_store_paths["targets"])) == len(targets)
    assert len(first_insert) == 80
    assert (
        stored_features.loc[stored_features["event_time"] == overlap_event_time, "us_aqi"].iloc[0]
        == 999.0
    )


def test_insert_functions_raise_on_incoming_duplicate_keys(
    local_store_paths: dict[str, Path],
) -> None:
    raw = _load_raw_frame().iloc[96:240].reset_index(drop=True)
    features = build_features(raw).iloc[:10].copy()
    duplicate_features = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    targets = build_targets(raw).iloc[:10].copy()
    duplicate_targets = pd.concat([targets, targets.iloc[[0]]], ignore_index=True)

    with pytest.raises(OpenMeteoClientError, match="duplicate keys"):
        insert_features(duplicate_features, path=local_store_paths["features"])

    with pytest.raises(OpenMeteoClientError, match="duplicate keys"):
        insert_targets(duplicate_targets, path=local_store_paths["targets"])


def test_verify_and_feature_view_match_expected_training_rows(
    local_store_paths: dict[str, Path],
) -> None:
    raw = _load_raw_frame().head(240)
    features = build_features(raw)
    targets = build_targets(raw)

    insert_features(features, path=local_store_paths["features"])
    insert_targets(targets, path=local_store_paths["targets"])

    stored_features = load_features(path=local_store_paths["features"])
    stored_targets = load_targets(path=local_store_paths["targets"])
    feature_view = create_feature_view(
        features=stored_features,
        targets=stored_targets,
        path=local_store_paths["feature_view"],
    )
    reloaded_feature_view = load_feature_view(path=local_store_paths["feature_view"])

    features_summary = verify_feature_group(stored_features, FEATURES_PRIMARY_KEYS)
    targets_summary = verify_feature_group(stored_targets, TARGETS_PRIMARY_KEYS)

    assert features_summary["row_count"] == len(features)
    assert targets_summary["row_count"] == len(targets)
    assert features_summary["duplicate_count"] == 0
    assert targets_summary["duplicate_count"] == 0
    assert len(feature_view) == len(targets)
    pd.testing.assert_frame_equal(feature_view, reloaded_feature_view)


def test_verify_feature_group_reports_duplicates_without_crashing() -> None:
    raw = _load_raw_frame().iloc[96:110].reset_index(drop=True)
    features = build_features(raw).iloc[:5].copy()
    duplicate_frame = pd.concat([features, features.iloc[[0]]], ignore_index=True)

    summary = verify_feature_group(duplicate_frame, FEATURES_PRIMARY_KEYS)

    assert summary["duplicate_count"] == 1


def test_run_backfill_rebuilds_training_data_from_local_store(
    tmp_path: Path,
    local_store_paths: dict[str, Path],
) -> None:
    raw = _load_raw_frame().head(240)
    raw_cache_path = tmp_path / "raw_cache.parquet"

    artifacts = run_backfill(
        start_date=raw["event_time"].min().date().isoformat(),
        end_date=raw["event_time"].max().date().isoformat(),
        source_frame=raw,
        raw_cache_path=raw_cache_path,
    )

    assert len(artifacts.features) == len(build_features(raw))
    assert len(artifacts.targets) == len(build_targets(raw))
    assert len(artifacts.feature_view) == len(artifacts.targets)
    assert artifacts.features_summary["date_min"] == artifacts.raw_frame["event_time"].min()
    assert artifacts.targets_summary["date_max"] == artifacts.targets["event_time"].max()


def test_run_backfill_respects_requested_date_range_when_cache_exists(
    tmp_path: Path,
    local_store_paths: dict[str, Path],
) -> None:
    raw = _load_raw_frame().iloc[96:400].reset_index(drop=True)
    raw_cache_path = tmp_path / "raw_cache.parquet"
    raw.to_parquet(raw_cache_path, index=False)

    start_date = raw["event_time"].iloc[24].date().isoformat()
    end_date = raw["event_time"].iloc[120].date().isoformat()

    artifacts = run_backfill(
        start_date=start_date,
        end_date=end_date,
        raw_cache_path=raw_cache_path,
    )

    assert artifacts.raw_frame["event_time"].min() >= pd.Timestamp(start_date, tz="UTC")
    assert artifacts.raw_frame["event_time"].max() < (
        pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    )


def test_day3_gate_rebuilds_training_set_from_feature_store_parquet_alone(
    tmp_path: Path,
    local_store_paths: dict[str, Path],
) -> None:
    raw = _load_raw_frame().iloc[96:400].reset_index(drop=True)
    raw_cache_path = tmp_path / "raw_cache.parquet"
    raw.to_parquet(raw_cache_path, index=False)

    run_backfill(
        start_date=raw["event_time"].min().date().isoformat(),
        end_date=raw["event_time"].max().date().isoformat(),
        raw_cache_path=raw_cache_path,
    )

    raw_cache_path.unlink()

    rebuilt_training_set = load_feature_view(path=local_store_paths["feature_view"])
    expected_training_set = create_feature_view(
        features=load_features(path=local_store_paths["features"]),
        targets=load_targets(path=local_store_paths["targets"]),
        path=local_store_paths["feature_view"],
    )

    pd.testing.assert_frame_equal(rebuilt_training_set, expected_training_set)


def test_run_backfill_rejects_all_null_weather_column(
    local_store_paths: dict[str, Path],
) -> None:
    raw = _load_raw_frame().iloc[96:240].reset_index(drop=True)
    raw["wind_speed_10m"] = pd.NA

    with pytest.raises(OpenMeteoClientError, match="all-null columns: wind_speed_10m"):
        run_backfill(
            start_date=raw["event_time"].min().date().isoformat(),
            end_date=raw["event_time"].max().date().isoformat(),
            source_frame=raw,
        )


def test_run_backfill_rejects_all_null_aqi_column(
    local_store_paths: dict[str, Path],
) -> None:
    raw = _load_raw_frame().iloc[96:240].reset_index(drop=True)
    raw["us_aqi"] = pd.NA

    with pytest.raises(OpenMeteoClientError, match="all-null columns: us_aqi"):
        run_backfill(
            start_date=raw["event_time"].min().date().isoformat(),
            end_date=raw["event_time"].max().date().isoformat(),
            source_frame=raw,
        )


def test_run_backfill_wraps_air_quality_fetch_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.pipelines.backfill.fetch_air_quality",
        lambda start, end, city: (_ for _ in ()).throw(OpenMeteoClientError("timeout during AQI fetch")),
    )
    monkeypatch.setattr(
        "src.pipelines.backfill.fetch_weather",
        lambda start, end, city: pd.DataFrame(),
    )

    with pytest.raises(OpenMeteoClientError, match="Backfill failed while fetching air quality"):
        run_backfill("2026-08-20", "2026-08-21")


def test_run_backfill_wraps_weather_fetch_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.pipelines.backfill.fetch_air_quality",
        lambda start, end, city: pd.DataFrame({"city_id": [], "latitude": [], "longitude": [], "event_time": []}),
    )
    monkeypatch.setattr(
        "src.pipelines.backfill.fetch_weather",
        lambda start, end, city: (_ for _ in ()).throw(OpenMeteoClientError("timeout during weather fetch")),
    )

    with pytest.raises(OpenMeteoClientError, match="Backfill failed while fetching weather"):
        run_backfill("2026-08-20", "2026-08-21")
