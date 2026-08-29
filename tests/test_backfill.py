"""Day 3 tests for the backfill pipeline against the Hopsworks feature store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.open_meteo import OpenMeteoClientError
from src.feature_store import store as store_module
from src.feature_store.store import load_feature_view, load_features, load_targets
from src.features.build_features import build_features
from src.features.build_targets import build_targets
from src.pipelines.backfill import run_backfill
from tests.test_feature_store import FakeFeatureStore


@pytest.fixture()
def fake_feature_store(monkeypatch: pytest.MonkeyPatch) -> FakeFeatureStore:
    fs = FakeFeatureStore()
    monkeypatch.setattr(store_module, "_get_feature_store", lambda: fs)
    return fs


def _load_raw_frame() -> pd.DataFrame:
    return (
        pd.read_parquet("data/raw/aqi_weather_2022-08-01_2026-08-24.parquet")
        .sort_values("event_time")
        .reset_index(drop=True)
    )


def test_run_backfill_rebuilds_training_data_from_local_store(
    tmp_path: Path,
    fake_feature_store: FakeFeatureStore,
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
    fake_feature_store: FakeFeatureStore,
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
    fake_feature_store: FakeFeatureStore,
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

    rebuilt_training_set = load_feature_view()
    expected_training_set = pd.merge(
        load_features(),
        load_targets(),
        on=["city_id", "event_time"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["city_id", "event_time"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(rebuilt_training_set, expected_training_set)


def test_run_backfill_rejects_all_null_weather_column(
    fake_feature_store: FakeFeatureStore,
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
    fake_feature_store: FakeFeatureStore,
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
