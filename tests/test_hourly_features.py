"""Day 8 tests for the hourly feature refresh pipeline."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.data.open_meteo import OpenMeteoClientError
from src.feature_store import store as store_module
from src.feature_store.store import insert_features, insert_targets, load_features, load_targets
from src.features.build_features import build_features
from src.features.build_targets import build_targets
from src.pipelines.hourly_features import run_hourly_refresh
from tests.test_feature_store import FakeFeatureStore


@pytest.fixture()
def fake_feature_store(monkeypatch: pytest.MonkeyPatch) -> FakeFeatureStore:
    fs = FakeFeatureStore()
    monkeypatch.setattr(store_module, "_get_feature_store", lambda: fs)
    monkeypatch.setattr(
        "src.feature_store.store.load_city_config",
        lambda: type(
            "City",
            (),
            {
                "city_id": "lahore",
                "name": "Lahore",
                "latitude": 31.5497,
                "longitude": 74.3436,
                "timezone": "Asia/Karachi",
            },
        )(),
    )
    return fs


def _load_raw_frame() -> pd.DataFrame:
    return (
        pd.read_parquet("data/raw/aqi_weather_2022-08-01_2026-08-24.parquet")
        .sort_values("event_time")
        .reset_index(drop=True)
    )


def test_hourly_refresh_appends_new_features_and_backfills_newly_eligible_targets(
    fake_feature_store: FakeFeatureStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _load_raw_frame().iloc[96:360].reset_index(drop=True)
    existing_raw = raw.iloc[:168].reset_index(drop=True)
    refresh_raw = raw.iloc[24:].reset_index(drop=True)

    existing_features = build_features(existing_raw)
    existing_targets = build_targets(existing_raw).iloc[:-12].reset_index(drop=True)
    expected_all_features = build_features(raw)
    expected_all_targets = build_targets(raw)

    insert_features(existing_features)
    insert_targets(existing_targets)

    monkeypatch.setattr("src.pipelines.hourly_features.fetch_air_quality", lambda start, end, city: refresh_raw)
    monkeypatch.setattr(
        "src.pipelines.hourly_features.fetch_weather",
        lambda start, end, city: refresh_raw[["city_id", "latitude", "longitude", "event_time"]],
    )
    monkeypatch.setattr("src.pipelines.hourly_features.merge_air_quality_and_weather", lambda air, weather: air)
    monkeypatch.setattr(
        "src.pipelines.hourly_features.load_city_config",
        lambda: type(
            "City",
            (),
            {
                "city_id": "lahore",
                "name": "Lahore",
                "latitude": 31.5497,
                "longitude": 74.3436,
                "timezone": "Asia/Karachi",
            },
        )(),
    )

    artifacts = run_hourly_refresh(now_utc=refresh_raw["event_time"].max().to_pydatetime())

    stored_features = load_features()
    stored_targets = load_targets()

    assert len(stored_features) == len(expected_all_features)
    assert len(artifacts.inserted_features) == len(expected_all_features)
    assert len(stored_targets) == len(expected_all_targets)
    assert len(artifacts.inserted_targets) == len(expected_all_targets)
    pd.testing.assert_frame_equal(
        stored_targets.sort_values(["city_id", "event_time"]).reset_index(drop=True),
        expected_all_targets.sort_values(["city_id", "event_time"]).reset_index(drop=True),
    )


def test_hourly_refresh_is_idempotent_when_replayed_with_same_source_window(
    fake_feature_store: FakeFeatureStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _load_raw_frame().iloc[96:360].reset_index(drop=True)
    baseline_raw = raw.iloc[:240].reset_index(drop=True)

    insert_features(build_features(baseline_raw))
    insert_targets(build_targets(baseline_raw))

    monkeypatch.setattr("src.pipelines.hourly_features.fetch_air_quality", lambda start, end, city: baseline_raw)
    monkeypatch.setattr(
        "src.pipelines.hourly_features.fetch_weather",
        lambda start, end, city: baseline_raw[["city_id", "latitude", "longitude", "event_time"]],
    )
    monkeypatch.setattr("src.pipelines.hourly_features.merge_air_quality_and_weather", lambda air, weather: air)
    monkeypatch.setattr(
        "src.pipelines.hourly_features.load_city_config",
        lambda: type(
            "City",
            (),
            {
                "city_id": "lahore",
                "name": "Lahore",
                "latitude": 31.5497,
                "longitude": 74.3436,
                "timezone": "Asia/Karachi",
            },
        )(),
    )

    before_features = load_features()
    before_targets = load_targets()
    artifacts = run_hourly_refresh(now_utc=baseline_raw["event_time"].max().to_pydatetime())
    after_features = load_features()
    after_targets = load_targets()

    assert len(artifacts.inserted_features) == len(before_features)
    assert len(artifacts.inserted_targets) == len(before_targets)
    pd.testing.assert_frame_equal(before_features, after_features)
    pd.testing.assert_frame_equal(before_targets, after_targets)


def test_hourly_refresh_requires_existing_backfill(
    fake_feature_store: FakeFeatureStore,
) -> None:
    with pytest.raises(OpenMeteoClientError, match="run the historical backfill"):
        run_hourly_refresh(now_utc=datetime(2026, 8, 29, tzinfo=UTC))
