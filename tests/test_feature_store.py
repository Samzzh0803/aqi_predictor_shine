"""Tests for the Hopsworks-backed feature store, using a fake Hopsworks client."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data.open_meteo import OpenMeteoClientError
from src.feature_store import store as store_module
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


def _load_raw_frame() -> pd.DataFrame:
    return (
        pd.read_parquet("data/raw/aqi_weather_2022-08-01_2026-08-24.parquet")
        .sort_values("event_time")
        .reset_index(drop=True)
    )


class FakeFeatureGroup:
    """In-memory stand-in for one Hopsworks FeatureGroup.

    Upserts by primary_key + event_time, matching how Hopsworks complements the
    primary key with the event-time column for offline/Hudi uniqueness when
    time-travel is enabled (the default).
    """

    def __init__(self, name: str, primary_key: list[str], event_time: str) -> None:
        self.name = name
        self._key_columns = [*primary_key, event_time]
        self._frame: pd.DataFrame | None = None

    def insert(self, frame: pd.DataFrame, **kwargs: Any) -> None:
        combined = frame.copy() if self._frame is None else pd.concat([self._frame, frame], ignore_index=True)
        combined = combined.sort_values(self._key_columns).drop_duplicates(subset=self._key_columns, keep="last")
        self._frame = combined.reset_index(drop=True)

    def read(self, **kwargs: Any) -> pd.DataFrame:
        if self._frame is None:
            return pd.DataFrame()
        return self._frame.copy()

    def select_all(self, **kwargs: Any) -> _FakeQuery:
        return _FakeQuery(self)

    def select(self, features: list[str]) -> _FakeQuery:
        return _FakeQuery(self, selected_columns=list(features))


class _FakeQuery:
    def __init__(
        self,
        feature_group: FakeFeatureGroup,
        selected_columns: list[str] | None = None,
        joined_with: _FakeQuery | None = None,
        on: list[str] | None = None,
        join_type: str = "left",
    ) -> None:
        self._feature_group = feature_group
        self._selected_columns = selected_columns
        self._joined_with = joined_with
        self._on = on
        self._join_type = join_type

    def join(self, other: _FakeQuery, on: list[str], join_type: str = "left") -> _FakeQuery:
        return _FakeQuery(self._feature_group, joined_with=other, on=on, join_type=join_type)

    def read(self) -> pd.DataFrame:
        left = self._select(self._feature_group.read())
        if self._joined_with is None:
            return left
        right = self._joined_with.read()
        # A real Hopsworks query always retains the join-key columns even when the
        # right side's `select()` didn't list them (needed to perform the join).
        for key in self._on:
            if key not in right.columns:
                right[key] = self._joined_with._feature_group.read()[key]
        # Real hsfs temporally aligns on each feature group's declared event_time
        # column in addition to whatever primary-key columns are passed as `on`
        # (event_time itself can't be listed there -- it isn't a primary key). Emulate
        # that here, or a same-primary-key join fans out into a many-to-many merge.
        merge_keys = [*self._on, "event_time"]
        if "event_time" not in right.columns:
            right["event_time"] = self._joined_with._feature_group.read()["event_time"]
        return left.merge(right, on=merge_keys, how=self._join_type, validate="one_to_one")

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._selected_columns is None:
            return frame
        return frame[self._selected_columns]


class FakeFeatureView:
    def __init__(self, query: _FakeQuery) -> None:
        self._query = query

    def get_batch_data(self, **kwargs: Any) -> pd.DataFrame:
        return self._query.read()


class FakeFeatureStore:
    """In-memory stand-in for a Hopsworks FeatureStore, scoped to one test."""

    def __init__(self) -> None:
        self._feature_groups: dict[tuple[str, int], FakeFeatureGroup] = {}
        self._feature_views: dict[tuple[str, int], FakeFeatureView] = {}

    def get_or_create_feature_group(
        self,
        name: str,
        version: int,
        primary_key: list[str] | None = None,
        event_time: str | None = None,
        **kwargs: Any,
    ) -> FakeFeatureGroup:
        key = (name, version)
        if key not in self._feature_groups:
            self._feature_groups[key] = FakeFeatureGroup(name, primary_key or [], event_time or "event_time")
        return self._feature_groups[key]

    def get_or_create_feature_view(
        self, name: str, version: int, query: _FakeQuery, **kwargs: Any
    ) -> FakeFeatureView:
        key = (name, version)
        if key not in self._feature_views:
            self._feature_views[key] = FakeFeatureView(query)
        return self._feature_views[key]

    def get_feature_view(self, name: str, version: int) -> FakeFeatureView:
        key = (name, version)
        if key not in self._feature_views:
            raise KeyError(f"No feature view {name} version {version}")
        return self._feature_views[key]


@pytest.fixture()
def fake_feature_store(monkeypatch: pytest.MonkeyPatch) -> FakeFeatureStore:
    fs = FakeFeatureStore()
    monkeypatch.setattr(store_module, "_get_feature_store", lambda: fs)
    return fs


def test_insert_functions_upsert_without_duplicate_keys(fake_feature_store: FakeFeatureStore) -> None:
    raw = _load_raw_frame().iloc[96:240].reset_index(drop=True)
    features = build_features(raw)
    targets = build_targets(raw)

    updated_overlap = features.iloc[40:].copy()
    overlap_event_time = updated_overlap.iloc[0]["event_time"]
    updated_overlap.loc[updated_overlap.index[0], "us_aqi"] = 999.0

    first_insert = insert_features(features.iloc[:80])
    second_insert = insert_features(updated_overlap)
    insert_targets(targets.iloc[:20])
    insert_targets(targets)
    stored_features = load_features()

    assert len(second_insert) == len(features)
    assert not second_insert.duplicated(subset=FEATURES_PRIMARY_KEYS).any()
    assert len(load_targets()) == len(targets)
    assert len(first_insert) == 80
    assert (
        stored_features.loc[stored_features["event_time"] == overlap_event_time, "us_aqi"].iloc[0]
        == 999.0
    )


def test_insert_functions_raise_on_incoming_duplicate_keys(fake_feature_store: FakeFeatureStore) -> None:
    raw = _load_raw_frame().iloc[96:240].reset_index(drop=True)
    features = build_features(raw).iloc[:10].copy()
    duplicate_features = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    targets = build_targets(raw).iloc[:10].copy()
    duplicate_targets = pd.concat([targets, targets.iloc[[0]]], ignore_index=True)

    with pytest.raises(OpenMeteoClientError, match="duplicate keys"):
        insert_features(duplicate_features)

    with pytest.raises(OpenMeteoClientError, match="duplicate keys"):
        insert_targets(duplicate_targets)


def test_verify_and_feature_view_match_expected_training_rows(fake_feature_store: FakeFeatureStore) -> None:
    raw = _load_raw_frame().head(240)
    features = build_features(raw)
    targets = build_targets(raw)

    insert_features(features)
    insert_targets(targets)

    stored_features = load_features()
    stored_targets = load_targets()
    feature_view = create_feature_view(features=stored_features, targets=stored_targets)
    reloaded_feature_view = load_feature_view()

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


def test_load_feature_view_raises_when_not_created_yet(fake_feature_store: FakeFeatureStore) -> None:
    with pytest.raises(OpenMeteoClientError, match="does not exist yet"):
        load_feature_view()
