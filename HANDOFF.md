# HANDOFF.md

Overwrite the block below at the end of **every** agent session. Keep only the current state plus the last two entries - this file is a baton, not a diary. It exists so the next session doesn't spend twenty minutes rediscovering context.

---

## CURRENT STATE

**Day:** 6 of 10 - complete. Local-first inference and API path implemented and verified against real, fresh data end to end.
**Last updated:** 2026-08-27 by Claude

**Test status:**
- `pytest tests/test_predictor.py tests/test_api.py -q` -> `11 passed, 3 warnings in 14.47s`
- `python -m ruff check src/inference src/api tests/test_predictor.py tests/test_api.py src/models/registry.py` -> `All checks passed!`

**Completed this session**
- Added `src/inference/predictor.py` with `predict_next_3_days()` using the local fallback registry and local feature store.
- Added shared AQI helpers in `src/inference/aqi.py` as the single source of truth for category and alert thresholds.
- Enforced Day 6 inference constraints:
  - load champion from registry
  - reconstruct exact registered feature order
  - reject missing or null features
  - clip predictions to `[0, 500]`
  - attach AQI category and alert labels
  - reject stale features older than 48 hours with a clear error
- Added `src/api/main.py` FastAPI endpoints for `/health`, `/forecast`, `/model-info`, and `/history`.
- Extended `src/models/registry.py` so inference can load serialized local model artifacts, including TensorFlow bundles if a future champion is the MLP.
- Installed missing FastAPI runtime dependencies in the active Python environment so Day 6 API tests run for real.
- Verified real local smoke behavior on Thursday, August 27, 2026:
  - `/health` -> `200` with `{"status": "ok"}`
  - `/forecast` -> `503` with `Latest features are stale: event_time=2026-08-24T23:00:00+00:00 is older than 48 hours`

**Files changed this session**
- `src/inference/aqi.py`
- `src/inference/__init__.py`
- `src/inference/predictor.py`
- `src/api/__init__.py`
- `src/api/main.py`
- `src/models/registry.py`
- `tests/test_predictor.py`
- `tests/test_api.py`
- `HANDOFF.md`

**How to verify**
```bash
pytest tests/test_predictor.py tests/test_api.py -q
python -m ruff check src/inference src/api tests/test_predictor.py tests/test_api.py src/models/registry.py
python -c "from fastapi.testclient import TestClient; from src.api.main import app; client = TestClient(app); print(client.get('/health').status_code); print(client.get('/forecast').status_code); print(client.get('/forecast').json())"
```

**Current blockers / follow-up**
1. Hopsworks is still not required for Day 6/7 local development; local registry + local feature store are enough.
2. Refreshed `data/raw/*.parquet` and the local feature store with real data through `2026-08-27T23:00:00Z` (fetched via `fetch_air_quality_recent`/`fetch_weather_recent`, merged into the existing raw cache, re-ran `backfill.py`). `/forecast` now returns `200` with real predictions on a fresh, unmocked process - verified directly, not just via mocked tests. No hourly automation exists yet (that's Day 8); this was a one-off manual refresh, and features will go stale again after ~48h without it.
3. `targets` max event_time is legitimately behind `features` max event_time (`2026-08-24` vs `2026-08-27`) - this is correct, not a bug: the most recent ~72h of rows can't have targets yet since day3 needs t+49..t+72h of future data that doesn't exist yet.

**Next task**
- Day 7: Streamlit dashboard. Reuse the same "fail visibly, never silently" pattern the API already has for the stale-features case.

**Gate (Day 6):** MET. `/health` -> 200. `/forecast` -> 200 with three real, correctly-labeled predictions on fresh unmocked data (verified). API fails cleanly (503, no stack trace) when features are stale or the registry is empty.

---

## PREVIOUS ENTRIES

### 2026-08-27 - Claude
- Found and fixed a real Day 2 contract violation: `build_features.py` was still emitting `aqi_lag_1h`, `pm25_lag_1h`, and `aqi_change_1h`, which `PROJECT_CONTRACT.md` section 4 explicitly excludes.
- Rebuilt downstream local artifacts and re-verified that `ridge` remained the Day 5 champion on the corrected 46-feature set.

### 2026-08-26 - Codex
- Implemented `src/pipelines/train.py` for Day 4 with chronological splits, baselines, Ridge, RandomForest, HistGradientBoosting, and comparison table persistence.
- Implemented Day 3 local Parquet feature-store fallback in `src/feature_store/store.py` and `src/pipelines/backfill.py`.
