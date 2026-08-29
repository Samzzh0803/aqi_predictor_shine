# HANDOFF.md

Overwrite the block below at the end of **every** agent session. Keep only the current state plus the last two entries - this file is a baton, not a diary. It exists so the next session doesn't spend twenty minutes rediscovering context.

---

## CURRENT STATE

**Day:** Post-10, automation live and verified. Feature Store, Model Registry, hourly refresh, and daily training/promotion are all genuinely live on real infrastructure. Remaining gap is public hosting only.
**Last updated:** 2026-08-30 by Claude

**What happened this session**
- Reviewed Codex's Day 8 automation diff (`src/pipelines/hourly_features.py`, `run_daily_training_job()` in `train.py`, both workflow YAMLs): hand-verified the `>=72h` target-backfill boundary against `build_targets()`'s actual completeness semantics — correct. Added the one missing test (candidate-beats-incumbent promotion) and restored two README sections a broader rewrite had dropped. Committed and pushed.
- Live dispatch surfaced three real, previously-undetectable bugs — offline tests never caught any of them because their fixtures don't reflect what a live Hopsworks schema or a live Open-Meteo response actually look like:
  1. **`ModuleNotFoundError: No module named 'nest_asyncio'`** — `hsfs` imports it unconditionally but the resolved `hopsworks` build didn't declare it as a dependency (same class of gap as the earlier `confluent-kafka` issue). Fixed: pinned `hopsworks==5.0.6` exactly and added `nest_asyncio` to `requirements.txt`.
  2. **`us_aqi` schema mismatch** (`expected type: 'double', derived from input: 'bigint'`) — a live incremental fetch with no fractional/null AQI values that hour made pandas infer `int64`, disagreeing with the `float64` the historical backfill had locked in. First fix (force `float64` for all Open-Meteo fields in `open_meteo.py`) was too broad.
  3. **The opposite mismatch** (`relative_humidity_2m`/`cloud_cover`/`wind_direction_10m` expected `bigint`, got `double`) — the fix in #2 broke these three, which the original ~4-year backfill happened to lock in as integer types (never a fractional/null value in the whole history). Real fix: `store.py::_conform_to_schema` now reads the feature group's actual live schema (`fg.columns`) and casts every insert to match it, in whichever direction is needed, rather than guessing a single "correct" dtype at the ingestion layer. Verified directly against the live `aqi_features_v1` schema before pushing. Taught the `FakeFeatureGroup` test double to enforce schema compatibility the way real hsfs does (it previously tolerated any dtype mix silently), and added regression tests for both the `open_meteo.py` dtype-stability case and the schema-conform case.
- **Live verification, both workflows, real production writes:**
  - `hourly_features.yml`: `aqi_features_v1` 35,712 → 35,757 rows (+45), `aqi_targets_v1` 35,545 → 35,590 rows (+45), date range extended to 2026-08-29 20:00 UTC, 0 duplicates.
  - `daily_training.yml`: registered **Model Registry version 2** (Ridge, `mae_mean` ≈16.83) — the promote-vs-skip decision (`_should_promote_candidate`) correctly evaluated the fresh candidate against the incumbent (version 1) and promoted it live.

**How to verify**
```bash
pytest tests/ -q          # 76 passed, offline, no credentials needed (fakes only)
ruff check src tests dashboard
```
Live path already proven: both `.github/workflows/hourly_features.yml` and `.github/workflows/daily_training.yml` have a real green dispatch on `main` as of this session.

**Current blockers / follow-up**
1. Not publicly hosted — per `ADR-007`, intended path is Streamlit Community Cloud (dashboard) + Hugging Face Spaces (API). This is now the only remaining gap in the whole project.
2. On a fresh Windows dev machine, `pip install -r requirements.txt` can still hit the `twofish` build failure unless a compiler is available — see `ADR-011` for the local workaround (not needed on GitHub Actions' Ubuntu runners, which build it fine).
3. Repository secrets (`HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`) are now set on GitHub — don't re-add/rotate without updating both.

**Next task**
- Hosting setup (Streamlit Community Cloud + Hugging Face Spaces). No known blockers remain — the Day 8 gate that used to make a hosted dashboard "stale-looking" is now closed for real.

**Gate (Day 8):** MET. Both workflows dispatched live, wrote real data / registered a real model version, verified server-side against Hopsworks directly (not just green CI checkmarks).

---

## PREVIOUS ENTRIES

### 2026-08-29 - Codex (Day 8, hourly/daily automation authored)
- Added `src/pipelines/hourly_features.py` and `run_daily_training_job()`; added both workflow YAMLs and offline tests. Code-complete and offline-tested but not yet live-dispatched at end of this entry — see this session's entry above for the live verification and the three bugs that surfaced.

### 2026-08-29 - Claude (Feature Store swap to live Hopsworks)
- Swapped `src/feature_store/store.py` from the Day 3 local-Parquet fallback to real Hopsworks. Six real issues fixed getting an unmocked run working — see `ADR-011`. Full historical backfill run live (35,712 / 35,545 rows), Day 3 gate re-verified from Hopsworks alone.

### 2026-08-28 - Codex (Day 9, adversarial hardening)
- Hardened `src/pipelines/backfill.py` against all-null critical columns and mid-fetch failures; wrote `docs/operational_decision_support.md`.
