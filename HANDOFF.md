# HANDOFF.md

Overwrite the block below at the end of **every** agent session. Keep only the current state plus the last two entries - this file is a baton, not a diary. It exists so the next session doesn't spend twenty minutes rediscovering context.

---

## CURRENT STATE

**Day:** Post-10, feature-store swap. Both Hopsworks-backed pieces (Model Registry, Feature Store) are now genuinely live. Remaining gaps are automation and public hosting only.
**Last updated:** 2026-08-29 by Claude

**What happened this session**
- Swapped `src/feature_store/store.py` from the Day 3 local-Parquet fallback (`ADR-008`) to the real Hopsworks Feature Store, mirroring `ADR-010`'s pattern exactly: same public function signatures (minus the local-only `path=` test parameter), lazy/cached `_get_feature_store()` login. Zero changes needed to `backfill.py`, `train.py`, `predictor.py`, `api/main.py`.
- Found and fixed six real, non-cosmetic issues getting a genuine unmocked pipeline run working (full detail in `ADR-011`): `DELTA` time-travel format needs an uninstalled package (switched to `HUDI`); free-tier per-insert statistics computation is flaky (HTTP 500, disabled); `Query.join()` on `event_time` fails outright since it isn't a primary key, and the resulting Feature View read path turned out to silently reuse stale target values for rows that shouldn't have known targets yet (a real leakage risk) — worked around by building the actual training frame from a verified local `(city_id, event_time)` inner merge rather than trusting the Feature View's own batch-read; ambiguous-column auto-prefixing on overlapping join columns; a too-short default Kafka metadata timeout for this network path (broker reachable, cert valid, just slow — raised to 60s).
- Also needed, to get `hopsworks` installed and working on this Windows dev machine at all (separate from the product code, recorded in `ADR-011` for the next session's benefit): a local no-op stub for `twofish` (a hard transitive dependency of `pyjks` with no prebuilt Windows wheel, needed for a legacy Java-keystore decryption path this project never exercises), `confluent-kafka` installed explicitly (now added to `requirements.txt`), and a VC++ 2022 redistributable install to fix an unrelated pre-existing TensorFlow DLL load failure that was blocking 5 of 9 test files from even collecting.
- New `tests/test_feature_store.py`: a `FakeFeatureStore`/`FakeFeatureGroup` in-memory double (mirrors `test_registry.py`'s `FakeModelRegistry`), reused by `tests/test_backfill.py` and `tests/test_train.py` in place of the old local-parquet-path fixtures. Offline suite needs no real `hopsworks` package installed (same lazy-import pattern as `registry.py`).
- Ran the real Day 3 gate against live Hopsworks: full historical backfill (2022-08-01 → today), then a fresh process rebuilding the training set from the Feature Store alone with no local cache involved. Results: `aqi_features_v1` 35,712 rows (2022-08-01 00:00 → 2026-08-27 23:00 UTC, 0 duplicates) — matches the local raw cache's row count exactly; `aqi_targets_v1` 35,545 rows (0 duplicates); the joined training set is 35,545 rows (53 columns), 0 duplicate keys, 0 nulls in any of the three target columns.
- Updated `README.md` (feature store no longer described as local-only) and `requirements.txt` (`confluent-kafka` added).

**How to verify**
```bash
pytest tests/ -q
ruff check src tests dashboard
```
Both offline, no credentials needed (fakes only). For the live path, `.env` already has working `HOPSWORKS_API_KEY`/`HOPSWORKS_PROJECT` (same ones the Model Registry uses) — `python -m src.pipelines.backfill` populates the real Feature Store.

**Current blockers / follow-up**
1. No hourly/daily automation yet (`hourly_features.yml`, `daily_training.yml` per `ARCHITECTURE.md` don't exist) — this is now genuinely unblocked (the Feature Store is live), just not built.
2. Not publicly hosted — per `ADR-007`, intended path is Streamlit Community Cloud (dashboard) + Hugging Face Spaces (API), both free/no card. Also now unblocked.
3. On a fresh Windows dev machine, `pip install -r requirements.txt` will hit the `twofish` build failure again unless a compiler is available — see `ADR-011` for the stub workaround (not committed to the repo, since it's a local, machine-specific fix, not a real package).

**Next task**
- Automation (`hourly_features.yml` + `daily_training.yml`) or hosting setup — both are now genuinely unblocked. Ask which to prioritize; they don't strictly depend on each other, though automation makes a hosted dashboard meaningfully fresher.

**Gate:** MET. Day 3 gate re-verified against the live Feature Store (fresh process, no local cache): 35,712 feature rows / 35,545 target rows / 35,545-row training set, 0 duplicates, 0 target nulls.

---

## PREVIOUS ENTRIES

### 2026-08-29 - Claude (Day 10, report/README/demo)
- Wrote `README.md`, `docs/REPORT.md`, `docs/DEMO.md`; added `ADR-010` recording the Model Registry swap. Full suite 67 passed. See prior git history for detail — superseded by this session's entry above.

### 2026-08-28 - Claude (Day 9, Hopsworks live)
- Diagnosed and fixed four real Hopsworks SDK/scope issues (Windows cert path, a genuine SDK bug in free-tier error handling, two missing API-key scopes, a `description` length limit) to get the Model Registry genuinely working - verified with a real, unmocked pipeline run and a real registered champion.
- Found and fixed a critical `.gitignore` bug (`models/` matching `src/models/`) that had silently excluded the entire registry module from every GitHub push since Day 8, and a missing `tests/__init__.py` that caused CI to fail on Ubuntu (found via the actual GitHub Actions API logs, not guessing).

### 2026-08-28 - Codex (Day 9, adversarial hardening)
- Hardened `src/pipelines/backfill.py` against all-null critical columns and mid-fetch failures; wrote `docs/operational_decision_support.md`.
