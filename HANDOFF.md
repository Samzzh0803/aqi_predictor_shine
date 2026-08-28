# HANDOFF.md

Overwrite the block below at the end of **every** agent session. Keep only the current state plus the last two entries - this file is a baton, not a diary. It exists so the next session doesn't spend twenty minutes rediscovering context.

---

## CURRENT STATE

**Day:** 9 of 10 - Hopsworks is live. Model Registry swapped in for real; feature store is still the local fallback.
**Last updated:** 2026-08-28 by Claude

**What happened this session (the big one)**
- Hopsworks is no longer blocked. The "clustering problem" from earlier days was never real - it was two client-side bugs plus two API-key scope gaps, all fixed:
  1. `hopsworks.login()` defaults to a Unix `/tmp/...` cert path that doesn't exist on Windows - fixed with `cert_folder="data/.hopsworks_certs"`.
  2. The installed `hopsworks` 5.0.6 package has a real bug in its own free-tier SERVING-scope error handling (`e.response.error_code` should be `e.response.status_code`) - patched around it in `_get_model_registry()`.
  3. The API key needed `DATASET_CREATE` (to create the "Models" dataset Hopsworks' Model Registry requires) and `MODELREGISTRY` scopes - human regenerated the key with broader scopes.
  4. `mr.python.create_model(...).save()` failed because Hopsworks' `description` column has a length limit our full JSON metadata (46 features + 12 artifact filenames) exceeded - fixed by moving that metadata into a small `extra_manifest.json` file uploaded alongside the model instead of cramming it into `description`.
- **Found and fixed a critical, previously-invisible bug**: `.gitignore` had a bare `models/` pattern meant to exclude a top-level model-artifacts folder, but it matched *any* directory named "models" anywhere in the tree - including `src/models/`. The entire model registry module had never been pushed to GitHub since the repo was connected on Day 8, despite being imported by `train.py`, `predictor.py`, and `api/main.py`. This is almost certainly why every CI run failed. Fixed the pattern (`/models/`, anchored to root) and pushed the missing module.
- Also caught a real API key sitting in `.env.example` (the committed template, not `.env`) - checked git history, confirmed it was never actually committed or pushed, reverted locally. No rotation needed.
- **Swapped `src/models/registry.py` from the local Parquet fallback to real Hopsworks Model Registry** (`register_model_version`, `get_champion`, `list_registered_versions`, `load_registered_models` - same signatures, zero changes needed in `train.py`/`predictor.py`/`api/main.py`). Feature store (`src/feature_store/store.py`) is intentionally **not** swapped yet - human chose the "registry only" minimal swap to keep Day 9/10 timeline safe.
- Rewrote `tests/test_registry.py` and the champion-selection tests in `tests/test_train_day5.py` to use an in-memory fake Hopsworks client (`FakeModelRegistry`) instead of live credentials, so the test suite runs offline and in CI without needing `HOPSWORKS_API_KEY`.
- **Verified for real, not just mocks**: ran the full unmocked Day 5 pipeline against live Hopsworks. Champion is registered at `https://eu-west.cloud.hopsworks.ai:443/p/41216/models/pearls_aqi_forecaster/1` (ridge, mae_mean 16.83, selection_mae_mean 18.07, 46 features, all 3 model files + 9 SHAP artifacts attached). Confirmed `get_champion()`, `load_registered_models()`, and `predict_next_3_days()` all work correctly reading it back from Hopsworks.
- Full suite: 67 passed. `ruff check src tests dashboard`: clean.

**Files changed this session**
- `.gitignore` (critical fix: `models/` -> `/models/`)
- `src/models/registry.py` (rewritten for Hopsworks; now actually tracked in git for the first time)
- `src/models/__init__.py` (now actually tracked in git for the first time)
- `tests/test_registry.py` (rewritten with `FakeModelRegistry`)
- `tests/test_train_day5.py` (champion-selection tests use the fake registry)
- `tests/test_predictor.py` (one test now mocks `load_registered_models` directly - it tests predictor logic, not the registry backend)
- `.env.example` (reverted an accidentally-real key back to placeholder; never actually pushed)

**How to verify**
```bash
pytest tests/ -q
ruff check src tests dashboard
python -c "from src.inference.predictor import predict_next_3_days; r = predict_next_3_days(); print(r.model_version, r.model_type); [print(p) for p in r.forecast]"
```
Also: check the GitHub Actions tab - a new CI run should be green now that `src/models/` actually exists in the repo.

**Current blockers / follow-up**
1. Feature store is still local-only (`data/feature_store/`, gitignored, this machine only). Swapping it to real Hopsworks Feature Store was explicitly deferred - only the registry was in scope this session.
2. `HOPSWORKS_API_KEY`/`HOPSWORKS_PROJECT` need to be set (real values, not placeholders) wherever this code runs next - they're in the local `.env` now, but **not** set as GitHub Actions secrets yet, so any CI/automation step that touches the registry (none currently do - tests use the fake) would need that added first.
3. `_get_model_registry()`'s login-workaround (the two SDK bug patches) lives inline in `registry.py`. If `hopsworks` gets upgraded and fixes these bugs upstream, the patches are harmless no-ops, but worth knowing they're there.

**Next task**
- Confirm the new CI run is green on GitHub (this was the actual fix for the failures, not a network/Ubuntu issue as first suspected).
- Day 10 prep: report + demo. The report can now honestly show a *real* registered Hopsworks Model Registry entry with a live URL, not just local artifacts.

**Gate (Day 9):** MET. Adversarial hardening, decision-support doc, and a genuinely live Hopsworks Model Registry integration are all done and verified - the last one by running the real, unmocked pipeline against the real service, not by trusting a report.

---

## PREVIOUS ENTRIES

### 2026-08-28 - Codex (Day 9, earlier this session)
- Hardened `src/pipelines/backfill.py` with `validate_backfill_source_frame()` (rejects all-null critical columns) and stage-specific fetch-failure wrapping.
- Wrote `docs/operational_decision_support.md` (DATA -> FORECAST -> RISK -> ACTION framing, 6 categories, worked example) - the Day 9 consulting-layer deliverable.

### 2026-08-27 - Codex + Claude (Day 7)
- Built the Streamlit dashboard. Claude caught a missing Row 3 (current pollutant/weather values) and a suppressed loading state; both fixed and verified live against a running FastAPI server.
