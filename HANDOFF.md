# HANDOFF.md

Overwrite the block below at the end of **every** agent session. Keep only the current state plus the last two entries - this file is a baton, not a diary. It exists so the next session doesn't spend twenty minutes rediscovering context.

---

## CURRENT STATE

**Day:** 8 of 10 (partial) - repo connected to GitHub, CI is green; automation and hosting deliberately deferred pending Hopsworks.
**Last updated:** 2026-08-28 by Claude

**What happened this session**
- Connected the local repo to `https://github.com/Samzzh0803/aqi_predictor_shine` (merged with the remote's auto-created `.gitignore`, kept ours - it's the tailored one).
- Cleaned real junk that was about to get committed: stray `pip install` log files (`0.43`, `3.7`, `6.31.1`), empty root-level `Untitled*.ipynb` scratch notebooks, `.claude/`, `.virtual_documents/`, `anaconda_projects/` local tooling state. None of that is project content.
- Extended `.gitignore` to cover `data/feature_store/`, `data/metrics/`, `data/model_registry/` - these didn't exist when the original `.gitignore` was written (Hopsworks was going to hold all of that remotely); the local fallback now generates them for real and they were about to get committed.
- **Found and flagged a real architectural gap before building blindly**: every Day 8 deliverable (CI, hourly/daily automation, hosting) needs storage that persists across separate runs. Hopsworks was always meant to be that. The local Parquet fallback isn't - it only exists on this machine's disk. Confirmed concretely: 37 references across 4 test files load `data/raw/aqi_weather_...parquet` directly, which doesn't exist on a clean GitHub Actions checkout.
- **Human decision:** ship CI now (fetch a small live sample from Open-Meteo inside the workflow instead of needing the gitignored multi-year cache), defer hourly/daily automation and live hosting until Hopsworks is actually available. Documented as a known, deliberate limitation, not a bug.
- Built `.github/workflows/ci.yml`: checkout -> setup-python 3.11 -> `ruff check` -> fetch 60 days of real Open-Meteo data into the path tests expect -> `pytest`.
- **Verified the workflow logic locally before trusting the YAML**: moved the real `data/raw`, `data/feature_store`, `data/metrics`, `data/model_registry` aside to simulate a clean checkout, ran the exact same steps the workflow runs. This caught 2 real test bugs that would have broken CI on the very first run.
- Fixed both: `test_targets.py::test_build_targets_drops_incomplete_trailing_rows` hardcoded a "-1 hour" offset that only holds for the original dataset's specific leading-NaN shape (CAMS data starts 2022-08-05, fetch started 2022-08-01) - replaced with a dataset-agnostic recomputation of the actual invariant. `test_backfill.py::test_run_backfill_rebuilds_training_data_from_local_store` hardcoded literal `2022-08-01`/`2022-08-10` dates - now derives them from the loaded data, matching the pattern its own sibling tests already used correctly. Verified both fixes pass against the small CI-sized sample AND the full 4-year local dataset (no regression).
- Made `dashboard/app.py`'s `API_BASE_URL` read from an environment variable (was hardcoded to `localhost:8000`) so a future deployed dashboard can point at a deployed API. No behavior change locally (still defaults to localhost).
- Fixed pre-existing ruff findings across the repo (`tests/test_data.py`, `src/data/open_meteo.py` - import order, `datetime.UTC` alias) so the new CI lint gate starts green instead of failing on unrelated pre-existing issues.
- `python -m ruff check src tests dashboard` -> All checks passed. Full local test suite: 58 passed.

**Deliberately NOT done this session, and why**
- `hourly_features.yml` / `daily_training.yml`: would need to persist the feature store/registry across ephemeral GitHub Actions runs. Without Hopsworks, the only ways to do that are hacky (commit data back to the repo as a snapshot, or rebuild everything from scratch every run). Human chose to wait for Hopsworks rather than build a workaround.
- Streamlit Community Cloud / Hugging Face Spaces accounts and deployment: same underlying blocker - the API and dashboard both read local files (registry, feature store, SHAP artifacts, day5_summary.json) that don't exist anywhere but this machine. Deploying today would either crash or serve nothing real. Deferred alongside automation.
- `docker/Dockerfile.api`: drafted, then deleted - it referenced `data/model_registry/` and `data/feature_store/` as COPY sources, which don't exist in a fresh checkout. Revisit once the persistence question is resolved.

**Files changed this session**
- `.gitignore`
- `.github/workflows/ci.yml` (new)
- `dashboard/app.py`
- `src/data/open_meteo.py`
- `tests/test_backfill.py`
- `tests/test_data.py`
- `tests/test_targets.py`

**How to verify**
```bash
ruff check src tests dashboard
pytest tests/ -q
```
Check the GitHub Actions tab on the repo to confirm `ci.yml` is green on the real remote runner.

**Current blockers / follow-up**
1. Hopsworks account/project status: still not confirmed by the human as of this session. This is now the actual critical-path blocker for the rest of Day 8 (automation + hosting), not just a nice-to-have.
2. Once Hopsworks is live: swap `src/feature_store/store.py` and `src/models/registry.py` to the real backend behind the same function signatures (per ADR-008/009), then `hourly_features.yml`, `daily_training.yml`, and both hosting deployments become straightforward.
3. Never watched a real GitHub Actions run execute (no `gh` CLI in this environment) - only verified the workflow's logic locally by simulating a clean checkout.

**Next task**
- Confirm CI is green on GitHub. Then either (a) chase Hopsworks account/cluster status to unblock automation+hosting, or (b) if Hopsworks stays blocked, come back and deliberately choose one of the other two options from this session's decision (full-rebuild-every-run, or commit-data-as-snapshot) to get a real deployed demo for the report before Day 10.

**Gate (Day 8, partial):** MET for CI only. `ci.yml` exists, verified locally to pass from a simulated clean checkout. Automation and hosting gates are NOT met - deliberately deferred, not silently skipped.

---

## PREVIOUS ENTRIES

### 2026-08-27 - Codex + Claude (Day 7)
- Built the Streamlit dashboard (`dashboard/app.py`): header, alert banner, KPI cards, forecast chart, model-quality card, SHAP view, footer.
- Claude caught a real gap: Row 3 (current PM2.5/PM10/O3/NO2/humidity/wind) was missing, and `show_spinner=False` suppressed the required loading state. Codex fixed both. Verified live against a running FastAPI server (not just mocked tests) - real current-conditions values and full `main()` render confirmed working end to end.

### 2026-08-27 - Codex (Day 6)
- Implemented `src/inference/predictor.py::predict_next_3_days()`, `src/inference/aqi.py` (category/alert single source of truth), `src/api/main.py` (`/health`, `/forecast`, `/model-info`, `/history`).
- Claude verified live against the real registry and refreshed stale local features (fetched fresh Open-Meteo data, re-ran backfill) so `/forecast` returns real `200` predictions, not just a `503`.
