# HANDOFF.md

Overwrite the block below at the end of **every** agent session. Keep only the current state plus the last two entries - this file is a baton, not a diary. It exists so the next session doesn't spend twenty minutes rediscovering context.

---

## CURRENT STATE

**Day:** Post-10, automation built. Feature Store + Model Registry are live; hourly/daily workflows now exist. Remaining major gap is public hosting plus live workflow dispatch verification.
**Last updated:** 2026-08-29 by Codex

**What happened this session**
- Added `src/pipelines/hourly_features.py`, the missing Day 8 hourly refresh entry point. It loads existing Hopsworks state, fetches an overlap window with enough history for leakage-safe lag/rolling features, upserts only rows newer than the latest stored feature timestamp, and backfills targets only for rows now at least 72 hours old.
- Added `run_daily_training_job()` to `src/pipelines/train.py`. It reuses the existing Day 5 training/evaluation path but changes the automation behavior to compare the fresh candidate against the incumbent and register only if the candidate wins on validation selection metrics.
- Added focused offline tests: `tests/test_hourly_features.py` covers the subtle `>=72h` target-backfill boundary and idempotent replay of the same hourly window; `tests/test_daily_training_job.py` covers promote-vs-skip behavior for the daily automation path.
- Added `.github/workflows/hourly_features.yml` (cron `17 * * * *` + `workflow_dispatch`) and `.github/workflows/daily_training.yml` (cron `37 3 * * *` + `workflow_dispatch`). Both install dependencies, smoke-check required Hopsworks secrets, and then run the corresponding pipeline entry point.
- Updated `README.md` so the new hourly refresh command and automation behavior are discoverable.

**How to verify**
```bash
pytest tests/test_hourly_features.py tests/test_daily_training_job.py tests/test_backfill.py tests/test_train.py -q
ruff check src tests dashboard
```
These checks are green offline, no credentials needed (fakes only). For the live path, `.env` already has working `HOPSWORKS_API_KEY`/`HOPSWORKS_PROJECT`; the new commands are `python -m src.pipelines.hourly_features` and `python -c "from src.pipelines.train import run_daily_training_job; run_daily_training_job()"`.

**Current blockers / follow-up**
1. Workflows are authored but not yet manually dispatched/observed against live GitHub Actions from this session, so the Day 8 gate is only partially verified here.
2. Not publicly hosted - per `ADR-007`, intended path is Streamlit Community Cloud (dashboard) + Hugging Face Spaces (API).
3. On a fresh Windows dev machine, `pip install -r requirements.txt` can still hit the `twofish` build failure unless a compiler is available - see `ADR-011` for the local workaround.

**Next task**
- Claude review of the hourly/daily automation diff, especially the `>=72h` target-backfill logic and the workflow YAML. After that, either manually dispatch both workflows for live verification or move on to hosting setup.

**Gate:** PARTIAL. Day 8 automation code and offline tests are in place and green; live GitHub Actions dispatch plus Hopsworks-side verification still needs to be performed.

---

## PREVIOUS ENTRIES

### 2026-08-29 - Claude (Day 10, report/README/demo)
- Wrote `README.md`, `docs/REPORT.md`, `docs/DEMO.md`; added `ADR-010` recording the Model Registry swap. Full suite 67 passed. See prior git history for detail - superseded by this session's entry above.

### 2026-08-28 - Claude (Day 9, Hopsworks live)
- Diagnosed and fixed four real Hopsworks SDK/scope issues (Windows cert path, a genuine SDK bug in free-tier error handling, two missing API-key scopes, a `description` length limit) to get the Model Registry genuinely working - verified with a real, unmocked pipeline run and a real registered champion.
- Found and fixed a critical `.gitignore` bug (`models/` matching `src/models/`) that had silently excluded the entire registry module from every GitHub push since Day 8, and a missing `tests/__init__.py` that caused CI to fail on Ubuntu (found via the actual GitHub Actions API logs, not guessing).

### 2026-08-28 - Codex (Day 9, adversarial hardening)
- Hardened `src/pipelines/backfill.py` against all-null critical columns and mid-fetch failures; wrote `docs/operational_decision_support.md`.
