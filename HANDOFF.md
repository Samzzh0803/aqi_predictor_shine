# HANDOFF.md

Overwrite the block below at the end of **every** agent session. Keep only the current state plus the last two entries - this file is a baton, not a diary. It exists so the next session doesn't spend twenty minutes rediscovering context.

---

## CURRENT STATE

**Day:** Post-10, hosting package prepared. Feature Store, Model Registry, hourly refresh, and daily training/promotion are all genuinely live on real infrastructure. Deployment files and manual cloud setup instructions now exist for the API and dashboard. Remaining gap is creating the actual public deployments and filling in their URLs.
**Last updated:** 2026-08-30 by Codex

**What happened this session**
- Added `docker/Dockerfile.api` for Hugging Face Spaces' Docker SDK. It installs from the existing `requirements.txt` unchanged, copies only `src/` plus `config/`, and runs `uvicorn src.api.main:app --host 0.0.0.0 --port 7860`.
- Added `.dockerignore` excluding `data/`, `notebooks/`, `.venv/`, `tests/`, `docs/`, and `.git/` so the API image stays small and does not ship local caches or development-only content.
- Updated `.env.example` to include `API_BASE_URL`, since the Streamlit deployment needs it even though it is not a secret.
- Updated `README.md` Deployment and Known Limitations sections with placeholder public URLs, Docker build/run commands, and manual Hugging Face Spaces plus Streamlit Community Cloud setup steps.
- Updated `docs/REPORT.md` section 18 to reflect that hosting is now packaging/configuration work rather than an unresolved architecture blocker.
- Could not perform the requested local Docker verification here because this environment does not have the `docker` CLI installed. The commands are documented exactly, but the build/run smoke test still needs to be executed on a Docker-enabled machine.

**How to verify**
```bash
docker build -f docker/Dockerfile.api -t pearls-aqi-api .
docker run --rm -p 7860:7860 -e HOPSWORKS_API_KEY=... -e HOPSWORKS_PROJECT=... pearls-aqi-api
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/forecast
```
Then create the Streamlit Community Cloud app with `dashboard/app.py` as the entrypoint, Python 3.11, `API_BASE_URL=<hugging-face-space-url>`, and the same Hopsworks secrets in the app settings UI.

**Current blockers / follow-up**
1. The actual Hugging Face Space and Streamlit Community Cloud app still need to be created manually; no public URLs exist yet.
2. Local Docker smoke testing is still outstanding because the current environment lacks Docker.
3. On a fresh Windows dev machine, `pip install -r requirements.txt` can still hit the `twofish` build failure unless a compiler is available - see `ADR-011` for the local workaround.

**Next task**
- Create the Hugging Face Docker Space, add `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` as Space secrets, deploy the API on port 7860, then create the Streamlit Community Cloud app pointing `API_BASE_URL` at the Space URL.

**Gate (Hosting):** PARTIAL. Deployment files and instructions are ready; actual public cloud deployments and Docker smoke verification still need to be completed outside this environment.

---

## PREVIOUS ENTRIES

### 2026-08-30 - Claude (automation live verification)
- Reviewed Codex's Day 8 automation diff, hand-verified the `>=72h` target-backfill boundary, added the missing promote-when-better test, and pushed the automation work.
- Live dispatch surfaced three real bugs (`nest_asyncio`, live schema mismatch in both directions); both workflows were then dispatched successfully against the live Hopsworks project, growing the Feature Store and registering Model Registry version 2. See `ADR-012`.

### 2026-08-29 - Codex (Day 8, hourly/daily automation authored)
- Added `src/pipelines/hourly_features.py` and `run_daily_training_job()`; added both workflow YAMLs and offline tests. Code-complete and offline-tested before the later live dispatch/review.

### 2026-08-29 - Claude (Feature Store swap to live Hopsworks)
- Swapped `src/feature_store/store.py` from the Day 3 local-Parquet fallback to real Hopsworks. Six real issues fixed getting an unmocked run working - see `ADR-011`. Full historical backfill run live (35,712 / 35,545 rows), Day 3 gate re-verified from Hopsworks alone.
