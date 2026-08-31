# HANDOFF.md

Overwrite the block below at the end of **every** agent session. Keep only the current state plus the last two entries - this file is a baton, not a diary. It exists so the next session doesn't spend twenty minutes rediscovering context.

---

## CURRENT STATE

**Day:** Post-10, Karachi city-switch remains complete; the active production issue is the deployed Render API failing on Hopsworks-backed endpoints while `/health` still succeeds. The dashboard redesign is still on disk, unreviewed, and separate from this outage fix.
**Last updated:** 2026-08-31 by Codex (backend outage diagnosis / hotfix)

**What happened this session**
- Reproduced the public failure directly: Render `/health` returns `200`, while `/model-info` and `/forecast` return `500`, which is why the Streamlit dashboard shows "Dashboard unavailable."
- Verified the repo itself still works against live Hopsworks locally with the current `.env`: `get_champion()` succeeds and returns the Karachi-scoped champion (`version=5`, `city_id=karachi`).
- Traced the likely backend failure to model-registry candidate enumeration: `list_registered_versions()` tried to download every version's manifest, so one unreadable legacy version could crash champion lookup and therefore both `/model-info` and `/forecast`.
- Added a narrow hotfix:
  - `src/models/registry.py` now skips unreadable registry versions during candidate listing and wraps model-registry login failures in a clear `OpenMeteoClientError`.
  - `src/feature_store/store.py` now wraps feature-store login failures in a clear `OpenMeteoClientError`.
  - `src/api/main.py` now converts unexpected backend dependency exceptions on `/forecast`, `/model-info`, and `/history` into readable `503` responses instead of opaque `500`s.
- Added focused regression coverage:
  - `tests/test_registry.py::test_list_registered_versions_skips_version_when_manifest_download_fails`
  - `tests/test_api.py::test_model_info_endpoint_returns_503_for_unexpected_registry_exception`
- Kept this work intentionally separate from the unreviewed redesign/scenario changes already present in `dashboard/app.py`, `src/inference/predictor.py`, `.streamlit/`, and related tests.

**How to verify**
```bash
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\python.exe -m pytest tests/test_registry.py tests/test_api.py -q --basetemp=E:\Projects\aqi_predictor_shine\.tmp\pytest_registry_api -p no:cacheprovider
```
Expected focused pytest result: `17 passed`.

**Current blockers / follow-up**
1. The hotfix still needs to be committed, pushed, and redeployed on Render before the public API/dashboard can recover.
2. If Render still fails after redeploy, the next most likely cause is stale or invalid `HOPSWORKS_API_KEY` / `HOPSWORKS_PROJECT` secrets in Render; the new `503` responses should make that explicit.
3. The dashboard redesign remains a separate review/commit task and should not be bundled into the outage fix.

**Next task**
- Commit only the backend outage hotfix, redeploy Render, and re-check `/health`, `/model-info`, and `/forecast`. Review the dashboard redesign separately afterward.

**Gate (outage hotfix):** Local diagnosis and focused regression coverage PASS. Public recovery still depends on redeploying the updated backend.

---

## PREVIOUS ENTRIES

### 2026-08-30 - Codex (Karachi city-switch implementation)
- Switched `config/config.yaml` to Karachi, added city-scoped Feature Store reads and stale-cache rejection in `backfill.py`, ran a real live Karachi backfill (35,784 rows) and retrain (champion `hist_gradient_boosting`, mae_mean 7.81). See the current-state entry above for the review/fix pass that followed in the same day.

### 2026-08-30 - Claude (automation live verification)
- Reviewed Codex's Day 8 automation diff, hand-verified the `>=72h` target-backfill boundary, added the missing promote-when-better test, and pushed the automation work.
- Live dispatch surfaced three real bugs (`nest_asyncio`, live schema mismatch in both directions); both workflows were then dispatched successfully against the live Hopsworks project, growing the Feature Store and registering Model Registry version 2. See `ADR-012`.
