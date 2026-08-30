# HANDOFF.md

Overwrite the block below at the end of **every** agent session. Keep only the current state plus the last two entries - this file is a baton, not a diary. It exists so the next session doesn't spend twenty minutes rediscovering context.

---

## CURRENT STATE

**Day:** Post-10, Karachi city-switch complete and verified live, including a Model Registry gap Codex's ticket didn't cover. Live Hopsworks now holds both Lahore and Karachi cleanly (feature groups correctly filtered per read; model registry now correctly scoped per city too). `dashboard/app.py`'s separate redesign ticket is done but not yet reviewed or committed.
**Last updated:** 2026-08-30 by Claude (reviewing/fixing Codex's Karachi ticket)

**What happened this session**
- Reviewed Codex's Karachi city-switch diff (`config/config.yaml`, `src/feature_store/store.py`, `src/pipelines/backfill.py`, tests). Verified directly against live Hopsworks rather than trusting the self-report: `aqi_features_v1` genuinely holds 35,784 Karachi rows + 35,760 Lahore rows, and `_filter_to_configured_city` correctly isolates them.
- Found and fixed one test file Codex missed monkeypatching (`tests/test_hourly_features.py`'s `fake_feature_store` fixture) — it was silently reading the real, now-Karachi `config.yaml` against Lahore-era fixture data, failing with "Feature store is empty."
- **Found and fixed a real live-production bug outside the ticket's scope:** `src/models/registry.py`'s `get_champion()` had no city scoping at all — it picked the lowest MAE across *every* registered version regardless of which city trained it. With both cities' data now live, this meant the system could silently serve one city's forecast from a model trained on the other city's climate/pollution patterns the next time either city's daily training job registered a new version. Fixed by stamping `city_id` into the registration manifest (`register_model_version`) and filtering champion selection by the configured city (`get_champion`). Covered by two new tests (`test_get_champion_ignores_lower_mae_version_from_a_different_city`, `test_get_champion_raises_when_no_versions_match_configured_city`).
- Re-registered the live Karachi champion as version `5` (`hist_gradient_boosting`, `city_id=karachi`, `mae_mean=7.81`) by downloading and re-uploading the already-real, already-trained version `4` artifacts under the fixed code — no retraining, no synthetic data. Verified `get_champion()` now returns it correctly against live Hopsworks. Versions 1/2/4 remain in the registry untagged (pre-fix) and are now correctly never selected.
- Corrected `data5_summary.json`'s champion metadata reference in this file (previous entry said "champion `ridge`, version `1`" — actually `hist_gradient_boosting`, and now live as version `5`).

**How to verify**
```bash
.venv\Scripts\python.exe -m pytest tests/ -q
.venv\Scripts\python.exe -m ruff check src tests dashboard
```
79/79 tests pass, ruff clean. To verify the live registry fix directly:
```python
from dotenv import load_dotenv; load_dotenv()
from src.models import registry
print(registry.get_champion().version, registry.get_champion().city_id)  # -> 5 karachi
```

**Current blockers / follow-up**
1. The deployed Render API and Streamlit dashboard still need to be redeployed/restarted so the public URLs stop serving Lahore and start serving Karachi.
2. `hourly_features.yml` / `daily_training.yml` haven't been dispatched against the Karachi configuration yet — do this after redeploy, and confirm the daily job's promotion logic now correctly compares only Karachi-tagged versions.
3. `dashboard/app.py`'s visual redesign ticket (see `TASKS.md`) is implemented on disk but uncommitted and unreviewed — separate from this ticket, needs its own review pass before merging.

**Next task**
- Redeploy the live API/dashboard to pick up the Karachi config and the registry fix, then dispatch `hourly_features.yml` and `daily_training.yml` once each and confirm the public `/forecast` response is Karachi-backed. Review and commit the dashboard redesign separately.

**Gate (Karachi switch):** PASS for repo state, live backfill, and live Model Registry/Feature Store correctness. Still PARTIAL on public redeploy + post-switch workflow dispatch — those remain outside this environment.

---

## PREVIOUS ENTRIES

### 2026-08-30 - Codex (Karachi city-switch implementation)
- Switched `config/config.yaml` to Karachi, added city-scoped Feature Store reads and stale-cache rejection in `backfill.py`, ran a real live Karachi backfill (35,784 rows) and retrain (champion `hist_gradient_boosting`, mae_mean 7.81). See the current-state entry above for the review/fix pass that followed in the same day.

### 2026-08-30 - Claude (automation live verification)
- Reviewed Codex's Day 8 automation diff, hand-verified the `>=72h` target-backfill boundary, added the missing promote-when-better test, and pushed the automation work.
- Live dispatch surfaced three real bugs (`nest_asyncio`, live schema mismatch in both directions); both workflows were then dispatched successfully against the live Hopsworks project, growing the Feature Store and registering Model Registry version 2. See `ADR-012`.
