# HANDOFF.md

Overwrite the block below at the end of **every** agent session. Keep only the current state plus the last two entries - this file is a baton, not a diary. It exists so the next session doesn't spend twenty minutes rediscovering context.

---

## CURRENT STATE

**Day:** 10 of 10 - report, README, and demo script written. Project is MVP-complete with two honestly-documented gaps (feature store, hosting).
**Last updated:** 2026-08-29 by Claude

**What happened this session**
- Wrote `README.md` (didn't exist before): what it does, architecture, install, env vars, how to run every pipeline, deployment status, known limitations, data attribution.
- Wrote `docs/REPORT.md`: the full 23-section report from `TASKS.md`, built entirely from real, verified numbers - the Day 4/5 comparison table, the rolling-validation table, real SHAP feature rankings, real dataset row counts, the real Hopsworks Model Registry URL. No placeholder numbers anywhere.
- Wrote `docs/DEMO.md`: a 9-beat, <10-minute demo script, honest about what's live (Model Registry, CI, dashboard, API) vs. deferred (feature store swap, automation, public hosting) - includes a "what to say if it breaks live" section.
- Added `ADR-010` to `DECISIONS.md`, formally recording the Hopsworks Model Registry swap, the four SDK/scope bugs that had to be fixed to get there, and the reasoning for *not* swapping the feature store in the same cycle.
- Verified: full suite still 67 passed after all doc changes (no code touched this session).

**Files changed this session**
- `README.md` (new)
- `docs/REPORT.md` (new)
- `docs/DEMO.md` (new)
- `DECISIONS.md` (ADR-010 added)
- `HANDOFF.md`

**How to verify**
```bash
pytest tests/ -q
ruff check src tests dashboard
```
Read `docs/REPORT.md` end to end - every number in it should trace back to a file in `data/metrics/` or a command run during this project, not to something invented for the report.

**Current blockers / follow-up**
1. Feature store is still local-only (`ADR-008`) - the single biggest remaining gap between this system and the frozen contract's full objective.
2. No hourly/daily automation, no public hosting - both documented honestly in `README.md` §Known limitations and `docs/REPORT.md` §13, §15, §18, §19, not silently glossed over.
3. Everything else in the MVP scope (`PROJECT_CONTRACT.md` §7) is done: backfill, EDA, feature engineering, Ridge/RF/HGB/MLP, persistence baseline, metrics table, live Model Registry, 3-day prediction, FastAPI, Streamlit, SHAP, alerts, CI.

**Next task**
- If more time exists: the feature-store swap (§13, §22 of `docs/REPORT.md`) is the highest-leverage next increment - it directly unblocks automation and hosting, the only two things left.
- Otherwise: this is submittable as-is. Run through `docs/DEMO.md` once end to end before presenting.

**Gate (Day 10):** MET. Report, README, and demo script exist, are grounded in real verified numbers, and honestly state what's built vs. deferred rather than implying completeness that isn't there.

---

## PREVIOUS ENTRIES

### 2026-08-28 - Claude (Day 9, Hopsworks live)
- Diagnosed and fixed four real Hopsworks SDK/scope issues (Windows cert path, a genuine SDK bug in free-tier error handling, two missing API-key scopes, a `description` length limit) to get the Model Registry genuinely working - verified with a real, unmocked pipeline run and a real registered champion.
- Found and fixed a critical `.gitignore` bug (`models/` matching `src/models/`) that had silently excluded the entire registry module from every GitHub push since Day 8, and a missing `tests/__init__.py` that caused CI to fail on Ubuntu (found via the actual GitHub Actions API logs, not guessing).

### 2026-08-28 - Codex (Day 9, adversarial hardening)
- Hardened `src/pipelines/backfill.py` against all-null critical columns and mid-fetch failures; wrote `docs/operational_decision_support.md`.
