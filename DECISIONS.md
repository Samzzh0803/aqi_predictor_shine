# DECISIONS.md

Architecture Decision Records. Accepted ADRs are **closed** - agents may not reopen them. New decisions get appended with the next number.

---

## ADR-001 - Hopsworks, not Vertex AI, as Feature Store + Model Registry

**Status:** Accepted. Closed until submission.

**Context.** The brief offers "Hopsworks or Vertex AI." In 2026 these are not equivalent choices for a 10-day, zero-budget project.

**Evidence.**
- Vertex AI Feature Store (Legacy/V1) is deprecated, scheduled for full sunset on **17 February 2027**.
- Vertex AI Feature Store **Optimized online serving** is also deprecated: no new features from **17 May 2026**, critical patches only, sunset 17 February 2027. The migration target is **Bigtable online serving**.
- Bigtable at a single node costs roughly **$0.65/hour, ~$470/month**. There is no free tier that makes this viable for a student project.
- The current Vertex path therefore requires BigQuery + IAM + feature groups + feature views + Bigtable + sync configuration.
- Hopsworks Serverless offers a **$0 Free tier: 1 project, Feature Store + Model Registry, no credit card required**, driven by a Python-native SDK that works from a plain GitHub Actions runner.

**Decision.** Hopsworks Serverless Free tier.

**Consequence - important.** Hopsworks **Model Serving is not on the Free tier** (it appears under the paid SaaS tier). We therefore serve predictions ourselves via FastAPI. This is not a workaround to apologise for; self-hosted inference against a managed registry is a normal production pattern.

**For the report.** Frame this as a cost-and-lifecycle architecture decision, not as picking the easy option. "We selected Hopsworks after evaluating Vertex AI, whose current feature-serving path requires Bigtable at roughly $470/month minimum and whose legacy path is scheduled for sunset in February 2027."

---

## ADR-002 - Open-Meteo as primary data source

**Status:** Accepted.

**Context.** The brief suggests AQICN or OpenWeather. Both require API-key registration and have restrictive historical access on free plans.

**Decision.** Open-Meteo Air Quality API (`us_aqi` + pollutants) and Historical Weather API (ERA5) as primary. No key required for non-commercial use.

**Constraints accepted:**
- `past_days` is capped at 0-92; backfill must use `start_date`/`end_date`.
- For Pakistan only the **CAMS global** domain applies: 0.4° (~45 km), natively **3-hourly**, available **August 2022 onwards**. The 11 km hourly European reanalysis does not cover this region.
- Hourly air-quality values are therefore interpolated from 3-hourly model output. Stated in Limitations.
- Free tier is non-commercial. CAMS and Open-Meteo attribution required in README and dashboard footer.
- Day 1 backfill probe for the configured Lahore coordinates (`31.5497`, `74.3436`) returned the first non-null `us_aqi` at **2022-08-05T00:00:00Z**.

**Consequence.** Usable history is roughly 4 years, ~35,000 hourly rows. That is comfortably enough for tree models and adequate for a small MLP; it is *not* enough to justify a large deep architecture, which supports ADR-005.

---

## ADR-003 - GitHub Actions, not Airflow

**Status:** Accepted.

Airflow earns its keep on complex DAGs with backfill semantics, retries, and a scheduler you operate. We have two jobs on simple schedules. Standing up Airflow - or paying for a managed instance - adds infrastructure work without improving the AQI solution. GitHub Actions is explicitly permitted by the brief, gives scheduled workflows plus CI for free, and runs where the code already lives.

**Constraint accepted.** GitHub documents that scheduled workflows can be delayed during periods of high load, with the top of the hour worst affected. We schedule at minute **17** hourly and **03:37** daily to reduce contention. Delays remain possible and are documented rather than fought.

---

## ADR-004 - Per-horizon models instead of MultiOutputRegressor

**Status:** Accepted.

Three separately fitted estimators (day1, day2, day3) rather than one multi-output wrapper.

**Reasons.** SHAP's `TreeExplainer` does not cleanly accept a `MultiOutputRegressor` wrapper. Per-horizon models also permit honest per-horizon feature importance, which is a substantially better analytical result ("wind and current PM2.5 dominate day 1; seasonality dominates day 3"). Cost is 3x fit time, negligible at this data volume.

Exception: the TensorFlow MLP uses a single `Dense(3)` head.

---

## ADR-005 - Deep learning is included as an experiment, not as the expected winner

**Status:** Accepted.

The brief requires multiple modelling approaches including deep learning. With ~35k rows of smoothed, strongly autocorrelated tabular data, gradient-boosted trees are the favourites. We include a TensorFlow MLP, evaluate it fairly, and report whatever happens.

**Explicitly forbidden:** adjusting the methodology, the split, or the metric so that the neural model wins. "Tree-based models outperformed the neural architecture on this dataset" is a legitimate and defensible finding, and defending it well scores better than manufacturing a deep-learning victory.

---

## ADR-006 - Weather-forecast features are NOT used in v1

**Status:** Rejected for v1. Reconsider only after Day 8 is complete.

**Context.** Tomorrow's weather is genuinely knowable at prediction time via the Open-Meteo forecast endpoint, so using forecast wind/precipitation as features is not leakage in production and would likely improve accuracy.

**Why rejected anyway.** Training on *actual* future weather while serving on *forecast* future weather creates training/serving skew and produces optimistic validation numbers. Doing it correctly requires archived historical forecasts (the Historical Forecast API) so training sees the same forecast error the production system will see. That is a meaningful chunk of extra work.

**Decision.** Excluded from v1. This belongs in the report's *Future Improvements* section, described precisely as above - describing the trap you deliberately avoided demonstrates more judgement than falling into it.

---

## ADR-007 - Hosting without a credit card

**Status:** Accepted.

Cloud Run is genuinely serverless and scales to zero, but requires an enabled billing account with a payment method. To keep the deadline safe:

- **Primary:** Streamlit Community Cloud (dashboard) + Hugging Face Spaces, Docker SDK (FastAPI). Both free, neither requires a card.
- **Stretch:** Cloud Run, Day 8 evening only, only if everything else is green.
- **Fallback:** local demo with documented intended deployment.

The system remains architecturally serverless - stateless compute, managed scheduling, managed feature/model storage - regardless of which host runs the container.

---

## ADR-008 - Day 3 local Parquet fallback for Feature Store work

**Status:** Accepted as a temporary contingency on 2026-08-26.

**Context.** `TASKS.md` explicitly allows a Day 3 fallback if Hopsworks integration cannot be completed within the 2-hour cap. The project still targets Hopsworks as the intended platform, but Day 3 implementation work must not block the modelling critical path.

**Decision.** Implement Day 3 storage through local Parquet artifacts with the same public function signatures intended for the Hopsworks-backed version: `insert_features()`, `insert_targets()`, `create_feature_view()`, and the corresponding readback helpers.

**Consequence.** Day 3 and Day 4 work can proceed locally without changing pipeline APIs. A later Hopsworks implementation must swap the backend behind the same interfaces once a human has completed account/project setup and provided `HOPSWORKS_API_KEY` plus `HOPSWORKS_PROJECT`.

---

## ADR-009 - Champion selection uses validation; explainability follows the champion

**Status:** Accepted on 2026-08-26.

**Context.** The original Day 4 and Day 5 wording left room for a subtle mistake: selecting the production champion from final test performance or constraining champion choice to tree models because TreeExplainer is convenient. Both break the intended evaluation story.

**Decision.**
- Preserve an untouched final chronological test set for one-time reporting only.
- Select the champion from time-aware validation results on the pre-test timeline.
- Register the overall champion across Ridge, RandomForest, HistGradientBoosting, and TensorFlow MLP.
- Use a model-specific SHAP explainer for the chosen champion rather than restricting champion selection to tree models.

**Consequence.** Day 5 now follows `train -> tune/rolling validation -> champion selection -> refit on pre-test data -> single final test evaluation -> register -> explain`. This keeps the final test unbiased and ensures the registry champion matches the actual best validation candidate.

---

## ADR-010 - Hopsworks Model Registry swapped in for real; feature store stays local

**Status:** Accepted on 2026-08-28.

**Context.** Hopsworks had been unreachable since Day 3 (reported as a "clustering problem"). It turned out to be four fixable client-side issues, not a platform outage:

1. `hopsworks.login()` defaults to a Unix `/tmp/...` cert path that doesn't exist on Windows.
2. The installed `hopsworks` 5.0.6 package has a real bug in its own free-tier SERVING-scope error handling (`e.response.error_code` should be `e.response.status_code`), causing a crash instead of a graceful skip.
3. The API key's initial scopes lacked `DATASET_CREATE` (needed to create the "Models" dataset Model Registry requires) and `MODELREGISTRY`.
4. `mr.python.create_model(...).save()` failed because Hopsworks' `description` column has a length limit our full JSON metadata (46-item feature list + 12 artifact filenames) exceeded.

All four are worked around or fixed in `src/models/registry.py::_get_model_registry()` (issues 1-2) and by regenerating the API key with broader scopes (issue 3) and moving metadata into an uploaded `extra_manifest.json` file instead of `description` (issue 4).

**Decision.** Swap `src/models/registry.py` to the real Hopsworks Model Registry now that it demonstrably works — verified with a real, unmocked pipeline run, not just a login test. Do **not** swap `src/feature_store/store.py` in the same cycle; keep it on the Day 3 local-Parquet fallback (`ADR-008`).

**Why not swap the feature store too.** Time budget: Day 9 was nearly complete and Day 10 (report + demo) was next. The registry swap alone gives the report genuine "live Hopsworks Model Registry" evidence (a real URL, a real registered version) without the larger, riskier feature-store migration and its knock-on need for `hourly_features.yml`/`daily_training.yml` to persist state across ephemeral GitHub Actions runs — a problem the feature-store swap would also need to solve, since Hopsworks' Feature Store is exactly the piece designed to make that possible.

**Consequence.** `register_model_version`, `get_champion`, `list_registered_versions`, `load_registered_models` keep their exact function signatures, so `train.py`, `predictor.py`, and `api/main.py` needed zero changes. `tests/test_registry.py` and the champion-selection tests in `tests/test_train_day5.py` now use an in-memory fake Hopsworks client (`FakeModelRegistry`) instead of a local `registry_root`, so the suite runs offline with no credentials.

**Also fixed this cycle, recorded here since they're real architecture-adjacent bugs, not typos:**
- `.gitignore` had a bare `models/` pattern intended to exclude a top-level model-artifacts folder, but it matched *any* directory named `models` anywhere in the tree — including `src/models/`. The entire registry module had never been pushed to GitHub since the repo was connected on Day 8. Anchored the pattern to `/models/`.
- `tests/` had no `__init__.py`, so `tests/test_train_day5.py`'s `from tests.test_registry import FakeModelRegistry` resolved as an implicit namespace package whose behavior depends on pytest's file-collection order — which differs between Windows (where it happened to pass) and Ubuntu (GitHub's runner, where it didn't). Added `tests/__init__.py` to make it an explicit package, removing the OS-dependent behavior.

**For the report.** The Model Registry entry at the Hopsworks project URL is real and can be screenshotted directly; the feature store cannot be, and the report should say so plainly rather than imply otherwise.

---

## ADR-011 - [template]

**Status:** Proposed / Accepted / Rejected
**Context.**
**Decision.**
**Consequence.**
**Owner + date.**
