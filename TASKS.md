# TASKS.md — 10-day execution plan

One day ≈ **6 focused hours**: ~4h30 core work, ~1h integration/tests/docs, ~30min contingency. If you finish early, polish. Do not add scope.

Each day ends with a **gate**. If the gate fails, you do not start the next day's work — you use tomorrow's contingency budget on today's gate. The plan absorbs one full lost day without missing the deadline; it does not absorb two.

Timestamps below are offsets from the start of your working block.

---

# WEEK 1 — DATA + ML ENGINE

## DAY 1 — Contract, repo, data access proof

**Goal:** raw AQI + weather is in a pandas DataFrame. Nothing more.

| Time | Task |
|---|---|
| 00:00–00:30 | Create repo. Commit `PROJECT_CONTRACT.md`, `ARCHITECTURE.md`, `TASKS.md`, `DECISIONS.md`, `CLAUDE.md`, `AGENTS.md`, `HANDOFF.md`. Fill `config/config.yaml` with your city's lat/lon and `Asia/Karachi`. |
| 00:30–01:00 | Python 3.11 venv. `requirements.txt`. `.env.example`, `.gitignore`. Verify `pip install tensorflow` actually resolves on your machine **today** — if it doesn't, that's a Day 1 problem, not a Day 5 surprise. |
| 01:00–02:30 | `src/data/open_meteo.py`: `fetch_air_quality(start, end)`, `fetch_weather(start, end)`, `fetch_air_quality_recent()`, `fetch_weather_recent()`. Timeouts, `tenacity` retries with backoff, HTTP error handling, UTC tz-aware timestamps, column-presence validation. |
| 02:30–03:15 | **Small proof:** pull 7 days of AQI and 7 days of weather. Merge on timestamp. Print rows, missing %, duplicate timestamps, min/max time, AQI min/max. |
| 03:15–04:15 | **Backfill probe — do this today, not Day 3.** Request `start_date=2022-08-01&end_date=<today>` from the air-quality endpoint for your coordinates. Confirm it returns data rather than an error. Record the true earliest non-null `us_aqi` timestamp in `DECISIONS.md`. If it refuses a range that long, chunk into 6-month windows with a polite delay between calls. **If the earliest available date is later than 2022-08, that is your dataset start — write it down and move on.** |
| 04:15–05:00 | Tests: expected columns present · timestamps sorted and unique · `us_aqi` not entirely null · weather join produces no row explosion · a bad request raises cleanly. |
| 05:00–06:00 | Cache the full raw pull to `data/raw/*.parquet` (gitignored). Commit. Update `HANDOFF.md`. |

**Gate:** `API → DataFrame → AQI + pollutants + weather`, spanning ≥ 18 months, cached locally.
**Fallback:** if Open-Meteo air quality fails entirely (>90 min), switch to OpenAQ v3 as primary and record ADR. Do not spend Day 1 on AQICN registration.

---

## DAY 2 — EDA, features, targets, leakage tests

**This is the data-science day the brief explicitly asks for.**

| Time | Task |
|---|---|
| 00:00–00:30 | Load full history. Data-quality report: missing % per column, gap lengths, duplicate timestamps, outliers, the effective sampling interval (expect 3-hourly artefacts in pollutant columns). |
| 00:30–02:00 | EDA notebook, **8–10 charts, not 40**: AQI distribution · AQI over time · monthly seasonality · hour-of-day profile · day-of-week profile · PM2.5 vs AQI · wind speed vs AQI · humidity/temperature vs AQI · correlation heatmap · autocorrelation / lag plot of AQI. |
| 02:00–02:45 | For every chart write three lines: **Finding / Possible explanation / Model implication.** This is the section that turns the report from a lab exercise into analysis. |
| 02:45–04:15 | `src/features/build_features.py` — one canonical `build_features(df)` per the locked list in the contract. Closed-left windows (`.shift(1)` before `.rolling()`). Cyclical encodings computed in local time. Wind direction as sin/cos. |
| 04:15–05:00 | `src/features/build_targets.py` — forward 24h/48h/72h means. Drop the trailing rows whose targets are incomplete. |
| 05:00–06:00 | **Leakage tests (the most important tests in the repo).** For a random sample of timestamps `T`: recompute features using only `df[df.event_time <= T]` and assert they equal the vectorised output. Assert no feature column correlates ≥0.999 with any target. Assert target columns are absent from the feature matrix. |

**Gate:** `01_eda.ipynb` renders top to bottom · `build_features` + `build_targets` pass leakage tests.
**Fallback:** if leakage tests are still failing after 90 min, simplify — drop rolling features, keep lags only, fix later. A leaky model is worse than a simple one.

---

## DAY 3 — Hopsworks Feature Store + backfill

| Time | Task |
|---|---|
| 00:00–00:45 | Sign up at `app.hopsworks.ai` (free tier, no card). Create the project. Generate API key. Set `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT` locally and as GitHub repo secrets. Confirm `hopsworks.login()` works from a plain script. |
| 00:45–02:00 | Create `aqi_features_v1` and `aqi_targets_v1` per `ARCHITECTURE.md` §3. Explicit dtypes. `event_time` as the event-time column. |
| 02:00–03:00 | Run `src/pipelines/backfill.py`: fetch → `build_features` → `build_targets` → insert both groups. Insert in chunks; the free tier is not fast. |
| 03:00–04:00 | **Read it back and verify — do not trust a successful insert.** Row count · date range · first 5 / last 5 · null counts per column · no duplicate (`city_id`, `event_time`). Compare against your local DataFrame. |
| 04:00–05:00 | Create Feature View `aqi_fv_v1` joining features and targets. Materialise a training dataset from it. Confirm shape matches expectation. |
| 05:00–06:00 | Screenshots for the report: feature group schema, row count, Feature View, lineage. Commit. Update `HANDOFF.md`. |

**Gate:** delete your local parquet cache and rebuild a full training set from the Feature Store alone.
**Fallback:** 2 hours max on Hopsworks integration problems. Then continue with local parquet, keep the same function signatures, and retry Hopsworks on Day 5's contingency. Do **not** let this block modelling.

---

## DAY 4 — Baselines and honest validation

**Do not open TensorFlow today.**

| Time | Task |
|---|---|
| 00:00–00:45 | Pull training data from the Feature View. Sort chronologically. Chronological split: train / val / test, test = most recent ~15%. Write the split as a reusable function so every model uses the identical split. |
| 00:45–01:30 | **Baselines.** (1) Persistence: last observed AQI carried forward to all three horizons. (2) Seasonal naive: trailing 24h mean carried forward. Score both. These numbers go in every later table. |
| 01:30–02:15 | Ridge: `StandardScaler → Ridge`, one per horizon. Sweep alpha on a small log grid. |
| 02:15–03:45 | RandomForest, one per horizon. `n_estimators` 200–400, `max_depth` None/15/25, `min_samples_leaf` 1/2/4, `max_features` sqrt/0.7. Small `RandomizedSearchCV` with `TimeSeriesSplit`. **Hard stop at 90 minutes** — freeze the best run. |
| 03:45–04:45 | `HistGradientBoostingRegressor`, one per horizon. Usually the quiet winner on this kind of data. |
| 04:45–06:00 | Evaluation harness producing the canonical table: rows = models, columns = D+1 MAE, D+2 MAE, D+3 MAE, mean MAE, mean RMSE, mean R². Persistence is row one. Save as CSV + markdown for the report. |

**Gate:** at least one ML model beats persistence on `mae_mean`. If nothing beats persistence, stop and investigate — that is a signal of a target or split bug, not a modelling failure.

---

## DAY 5 — Deep learning, time-series validation, registry

| Time | Task |
|---|---|
| 00:00–01:30 | TensorFlow MLP: `Input → Dense(128, relu) → Dropout(0.2) → Dense(64, relu) → Dense(32, relu) → Dense(3)`. Scaled inputs. `EarlyStopping` + `ReduceLROnPlateau`. Loss = MAE. No 200-epoch ego training. |
| 01:30–02:15 | Compare all models. **The tree model will probably win. That is a result, not a failure.** Write the sentence: "Tree-based models outperformed the neural architecture on this medium-sized tabular dataset." Do not tune the methodology until deep learning wins. |
| 02:15–03:15 | Rolling-origin validation (`TimeSeriesSplit`, 4–5 folds) on the top two models. Report mean ± std of MAE per fold. Now your result isn't one lucky split. |
| 03:15–04:15 | SHAP on the tree champion. `TreeExplainer` on a sampled background set. Produce: global summary plot, top-10 bar, and **separate importance for day1 / day2 / day3** (this is why we trained per-horizon models). Save artefacts to disk. |
| 04:15–05:30 | Register in Hopsworks Model Registry as `pearls_aqi_forecaster`. Attach: model files, feature list (ordered!), metrics dict (`mae_mean`, `mae_day1..3`, `rmse_mean`, `r2_mean`), training timestamp, model type, data date range, SHAP artefacts. |
| 05:30–06:00 | Implement `get_champion()` — retrieve the registered model with lowest `mae_mean`, not the newest. Test it. |

**Gate:** a fresh Python process can pull the champion from the registry and produce three predictions.
**Fallbacks:** TensorFlow trouble → 2h cap, record the experiment honestly and move on. SHAP trouble → 90 min cap, ship global SHAP only. LSTM → do not start it this week.

---

# WEEK 2 — TURN THE MODEL INTO A SYSTEM

## DAY 6 — Inference pipeline + FastAPI

| Time | Task |
|---|---|
| 00:00–01:30 | `src/inference/predictor.py::predict_next_3_days(city)`: load champion → fetch latest features from the Feature Store → **assert feature order matches the registered feature list** → predict → clip to [0,500] → attach categories and alert level. |
| 01:30–03:00 | FastAPI: `GET /health`, `GET /forecast`, `GET /model-info`, `GET /history?days=7`. Pydantic response models. No auth, no user accounts, no database. |
| 03:00–03:45 | `aqi_category(value)` as the single source of truth for bands and alert levels, imported by both API and dashboard. Unit-test every boundary value (50/51, 100/101, 150/151, 200/201, 300/301, 500). |
| 03:45–04:30 | `GET /explanation` — top contributing features for the current prediction. Optional; skip if behind. |
| 04:30–06:00 | API tests with `TestClient`. Response schema, status codes, behaviour when the registry is empty, behaviour when features are stale. |

**Gate:** `curl /health` → 200; `curl /forecast` → three predictions with categories.
**Fallback:** FastAPI deployment issues → 2h cap, then Streamlit calls `predictor.py` in-process. Keep the API code and document it.

Reference response shape:
```json
{
  "city": "Lahore",
  "generated_at": "2026-08-24T09:00:00Z",
  "model_version": 4,
  "model_type": "HistGradientBoosting",
  "current_aqi": 123,
  "forecast": [
    {"horizon": "day_1", "aqi": 137, "category": "Unhealthy for Sensitive Groups", "alert": "none"},
    {"horizon": "day_2", "aqi": 149, "category": "Unhealthy for Sensitive Groups", "alert": "none"},
    {"horizon": "day_3", "aqi": 161, "category": "Unhealthy", "alert": "warning"}
  ]
}
```

---

## DAY 7 — Streamlit dashboard

**One professional page, not ten.**

| Time | Task |
|---|---|
| 00:00–01:00 | Shell: header with city, last data update, last model training, champion version. |
| 01:00–02:30 | API integration with caching, loading states, and a visible error state when the API is down. |
| 02:30–03:30 | Row 1 — four KPI cards: Current AQI, Day+1, Day+2, Day+3, each colour-coded by category. Row 2 — Plotly chart: past 7 days actual ──── today ─── next 3 days forecast, with a visual break at the forecast origin. |
| 03:30–04:15 | Row 3 — current pollutant/weather drivers (PM2.5, PM10, O₃, NO₂, humidity, wind). Row 4 — SHAP: "Why is AQI expected to rise/fall?" top 5 features in plain language. |
| 04:15–05:00 | Row 5 — model quality card (champion, MAE, RMSE, R², training date). Alert banner when any horizon ≥151. |
| 05:00–06:00 | Polish: mobile width, empty states, CAMS + Open-Meteo attribution in the footer, "forecast, not measurement" disclaimer. |

**Gate:** a stranger opens the dashboard and understands tomorrow's air quality within 5 seconds.

---

## DAY 8 — Automation + deployment

| Time | Task |
|---|---|
| 00:00–00:45 | `ci.yml`: on push/PR → ruff + pytest. Green badge in README. |
| 00:45–02:00 | `hourly_features.yml`, cron `17 * * * *`. **Not `0 * * * *`** — GitHub warns scheduled workflows can be delayed during high-load periods, and the top of the hour is the worst offender. Steps: checkout → setup-python → cached pip install → smoke check → `python -m src.pipelines.hourly_features`. Must also backfill targets for rows now ≥72h old. |
| 02:00–03:00 | `daily_training.yml`, cron `37 3 * * *`. Retrieve training data → train candidates → evaluate → register → **promote only if `mae_mean` beats the incumbent**. Log the decision either way. |
| 03:00–03:45 | Add `workflow_dispatch` to both. Trigger each manually and watch it succeed. Verify new rows appear in Hopsworks and a new model version appears in the registry. |
| 03:45–05:00 | Deploy dashboard to Streamlit Community Cloud (repo-connected, secrets via UI). Deploy API to Hugging Face Spaces (Docker SDK, uvicorn). Point the dashboard at the deployed API URL. |
| 05:00–06:00 | End-to-end smoke test against the *deployed* URLs, not localhost. Screenshots of green Action runs. |

**Gate:** both workflows have at least one green scheduled or dispatched run, and the public dashboard URL loads.
**Fallbacks:** cron not firing → 90 min cap, rely on `workflow_dispatch` and document. Hosting problems → 2h cap, run the dashboard locally for the demo and document the intended deployment. Cloud Run is stretch-only and needs a billing account — do not start it before this day's work is done.

> Note for later: GitHub disables scheduled workflows in repos with ~60 days of no activity. Irrelevant inside 10 days, worth one line in the report's Limitations.

---

## DAY 9 — Break it, then add the consulting layer

| Time | Task |
|---|---|
| 00:00–02:00 | **Adversarial testing.** Simulate and handle: Open-Meteo down · Hopsworks down · missing AQI values · duplicate timestamps · all-null weather column · no model in registry · features 48h stale · prediction returns negative · prediction >500 · network timeout mid-backfill. Every one should degrade gracefully with a clear message, never a stack trace on screen. |
| 02:00–03:00 | Full workflow rehearsal: trigger hourly Action → verify Hopsworks → trigger daily training → verify registry → refresh dashboard → confirm the new prediction appears. Screenshot all three. |
| 03:00–04:00 | *Optional, only if ahead:* OpenAQ v3 validation. Pull real station PM2.5 for your city, compare correlation and bias against CAMS. This is a genuinely strong limitations section — "the production pipeline used a stable global reanalysis source, while independent ground-sensor data was used as a validation layer." |
| 04:00–06:00 | **Operational Decision Support** section. Translate the forecast into decisions: outdoor workforce exposure and shift scheduling · construction and site planning · logistics and last-mile routing · employee travel policy · outdoor event go/no-go · occupational health planning. Frame as `DATA → FORECAST → RISK → ACTION`. Include a worked example: "a Day+2 forecast of 165 triggers X." |

**Gate:** no unhandled exception path remains in the demo flow.

---

## DAY 10 — Report, demo, buffer

**No essential coding remains today.** This day is protection, not production.

| Time | Task |
|---|---|
| 00:00–01:30 | Report draft (structure below). |
| 01:30–02:30 | Final architecture figure + the model comparison table with real numbers. |
| 02:30–03:30 | README: what it does · architecture · install · env vars · run backfill · run training · run API · run dashboard · run tests · deployment · data attribution. Someone cloning it must succeed without asking you anything. |
| 03:30–04:30 | Demo rehearsal, 9 beats: problem → architecture → data → Feature Store → model comparison → SHAP → automation → dashboard → business decision enabled. Time it. Aim under 10 minutes. |
| 04:30–06:00 | Emergency buffer. Broken secret, missing screenshot, UI bug, failed run. **No new features.** |

**Report structure:**
1 Executive summary · 2 Business problem · 3 Requirements · 4 Architecture · 5 Data sources · 6 Data quality · 7 EDA · 8 Feature engineering · 9 Forecasting methodology · 10 Model experiments · 11 Evaluation · 12 Explainability · 13 Feature Store · 14 Model Registry · 15 Automated pipelines · 16 API · 17 Dashboard · 18 Deployment · 19 Limitations · 20 Cost and scalability · 21 Business applications · 22 Future improvements · 23 Conclusion

---

# Critical path

```
D1 data access → D2 features+targets → D3 feature store → D4 baselines
→ D5 champion+registry → D6 inference API → D7 dashboard
→ D8 automation+deploy → D9 hardening+business → D10 report+demo
```
If a task doesn't feed this chain, question whether it belongs in this project.

# Fallback table

| Problem | Max time before falling back | Fall back to |
|---|---:|---|
| Open-Meteo air quality access | 90 min | OpenAQ v3 as primary, record ADR |
| Backfill range rejected | 30 min | Chunk into 6-month windows; accept shorter history |
| Leakage tests failing | 90 min | Lags only, drop rolling features |
| Hopsworks integration | 2 hrs | Local parquet, same interfaces, retry Day 5 |
| RandomForest tuning | 90 min | Freeze best run |
| TensorFlow install/training | 2 hrs | Record experiment honestly, continue |
| SHAP per-prediction | 90 min | Global SHAP only |
| FastAPI deployment | 2 hrs | Streamlit calls predictor in-process |
| Hosting / Cloud Run | 2 hrs | Local demo + documented intended deployment |
| GitHub cron not firing | 90 min | `workflow_dispatch`, document |
| Model barely beats persistence | 0 min hiding it | Report honestly against baseline |
| Vertex AI curiosity | 0 min | ADR-001 is closed |
| Multi-city request | not before Day 9 | Configurable single city first |

---

# POST-10 TICKETS

Tickets added after the original 10-day plan closed. Each still goes through the same review split: Codex implements, Claude reviews.

## TICKET — Switch configured city from Lahore to Karachi

**Raised:** 2026-08-30 by the human owner, during hosting/deployment work.

**Why:** the deployed system (Hopsworks Feature Store, Model Registry champion, live API/dashboard) is currently built entirely on Lahore data (`config/config.yaml`: `city_id: "lahore"`, lat 31.5497 / lon 74.3436). The human wants Karachi as the target city instead.

**This is not a config-only change.** Editing `config/config.yaml`'s city block does not retroactively convert existing history — it just means the pipeline starts writing rows under a new `city_id` with zero backfilled history behind it. Concretely, this ticket needs:

1. Update `config/config.yaml` — `city.id = "karachi"`, `city.name = "Karachi"`, correct lat/lon (~24.8607, 67.0011 — verify precisely), keep `timezone: "Asia/Karachi"`.
2. Re-run the full historical backfill (`src/pipelines/backfill.py`) against Karachi's coordinates — mirrors the Day 3 gate (`ADR-011`): fetch from `2022-08-01` (or the true earliest available date per the Day 1 backfill probe) through today, insert into `aqi_features_v1` / `aqi_targets_v1` under `city_id="karachi"`. Verify row count, date range, 0 duplicate `(city_id, event_time)` keys, matching the same rigor `ADR-011` used.
3. Re-run the full model training/evaluation/registry flow (Day 4/5 equivalent) against the new Karachi training frame — baselines, Ridge, RandomForest, HistGradientBoosting, TensorFlow MLP, rolling-origin validation, SHAP artifacts, register champion in the Model Registry.
4. Regenerate `data/metrics/day5_summary.json` and `data/metrics/shap/{champion}_target_aqi_day{1,2,3}_*` for the new Karachi champion — the dashboard reads these directly (see the Aug 30 session's dashboard fix; they're the exception carved out of `.gitignore`'s `data/metrics/` rule).
5. Confirm `hourly_features.yml` / `daily_training.yml` operate correctly against the new `city_id` on their next scheduled or dispatched run.
6. Existing Lahore rows in Hopsworks can stay (no cost to leaving them; contract's `city_id` key already supports multiple cities coexisting) — just don't let both cities' data get mixed into one training frame. Confirm `load_features()`/`load_targets()`/training code filters or is otherwise scoped correctly if the feature groups end up holding both cities.
7. Update `docs/REPORT.md`, `README.md` city references from Lahore to Karachi where they describe the live system.

**Gate:** `/forecast` on the live API returns Karachi data with a real champion model trained on real Karachi history — not a leftover Lahore artifact and not a stub.

**Note:** this is genuinely close to redoing Day 3 + Day 4/5's work for a new city — budget accordingly, don't try to rush it into the current hosting-verification session.

---

## TICKET — Dashboard visual redesign

**Raised:** 2026-08-30 by the human owner, after seeing the deployed Streamlit dashboard alongside a reference app (`https://atmokhi.streamlit.app/`, a different student's project for Karachi). Screenshots of both were reviewed in the Aug 30 session.

### Priority 1 — fix the readability bug (this is a real bug, not a style preference)

Nearly every non-KPI text element on the current dashboard renders white or near-white against the light cream gradient background, making it functionally invisible:
- The page title "Pearls AQI Predictor" and its subtitle
- The "Configured City" card's value line and sub-lines (city name, timestamps)
- The Plotly chart's legend text ("Past 7 days", "3-day forecast") and the "AQI history and forecast" section header
- The "Plain-language explanation" bullets under SHAP drivers (feature name badges are readable; the sentence text around them is not)
- The entire "Model quality" card body

This pattern — readable dark KPI numbers next to invisible white section text — strongly suggests custom CSS/markdown in `dashboard/app.py` hardcodes `color: white` (or a theme variable) assuming a dark background, while the actual background is light. Find and fix the actual mismatch (either the background or the text color assumption is wrong) — don't just find-and-replace individual white values without understanding why they're there, since some may be intentionally white against a genuinely dark element (e.g. inside a colored KPI card top-border).

### Priority 2 — visual style pass, modeled on the reference app's look

Not a pixel-for-pixel clone, but adopt its clarity:
- Bold, condensed/uppercase headers (reference uses a heavy display font for section titles like "CURRENT AIR QUALITY")
- Solid navy-blue accent cards on a warm cream page background — consistent, high-contrast card treatment throughout instead of the current low-contrast translucent cards
- A circular gauge chart for current AQI (reference shows a half-donut gauge with category bands 0–50/50–100/etc. and a needle) instead of (or alongside) the plain KPI number — this is a nice-to-have, the plain KPI cards are fine functionally, just less immediately legible
- Tabbed navigation instead of one long scroll: e.g. "Live & 3-Day Forecast" / "Model Metrics & SHAP" / "City Data Insights" / "Health Guidelines" as separate tabs (`st.tabs`), each showing a focused subset of what's currently all stacked on one page
- Full SHAP beeswarm summary plot (reference's "Feature Importance & SHAP Values" chart) as an option alongside the existing top-5 bar chart, if the underlying per-sample SHAP values are available/cheap to compute for the champion model — check before committing to this, don't recompute SHAP live in the request path if it's expensive

### Explicitly do NOT copy as-is

- **The reference's "Custom Scenario Simulator" tab's feature set.** Its top SHAP feature is `aqi_lag_1h`. `PROJECT_CONTRACT.md` §4 explicitly drops `aqi_lag_1h`/`pm25_lag_1h` from our feature set — the source data is natively 3-hourly, so a 1-hour lag is a near-duplicate of the current value and was a deliberate exclusion, not an oversight. If a scenario simulator is wanted, it has to run against our actual locked feature list (53 columns) and registered model — exposing that many sliders is likely impractical. **Get an explicit decision from the human before building this tab at all**; it may not be worth doing.
- **The reference's model choice (XGBoost).** Not in our locked stack (`PROJECT_CONTRACT.md` §2: Ridge / RandomForest / HistGradientBoosting / TensorFlow MLP only). Our champion stays whatever `get_champion()` returns — match the reference's *card styling* for showing model metrics, not its model type.
- Any feature engineering choices visible in their SHAP feature names (`aqi_rolling_3h`, `hour_category`, etc.) that don't match our locked feature list in `PROJECT_CONTRACT.md` §4 — cosmetic inspiration only, not a spec to implement against.

**File in scope:** `dashboard/app.py` only. No changes to `src/inference`, `src/models`, or the API — this is a presentation-layer ticket.

**Gate:** every piece of text on the dashboard is legible against its background at default browser zoom, in both the "everything's fine" state and the error state (`_render_error_state`). Existing functional content (KPIs, forecast chart, pollutant drivers, SHAP table, model quality, alert banner) all still present — restyled, not removed.
