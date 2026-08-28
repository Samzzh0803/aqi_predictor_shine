# Demo script

Target: under 10 minutes. Each beat names exactly what to show and say — no slide required, everything is either a terminal, a browser tab, or the dashboard itself.

## Before you start

```bash
# 1. Feature freshness — refresh if the last row is more than ~24h old
python -c "from src.feature_store import load_features; print(load_features()['event_time'].max())"

# 2. API running in one terminal
uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# 3. Dashboard running in another
streamlit run dashboard/app.py
```

---

**1. Problem (30s).** "Air quality changes hour to hour, but decisions about outdoor work, construction, or events get made in advance. A current reading alone doesn't tell you what tomorrow looks like — that's the gap this system closes."

**2. Architecture (60s).** Open `ARCHITECTURE.md`'s diagram or `docs/REPORT.md` §4. Walk the arrow: Open-Meteo → feature engineering (one function, every pipeline) → feature store → four models → Model Registry → API → dashboard. Name the one thing that's still a local fallback (feature store) and the one thing that's genuinely live (Model Registry) — say it plainly, don't gloss over it.

**3. Data (60s).** Open `notebooks/01_eda.ipynb`, scroll to Chart 3 (AQI over time) and Chart 10 (autocorrelation). "35,712 hourly rows, four years, strongly autocorrelated — that's exactly why a naive 'tomorrow looks like today' baseline is hard to beat, and why beating it honestly matters."

**4. Feature Store (30s).** Show `src/feature_store/store.py`'s function signatures next to `ARCHITECTURE.md` §3. "These are the exact interfaces a Hopsworks-backed implementation slots behind — the swap is a backend substitution, not a rewrite."

**5. Model comparison (90s).** Open `data/metrics/day4_model_comparison.csv` or the table in `docs/REPORT.md` §11. Point at persistence (22.70 MAE) vs. the champion Ridge (16.83) — "a 25.9% reduction, which is inside the realistic 20-40% range for this kind of smoothed, autocorrelated target — not suspiciously perfect." Then point at the validation table: "the MLP looked fine on the final test, but rolling-origin validation shows it's the least stable candidate — that's exactly why we select the champion on validation, not on a single split."

**6. SHAP / explainability (60s).** Open one of the `data/metrics/shap/ridge_target_aqi_day1_*.png` files, or the dashboard's SHAP panel. "PM2.5 and recent AQI history dominate — matches what the EDA already showed."

**7. Automation / CI (45s).** Open the GitHub Actions tab, show the green CI run. "This is real and running on every push. Hourly and daily retraining automation is scoped but not built yet — it needs the feature-store swap first, since GitHub Actions runners are ephemeral and need somewhere persistent to read and write features between runs."

**8. Dashboard (90s).** The main event. Point at the KPI row, the alert banner (if any horizon is ≥151), the 7-day-history-plus-forecast chart with the break at "today," the current pollutant readings, and the model-quality card at the bottom. "Someone with no data background can read tomorrow's air quality risk in about five seconds."

**9. Business decision (45s).** Open `docs/operational_decision_support.md`, jump to the worked example. "A Day+2 forecast of 165 — Unhealthy — triggers a specific, written action: move outdoor labor to lower-AQI hours, tighten PPE, issue a same-day advisory. That's the DATA → FORECAST → RISK → ACTION chain the whole system exists to support."

---

## If something breaks live

- **API not running / stale features →** `/forecast` returns a clean `503`, and the dashboard shows its error card, not a stack trace. Say so — it's a feature, not a bug: "the system fails loudly and specifically, it never guesses."
- **Hopsworks unreachable during the demo →** fall back to `data/metrics/day5_summary.json` and the SHAP PNGs already on disk; the Model Registry screenshot from `docs/REPORT.md` §14 stands on its own.
