# AGENTS.md — instructions for Codex (and any other coding agent)

## Your role on this project: builder

You implement discrete tickets from `TASKS.md`. Claude Code reviews your work against the contract. You then fix the review. Roles occasionally swap; the human will say so explicitly.

## Read before writing any code

1. `PROJECT_CONTRACT.md` — frozen. Overrides your judgement.
2. `ARCHITECTURE.md` — module boundaries and data flow.
3. `DECISIONS.md` — closed decisions, not open questions.
4. `TASKS.md` — the ticket and its acceptance criteria.
5. `HANDOFF.md` — last session's state.

## Rules

1. **Implement the ticket. Nothing else.** No adjacent refactors, no renames, no restructuring, no "while I was in here."
2. **Do not change API contracts, module signatures, file layout, or the feature list.** If the ticket seems to require it, stop and say so instead of doing it.
3. **The stack is locked.** No new libraries. No swapping scikit-learn for XGBoost because it's better. No adding Polars because it's faster.
4. **Never fabricate data.** If a fetch fails, surface the error. Do not generate synthetic rows, do not `np.random` your way to a passing test, do not mock an API into returning plausible-looking AQI numbers outside an explicitly-named test fixture.
5. **Write the tests the ticket asks for**, run them, and paste the real output.
6. **Stop when the acceptance criteria are met.** Do not continue into the next ticket.
7. **One module per session** where possible. Small, reviewable diffs.

## Non-negotiable engineering constraints

- **Timezone:** all stored timestamps are UTC and tz-aware. Convert to `Asia/Karachi` only for (a) calendar features and (b) display. Never mix.
- **No shuffling.** Time-ordered data. Chronological splits, `TimeSeriesSplit` for CV.
- **Closed-left windows.** `.shift(1)` before `.rolling()`. A feature at hour `t` never includes hour `t`'s own value in its aggregate, and never anything after `t`.
- **One `build_features()`.** Backfill and the hourly pipeline call the identical function. Do not write a "simplified version for the hourly job."
- **Feature order matters.** Inference must reconstruct the exact ordered feature list stored with the model in the registry. Assert it, don't assume it.
- **Idempotent pipelines.** Running the hourly job twice must not duplicate rows. Upsert on (`city_id`, `event_time`).
- **Clip predictions** to [0, 500] before they leave `predictor.py`.
- **Secrets from environment only.** Never hardcoded, never logged, never printed in a notebook cell.

## Code style

- Python 3.11, type hints on public functions, docstrings on modules and non-obvious functions.
- `ruff` clean.
- Fail loudly with clear messages. Do not swallow exceptions into a bare `except: pass`.
- Network calls: explicit timeout, `tenacity` retry with backoff, meaningful error on final failure.
- No print-debugging left in committed code; use `logging`.

## Ticket response format

```
## What I built
## Files changed
## Tests added and their output
## Assumptions I made
## What I did NOT do (and why)
```

The "assumptions" section is important. If you had to guess at something the ticket didn't specify, say so — that's exactly what the reviewer needs to check.

## When you finish

Update `HANDOFF.md`.
