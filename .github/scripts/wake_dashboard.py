"""Visit the live Streamlit dashboard with a real browser session.

A plain HTTP GET (curl, requests) cannot keep a Streamlit Community Cloud app
awake: a sleeping app answers with a 303 redirect into streamlit.io's cookie-
and JS-driven auth/wake gateway, which a stateless HTTP client can't complete.
Only a real browser session -- one that can click the wake button and hold a
WebSocket connection -- resets Streamlit's inactivity clock.
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

DASHBOARD_URL = "https://aqipredictor-samzzh.streamlit.app/"
EXPECTED_TITLE = "Pearls AQI Predictor"
WAKE_BUTTON_TEXT = "Yes, get this app back up!"


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(DASHBOARD_URL, wait_until="domcontentloaded", timeout=60_000)

        wake_button = page.get_by_text(WAKE_BUTTON_TEXT)
        try:
            wake_button.first.wait_for(state="visible", timeout=10_000)
            print("App was asleep; clicking the wake button.")
            wake_button.first.click()
        except Exception:
            print("Wake button never appeared; app was likely already awake.")

        try:
            page.wait_for_function(
                f"document.title.includes({EXPECTED_TITLE!r})", timeout=180_000
            )
        except Exception:
            print(
                f"Dashboard never reached title {EXPECTED_TITLE!r}; wake likely failed.",
                file=sys.stderr,
            )
            browser.close()
            return 1

        print("Dashboard is awake and responding.")
        browser.close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
