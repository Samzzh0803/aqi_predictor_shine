"""Basic checks for the operational decision-support document."""

from __future__ import annotations

from pathlib import Path


def test_operational_decision_support_document_exists() -> None:
    path = Path("docs/operational_decision_support.md")

    assert path.exists()


def test_operational_decision_support_document_includes_worked_example() -> None:
    content = Path("docs/operational_decision_support.md").read_text(encoding="utf-8")

    assert "DATA -> FORECAST -> RISK -> ACTION" in content
    assert "Day +2 AQI = 165" in content
