import pandas as pd

from report import render_report, write_report_atomic


def test_report_rendering_handles_optional_sections():
    frame = pd.DataFrame({"Team": ["Alpha"], "Power Score": [100]})
    content = render_report(year=2026, week=1, rankings=frame, summary=None, playoff_probabilities=None, expected_standings=frame, luck_index=frame)

    assert "Week 1 2026 Report" in content
    assert "Current Playoff Probabilities" not in content
    assert "Projected Standings" in content


def test_atomic_report_write_creates_parent_directory(tmp_path):
    destination = tmp_path / "2026Week1" / "index.md"

    write_report_atomic(destination, "new report")

    assert destination.read_text() == "new report"
    assert list(destination.parent.glob(".index.md.*")) == []
