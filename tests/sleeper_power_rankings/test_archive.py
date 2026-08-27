import json

import pandas as pd
import pytest

from sleeper_rankings.archive import build_archive, entry_path, with_previous
from sleeper_rankings.render import render_site


def test_saved_rank_not_recomputed_and_missing_history(tmp_path):
    current = pd.DataFrame([{"roster_id": 2, "Team": "Renamed", "Power Score": 90}, {"roster_id": 1, "Team": "A", "Power Score": 80}], index=[1, 2])
    assert with_previous(current, tmp_path, "123", "2026", 2)["Weekly Change"].tolist() == ["No prior snapshot"] * 2
    path = entry_path(tmp_path, "123", "2026", 1)
    path.mkdir(parents=True)
    (path / "rankings.json").write_text(json.dumps([{"roster_id": 1}, {"roster_id": 2}]))
    assert with_previous(current, tmp_path, "123", "2026", 2)["Weekly Change"].tolist() == ["↑ 1", "↓ 1"]


def test_archive_preserves_both_weeks_without_network(tmp_path):
    content, output = tmp_path / "content", tmp_path / "dist"
    frame = pd.DataFrame([{"Team": "Alpha", "Power Score": 50}], index=[1])
    for week in [1, 2]:
        path = entry_path(content, "123", "2026", week)
        render_site(output=path, title="Test", league_name="Test", season="2026", week=week, rankings=frame, summary=None, playoffs=None, standings=frame, luck=frame)
        (path / "report.json").write_text(json.dumps({"season": "2026", "week": week}))
    index = build_archive(content, output, {"title": "Test"})
    assert "Week 1" in index.read_text() and "Week 2" in index.read_text()
    assert (output / "reports/123/2026/week-01/index.html").exists()
    assert (output / "reports/123/2026/week-02/assets/site.css").exists()


def test_invalid_archive_keys_rejected(tmp_path):
    with pytest.raises(ValueError):
        entry_path(tmp_path, "../bad", "2026", 1)


def test_cli_refuses_existing_week(tmp_path, monkeypatch):
    from sleeper_rankings import cli
    class Client:
        def league(self, league_id):
            return {"name": "Test", "season": "2026"}
    monkeypatch.setattr(cli, "SleeperClient", Client)
    entry_path(tmp_path, "123", "2026", 1).mkdir(parents=True)
    args = cli.parser().parse_args(["--league-id", "123", "--week", "1", "--content", str(tmp_path)])
    with pytest.raises(ValueError, match="refusing to overwrite"):
        cli.run(args)


def test_archive_only_never_calls_sleeper(tmp_path, monkeypatch):
    from sleeper_rankings import cli
    def forbidden():
        raise AssertionError("Production archive build must not fetch data")
    monkeypatch.setattr(cli, "SleeperClient", forbidden)
    args = cli.parser().parse_args(["--archive-only", "--content", str(tmp_path / "empty"), "--output", str(tmp_path / "site")])
    assert "Something is coming" in cli.run(args).read_text()
