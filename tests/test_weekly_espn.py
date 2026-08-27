from types import SimpleNamespace

import pytest

import weekly_espn
import scrape_values


def setup(monkeypatch, tmp_path):
    config = SimpleNamespace(league_id=1, year=2026, week=2, espn_s2=None, swid=None, player_values_dir=tmp_path, report_root=tmp_path / "reports")
    monkeypatch.setattr(weekly_espn.AppConfig, "from_env", lambda **kw: config)
    monkeypatch.setattr(weekly_espn, "League", lambda *a: SimpleNamespace(nfl_week=3))
    return config


def test_espn_correction_reuses_snapshot(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    snapshot = tmp_path / "KTC_values_week2.csv"
    snapshot.write_text("saved values")
    monkeypatch.setattr(weekly_espn.scrape_values, "run", lambda *a: pytest.fail("must not scrape"))
    monkeypatch.setattr(weekly_espn.report, "run", lambda args: args)
    result = weekly_espn.run(["--year", "2026", "--week", "2", "--overwrite"])
    assert result.overwrite and result.week == 2
    assert snapshot.read_text() == "saved values"


def test_espn_correction_requires_saved_snapshot(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="saved KTC snapshot"):
        weekly_espn.run(["--week", "2", "--overwrite"])


def test_scraper_never_replaces_existing_values(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    monkeypatch.setattr(scrape_values, "League", lambda *a: SimpleNamespace(nfl_week=3))
    snapshot = tmp_path / "KTC_values_week2.csv"
    snapshot.write_text("original")
    monkeypatch.setattr(scrape_values, "build_session", lambda: pytest.fail("must not scrape"))
    scrape_values.run(scrape_values.parse_args(["--week", "2"]))
    assert snapshot.read_text() == "original"
