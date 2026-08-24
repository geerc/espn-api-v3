from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import main


def test_week_one_run_omits_weekly_change_and_creates_report(monkeypatch, tmp_path):
    config = SimpleNamespace(
        league_id=1,
        year=2026,
        week=1,
        espn_s2=None,
        swid=None,
        player_values_dir=tmp_path,
        names_file=tmp_path / "names.csv",
        report_root=tmp_path / "reports",
        simulations=10,
        random_seed=1,
        openai_api_key=None,
        ai_model="unused",
    )
    team = SimpleNamespace(team_name="Alpha", wins=1, losses=0, scores=[100])
    league = SimpleNamespace(teams=[team], settings=SimpleNamespace(reg_season_count=4, playoff_team_count=1))
    rankings = pd.DataFrame({"Team": ["Alpha"], "Power Score": [100], "Performance Rank": [1], "KTC Value Rank": [1]}, index=[1])
    expected = pd.DataFrame({"Team": ["Alpha"], "Projected Wins": [1], "Projected Losses": [0]}, index=[1])
    luck = pd.DataFrame({"Team": ["Alpha"], "Luck Index": [0]}, index=[1])

    monkeypatch.setattr(main.AppConfig, "from_env", lambda **kwargs: config)
    monkeypatch.setattr(main, "League", lambda *args, **kwargs: league)
    monkeypatch.setattr(main, "generate_power_rankings", lambda *args, **kwargs: (rankings, pd.DataFrame()))
    monkeypatch.setattr(main, "generate_expected_standings", lambda *args, **kwargs: expected)
    monkeypatch.setattr(main, "generate_luck_index", lambda *args, **kwargs: luck)

    output = main.run(Namespace(year=2026, week=1, simulations=10, seed=1, skip_ai=True, output=None))

    assert output.exists()
    assert "Weekly Change" not in output.read_text()


def test_missing_previous_week_values_is_nonfatal(monkeypatch, tmp_path, capsys):
    config = SimpleNamespace(
        league_id=1, year=2026, week=2, espn_s2=None, swid=None,
        player_values_dir=tmp_path, names_file=tmp_path / "names.csv",
        report_root=tmp_path / "reports", simulations=10, random_seed=None,
        openai_api_key=None, ai_model="unused",
    )
    team = SimpleNamespace(team_name="Alpha", wins=1, losses=1, scores=[100])
    league = SimpleNamespace(teams=[team], settings=SimpleNamespace(reg_season_count=4, playoff_team_count=1))
    rankings = pd.DataFrame({"Team": ["Alpha"], "Power Score": [100], "Performance Rank": [1], "KTC Value Rank": [1]}, index=[1])
    table = pd.DataFrame({"Team": ["Alpha"]}, index=[1])

    monkeypatch.setattr(main.AppConfig, "from_env", lambda **kwargs: config)
    monkeypatch.setattr(main, "League", lambda *args, **kwargs: league)
    monkeypatch.setattr(main, "generate_power_rankings", lambda *args, **kwargs: (rankings, pd.DataFrame()))
    monkeypatch.setattr(main, "generate_expected_standings", lambda *args, **kwargs: table)
    monkeypatch.setattr(main, "generate_luck_index", lambda *args, **kwargs: table)

    main.run(Namespace(year=2026, week=2, simulations=10, seed=None, skip_ai=True, output=tmp_path / "report.md"))

    assert "weekly change will be omitted" in capsys.readouterr().out


def test_ai_failure_is_nonfatal(monkeypatch, tmp_path, capsys):
    config = SimpleNamespace(
        league_id=1, year=2026, week=1, espn_s2=None, swid=None,
        player_values_dir=tmp_path, names_file=tmp_path / "names.csv",
        report_root=tmp_path / "reports", simulations=10, random_seed=None,
        openai_api_key="key", ai_model="model",
    )
    team = SimpleNamespace(team_name="Alpha", wins=1, losses=0, scores=[100])
    league = SimpleNamespace(teams=[team], settings=SimpleNamespace(reg_season_count=4, playoff_team_count=1))
    rankings = pd.DataFrame({"Team": ["Alpha"], "Power Score": [100], "Performance Rank": [1], "KTC Value Rank": [1]}, index=[1])
    table = pd.DataFrame({"Team": ["Alpha"]}, index=[1])

    monkeypatch.setattr(main.AppConfig, "from_env", lambda **kwargs: config)
    monkeypatch.setattr(main, "League", lambda *args, **kwargs: league)
    monkeypatch.setattr(main, "generate_power_rankings", lambda *args, **kwargs: (rankings, pd.DataFrame()))
    monkeypatch.setattr(main, "generate_expected_standings", lambda *args, **kwargs: table)
    monkeypatch.setattr(main, "generate_luck_index", lambda *args, **kwargs: table)
    monkeypatch.setattr(main, "generate_ai_summary", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    output = main.run(Namespace(year=2026, week=1, simulations=10, seed=None, skip_ai=False, output=tmp_path / "report.md"))

    assert output.exists()
    assert "AI summary failed" in capsys.readouterr().err


def test_main_rejects_invalid_simulation_count():
    result = main.main(["--simulations", "0", "--skip-ai"])
    assert result == 2
