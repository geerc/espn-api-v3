import pandas as pd

from sleeper_rankings.models import LeagueData, Team
from sleeper_rankings.rankings import add_weekly_change, performance_score, power_rankings


def fixture_data():
    first = Team(1, "a", "Alpha", "A", ["1"], wins=1, scores=[120], opponents=[2])
    second = Team(2, "b", "Beta", "B", ["2"], losses=1, scores=[100], opponents=[1])
    players = {
        "1": {"full_name": "Josh Allen", "position": "QB"},
        "2": {"full_name": "Lamar Jackson", "position": "QB"},
    }
    return LeagueData({"settings": {"playoff_week_start": 15, "playoff_teams": 1}}, [first, second], {}, players)


def test_winner_has_higher_performance_score():
    data = fixture_data()
    assert performance_score(data, data.teams[0], 1) > performance_score(data, data.teams[1], 1)


def test_power_rankings_match_sleeper_ids_to_values():
    data = fixture_data()
    values = pd.DataFrame([
        {"name": "Josh Allen", "position": "QB", "nfl_team": "BUF", "value": 9000},
        {"name": "Lamar Jackson", "position": "QB", "nfl_team": "BAL", "value": 8000},
    ])
    result = power_rankings(data, 1, values)
    assert result.iloc[0]["Team"] == "Alpha"
    assert list(result["roster_id"]) == [1, 2]


def test_weekly_change_uses_roster_id_not_mutable_team_name():
    current = pd.DataFrame([{"Team": "New name", "roster_id": 2, "Power Score": 90}, {"Team": "Alpha", "roster_id": 1, "Power Score": 80}], index=[1, 2])
    previous = pd.DataFrame([{"Team": "Alpha", "roster_id": 1}, {"Team": "Old name", "roster_id": 2}], index=[1, 2])
    result = add_weekly_change(current, previous)
    assert list(result["Weekly Change"]) == ["↑ 1", "↓ 1"]
