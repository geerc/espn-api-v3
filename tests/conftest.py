from types import SimpleNamespace

import pytest


class FakeTeam:
    def __init__(self, name, wins=0, losses=0, scores=None, roster=None):
        self.team_name = name
        self.wins = wins
        self.losses = losses
        self.scores = scores or []
        self.roster = roster or []
        self.schedule = []

    def __str__(self):
        return f"Team({self.team_name})"


class FakeLeague:
    def __init__(self, teams, power_scores=None, scoreboards=None):
        self.teams = teams
        self._power_scores = power_scores or {team.team_name: index + 1 for index, team in enumerate(teams)}
        self._scoreboards = scoreboards or {}

    def power_rankings(self, week):
        return [(str(self._power_scores[team.team_name]), team) for team in self.teams]

    def scoreboard(self, week):
        return self._scoreboards.get(week, [])


@pytest.fixture
def fake_player():
    def factory(name, position="RB", nfl_team="BUF"):
        return SimpleNamespace(name=name, position=position, proTeam=nfl_team)
    return factory


@pytest.fixture
def matchup():
    def factory(home, away):
        return SimpleNamespace(home_team=home, away_team=away)
    return factory
