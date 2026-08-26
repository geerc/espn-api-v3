from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Team:
    roster_id: int
    owner_id: str | None
    name: str
    owner: str
    players: list[str]
    wins: int = 0
    losses: int = 0
    ties: int = 0
    scores: list[float] = field(default_factory=list)
    opponents: list[int | None] = field(default_factory=list)

    @property
    def movements(self) -> list[float]:
        result = []
        for week, opponent_id in enumerate(self.opponents):
            if opponent_id is None or week >= len(self.scores):
                result.append(0.0)
            else:
                result.append(self.scores[week])
        return result


@dataclass
class LeagueData:
    league: dict
    teams: list[Team]
    matchups: dict[int, list[dict]]
    players: dict[str, dict]

    @property
    def by_roster(self) -> dict[int, Team]:
        return {team.roster_id: team for team in self.teams}

