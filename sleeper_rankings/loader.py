from __future__ import annotations

from .api import SleeperClient
from .models import LeagueData, Team


def completed_week(league: dict) -> int:
    settings = league.get("settings") or {}
    return int(settings.get("last_scored_leg") or 0)


def team_name(user: dict | None, roster_id: int) -> str:
    if not user:
        return f"Roster {roster_id}"
    metadata = user.get("metadata") or {}
    return metadata.get("team_name") or user.get("display_name") or f"Roster {roster_id}"


def load_league(client: SleeperClient, league_id: str, through_week: int | None = None) -> LeagueData:
    league = client.league(league_id)
    users = {user["user_id"]: user for user in client.users(league_id)}
    rosters = client.rosters(league_id)
    week = completed_week(league) if through_week is None else through_week
    regular_weeks = int((league.get("settings") or {}).get("playoff_week_start") or 15) - 1
    matchups = {number: client.matchups(league_id, number) for number in range(1, regular_weeks + 1)}

    teams = []
    for roster in rosters:
        owner_id = roster.get("owner_id")
        user = users.get(owner_id)
        settings = roster.get("settings") or {}
        teams.append(Team(
            roster_id=int(roster["roster_id"]),
            owner_id=owner_id,
            name=team_name(user, int(roster["roster_id"])),
            owner=(user or {}).get("display_name") or "Vacant",
            players=roster.get("players") or [],
            wins=int(settings.get("wins") or 0),
            losses=int(settings.get("losses") or 0),
            ties=int(settings.get("ties") or 0),
        ))

    by_roster = {team.roster_id: team for team in teams}
    for number, rows in matchups.items():
        groups: dict[int, list[int]] = {}
        for row in rows:
            matchup_id = row.get("matchup_id")
            if matchup_id is not None:
                groups.setdefault(int(matchup_id), []).append(int(row["roster_id"]))
        for row in rows:
            roster_id = int(row["roster_id"])
            team = by_roster[roster_id]
            while len(team.scores) < number - 1:
                team.scores.append(0.0)
                team.opponents.append(None)
            team.scores.append(float(row.get("custom_points") if row.get("custom_points") is not None else row.get("points") or 0))
            peers = [item for item in groups.get(int(row.get("matchup_id") or -1), []) if item != roster_id]
            team.opponents.append(peers[0] if peers else None)

    # Roster records contain current totals, which are wrong for historical --week runs.
    # Reconstruct results through the requested week, including Sleeper's league-median game.
    median_game = bool((league.get("settings") or {}).get("league_average_match"))
    for team in teams:
        team.wins = team.losses = team.ties = 0
    for number in range(min(week, regular_weeks)):
        weekly_scores = [team.scores[number] for team in teams if number < len(team.scores)]
        median = sorted(weekly_scores)[len(weekly_scores) // 2 - 1:len(weekly_scores) // 2 + 1]
        median_score = sum(median) / len(median) if median else 0
        for team in teams:
            if number >= len(team.scores):
                continue
            opponent_id = team.opponents[number]
            if opponent_id is not None:
                opponent_score = by_roster[opponent_id].scores[number]
                if team.scores[number] > opponent_score:
                    team.wins += 1
                elif team.scores[number] < opponent_score:
                    team.losses += 1
                else:
                    team.ties += 1
            if median_game:
                if team.scores[number] > median_score:
                    team.wins += 1
                elif team.scores[number] < median_score:
                    team.losses += 1
                else:
                    team.ties += 1
    return LeagueData(league=league, teams=teams, matchups=matchups, players=client.players())
