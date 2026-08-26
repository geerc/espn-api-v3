from __future__ import annotations

import random

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz, process

from .models import LeagueData, Team


def _normalize(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    return pd.Series(0.5, index=series.index) if span == 0 else (series - series.min()) / span


def performance_score(data: LeagueData, team: Team, week: int) -> float:
    teams = sorted(data.teams, key=lambda item: item.roster_id)
    index = {item.roster_id: number for number, item in enumerate(teams)}
    wins = np.zeros((len(teams), len(teams)), dtype=int)
    for candidate in teams:
        for number in range(min(week, len(candidate.scores))):
            opponent_id = candidate.opponents[number]
            if opponent_id is not None and candidate.scores[number] > data.by_roster[opponent_id].scores[number]:
                wins[index[candidate.roster_id], index[opponent_id]] += 1
    dominance = wins @ wins + wins
    value = int(dominance[index[team.roster_id]].sum())
    scores = team.scores[:week]
    margins = [score - data.by_roster[opponent].scores[number] for number, (score, opponent) in enumerate(zip(scores, team.opponents[:week])) if opponent is not None]
    average_score = sum(scores) / max(week, 1)
    average_margin = sum(margins) / max(week, 1)
    return round(value * 0.8 + int(average_score) * 0.15 + int(average_margin) * 0.05, 2)


def roster_values(data: LeagueData, values: pd.DataFrame) -> dict[int, float]:
    lookup = {str(player_id): player for player_id, player in data.players.items()}
    choices = values["name"].tolist()
    result = {}
    for team in data.teams:
        matched = []
        for player_id in team.players:
            player = lookup.get(str(player_id)) or {}
            if player.get("position") in {"DEF", "K"}:
                continue
            name = player.get("full_name") or " ".join(filter(None, [player.get("first_name"), player.get("last_name")]))
            match = process.extractOne(name, choices, scorer=fuzz.WRatio, score_cutoff=85)
            if match:
                matched.append(float(values.loc[values["name"] == match[0], "value"].iloc[0]))
        result[team.roster_id] = sum(matched) / len(matched) if matched else 0.0
    return result


def power_rankings(data: LeagueData, week: int, values: pd.DataFrame) -> pd.DataFrame:
    roster_value = roster_values(data, values)
    rows = [{"Team": team.name, "roster_id": team.roster_id, "Performance Score": performance_score(data, team, week), "Roster Value": roster_value[team.roster_id]} for team in data.teams]
    frame = pd.DataFrame(rows)
    weight = round(0.5585 * np.exp(-0.1147 * week), 2)
    frame["Power Score"] = ((_normalize(frame["Performance Score"]) * (1 - weight) + _normalize(frame["Roster Value"]) * weight) * 100).round().astype(int)
    frame["Performance Rank"] = frame["Performance Score"].rank(ascending=False, method="min").astype(int)
    frame["KTC Value Rank"] = frame["Roster Value"].rank(ascending=False, method="min").astype(int)
    frame = frame.sort_values(["Power Score", "Performance Score"], ascending=False).reset_index(drop=True)
    frame.index = range(1, len(frame) + 1)
    return frame


def add_weekly_change(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    current = current.copy()
    prior_positions = {int(row.roster_id): rank for rank, row in previous.iterrows()}
    changes = []
    for rank, row in current.iterrows():
        previous_rank = prior_positions.get(int(row.roster_id))
        if previous_rank is None:
            changes.append("New")
        elif previous_rank > rank:
            changes.append(f"↑ {previous_rank - rank}")
        elif previous_rank < rank:
            changes.append(f"↓ {rank - previous_rank}")
        else:
            changes.append("—")
    current.insert(current.columns.get_loc("Power Score") + 1, "Weekly Change", changes)
    return current


def luck_index(data: LeagueData, week: int) -> pd.DataFrame:
    rows = []
    for team in data.teams:
        total = 0.0
        history = np.array(team.scores[:week], dtype=float)
        mean, std = (history.mean(), history.std()) if len(history) else (0, 0)
        for number, score in enumerate(history):
            opponent_id = team.opponents[number]
            if opponent_id is None:
                continue
            all_scores = sorted((item.scores[number] for item in data.teams), reverse=True)
            rank = all_scores.index(score) + 1
            opponent_score = data.by_roster[opponent_id].scores[number]
            won = score > opponent_score
            schedule = (rank - 1) / (len(data.teams) - 1) if won else -(len(data.teams) - rank) / (len(data.teams) - 1)
            historical = np.clip((score - mean) / std / 2, -1, 1) if std else 0
            margin = np.clip((score - opponent_score) / max(min(score, opponent_score), 1) / 0.1, -1, 1)
            close_game = np.sign(margin) * (1 - abs(margin))
            total += schedule * 0.65 + historical * 0.25 + close_game * 0.10
        rows.append({"Team": team.name, "Luck Index": round(total, 2)})
    frame = pd.DataFrame(rows).sort_values("Luck Index", ascending=False).reset_index(drop=True)
    frame.index = range(1, len(frame) + 1)
    return frame


def projected_standings(data: LeagueData, rankings: pd.DataFrame, week: int) -> pd.DataFrame:
    scores = rankings.set_index("roster_id")["Power Score"].to_dict()
    regular_weeks = int(data.league["settings"].get("playoff_week_start", 15)) - 1
    rows = []
    median_game = bool(data.league["settings"].get("league_average_match"))
    ordered_strength = sorted(scores.values())
    for team in data.teams:
        expected = float(team.wins)
        remaining = 0
        for number in range(week, regular_weeks):
            if number >= len(team.opponents) or team.opponents[number] is None:
                continue
            opponent_score = scores[team.opponents[number]]
            denominator = scores[team.roster_id] + opponent_score
            expected += 0.5 if denominator == 0 else scores[team.roster_id] / denominator
            remaining += 1
            if median_game:
                below = sum(value < scores[team.roster_id] for value in ordered_strength)
                equal = sum(value == scores[team.roster_id] for value in ordered_strength) - 1
                expected += (below + equal * 0.5) / max(len(ordered_strength) - 1, 1)
                remaining += 1
        rows.append({"Team": team.name, "Projected Wins": round(expected, 2), "Projected Losses": round(team.losses + remaining - (expected - team.wins), 2)})
    frame = pd.DataFrame(rows).sort_values("Projected Wins", ascending=False).reset_index(drop=True)
    frame.index = range(1, len(frame) + 1)
    return frame


def playoff_probabilities(data: LeagueData, rankings: pd.DataFrame, week: int, simulations: int, seed: int | None) -> pd.DataFrame:
    rng = random.Random(seed)
    teams = data.teams
    index = {team.roster_id: number for number, team in enumerate(teams)}
    playoff_teams = int(data.league["settings"].get("playoff_teams", len(teams) // 2))
    regular_weeks = int(data.league["settings"].get("playoff_week_start", 15)) - 1
    strength = rankings.set_index("roster_id")["Power Score"].to_dict()
    median_game = bool(data.league["settings"].get("league_average_match"))
    counts = np.zeros((len(teams), playoff_teams), dtype=int)
    for _ in range(simulations):
        wins = np.array([team.wins for team in teams], dtype=float)
        points = np.array([sum(team.scores[:week]) for team in teams]) / 1_000_000
        for number in range(week, regular_weeks):
            seen = set()
            for team in teams:
                if number >= len(team.opponents) or team.opponents[number] is None:
                    continue
                pair = tuple(sorted((team.roster_id, team.opponents[number])))
                if pair in seen:
                    continue
                seen.add(pair)
                left, right = pair
                denominator = strength[left] + strength[right]
                probability = 0.5 if denominator == 0 else strength[left] / denominator
                wins[index[left if rng.random() < probability else right]] += 1
            if median_game:
                # Rank a noisy weekly strength draw; the top half earns the median-game win.
                weekly_order = sorted(teams, key=lambda item: strength[item.roster_id] + rng.gauss(0, 20), reverse=True)
                for item in weekly_order[: len(teams) // 2]:
                    wins[index[item.roster_id]] += 1
        order = np.argsort(wins + points)[::-1]
        for position, team_index in enumerate(order[:playoff_teams]):
            counts[team_index, position] += 1
    rows = []
    for number, team in enumerate(teams):
        row = {"Team": team.name, "Playoffs": counts[number].sum() / simulations * 100}
        row.update({f"Seed {seed_number + 1}": counts[number, seed_number] / simulations * 100 for seed_number in range(playoff_teams)})
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["Playoffs", "Seed 1"], ascending=False).reset_index(drop=True)
    for column in frame.columns[1:]:
        frame[column] = frame[column].map(lambda value: f"{value:.2f}%")
    frame.index = range(1, len(frame) + 1)
    return frame
