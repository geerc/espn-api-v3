from __future__ import annotations

import json
import os

from openai import OpenAI

from .models import LeagueData


def matchup_payload(data: LeagueData, week: int) -> list[dict]:
    by_roster = data.by_roster
    groups: dict[int, list[dict]] = {}
    for row in data.matchups.get(week, []):
        if row.get("matchup_id") is not None:
            groups.setdefault(int(row["matchup_id"]), []).append(row)
    payload = []
    for rows in groups.values():
        if len(rows) != 2:
            continue
        matchup = []
        for row in rows:
            player_points = row.get("players_points") or {}
            starters = set(row.get("starters") or [])
            players = []
            for player_id, points in player_points.items():
                player = data.players.get(str(player_id)) or {}
                players.append({
                    "name": player.get("full_name") or player_id,
                    "position": player.get("position"),
                    "points": points,
                    "starter": player_id in starters,
                })
            matchup.append({
                "team": by_roster[int(row["roster_id"])].name,
                "score": row.get("custom_points") if row.get("custom_points") is not None else row.get("points"),
                "players": players,
            })
        payload.append({"teams": matchup})
    return payload


def generate_summary(data: LeagueData, week: int, model: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    response = OpenAI(api_key=api_key).responses.create(
        model=model,
        instructions=(
            "Write a concise, playful newspaper-style fantasy football recap in Markdown. "
            "Identify winners, close games, standout starters, and bench players who outscored starters at the same position. "
            "Use only the supplied facts; Sleeper projections are unavailable. Do not add a title."
        ),
        input=json.dumps(matchup_payload(data, week)),
    )
    return response.output_text

