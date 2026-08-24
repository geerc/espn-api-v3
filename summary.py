import json
from urllib.parse import quote_plus

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def _fantasypros_url(player_name):
    """Return a safe lookup URL instead of guessing FantasyPros' canonical slug."""
    return f"https://www.fantasypros.com/nfl/players/?q={quote_plus(str(player_name))}"


def _player_metadata(names_path, values_path, previous_values_path=None):
    names = pd.read_csv(names_path)
    missing = {"Name", "Team"} - set(names.columns)
    if missing:
        raise ValueError(f"Player names file is missing required columns: {', '.join(sorted(missing))}")
    values = pd.read_csv(values_path, usecols=["Player Name", "Value"])
    values["Value"] = pd.to_numeric(values["Value"], errors="coerce")
    urls = {}
    for row in names.dropna(subset=["Name"]).to_dict("records"):
        canonical_url = row.get("URL")
        urls[row["Name"]] = (
            canonical_url
            if pd.notna(canonical_url) and str(canonical_url).strip()
            else _fantasypros_url(row["Name"])
        )
    if previous_values_path and previous_values_path.exists():
        previous = pd.read_csv(previous_values_path, usecols=["Player Name", "Value"])
        previous["Value"] = pd.to_numeric(previous["Value"], errors="coerce")
        values = values.merge(previous, on="Player Name", how="left", suffixes=("", " Previous"))
        values["Value Change"] = values["Value"] - values["Value Previous"]
    else:
        values["Value Change"] = values["Value"]
    return urls, dict(zip(values["Player Name"], values["Value Change"]))


def generate_ai_summary(league, week, names_path, values_path, previous_values_path, api_key, model):
    if not api_key:
        raise ValueError("OPEN_AI_KEY is required unless --skip-ai is used")
    urls, changes = _player_metadata(names_path, values_path, previous_values_path)

    def serialize_player(player):
        return {"player_name": player.name, "slot_position": player.slot_position, "position": player.position, "points": player.points, "projected_points": player.projected_points, "url": urls.get(player.name), "value_change": changes.get(player.name)}

    data = [{"home_team": matchup.home_team.team_name, "home_score": matchup.home_score, "home_projected": matchup.home_projected, "away_team": matchup.away_team.team_name, "away_score": matchup.away_score, "away_projected": matchup.away_projected, "home_players": [serialize_player(player) for player in matchup.home_lineup], "away_players": [serialize_player(player) for player in matchup.away_lineup]} for matchup in league.box_scores(week=week)]
    prompt = PromptTemplate.from_template(
        "Write a concise newspaper-style fantasy football recap from this JSON:\n{box_scores_json}\n\n"
        "BE and IR slot positions are bench players. Identify winners, close games, meaningful projection misses, "
        "and bench players who outscored starters at the same position. URLs are reference metadata only; do not "
        "claim to have visited them or invent external news."
    )
    llm = ChatOpenAI(model=model, temperature=0.4, api_key=api_key, timeout=60, max_retries=2)
    return (prompt | llm).invoke({"box_scores_json": json.dumps(data, indent=2)}).content
