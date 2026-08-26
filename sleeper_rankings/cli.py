from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .api import SleeperClient
from .loader import completed_week, load_league
from .rankings import add_weekly_change, luck_index, playoff_probabilities, power_rankings, projected_standings
from .render import render_preseason, render_site
from .summary import generate_summary
from .values import fetch_values


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate a Sleeper power rankings site")
    result.add_argument("--config", type=Path, default=Path("leagues/sypip.json"))
    result.add_argument("--league-id")
    result.add_argument("--week", type=int)
    result.add_argument("--output", type=Path, default=Path("dist"))
    result.add_argument("--skip-ai", action="store_true")
    return result


def run(args: argparse.Namespace) -> Path:
    load_dotenv()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    league_id = args.league_id or os.getenv("SLEEPER_LEAGUE_ID") or config.get("league_id")
    if not league_id:
        raise ValueError("A Sleeper league ID is required")
    client = SleeperClient()
    league = client.league(str(league_id))
    week = args.week if args.week is not None else completed_week(league)
    if week < 1:
        return render_preseason(
            output=args.output,
            title=config.get("title", f'{league["name"]} Power Rankings'),
            league_name=league["name"],
            season=league["season"],
        )
    data = load_league(client, str(league_id), through_week=week)
    if not any(team.players for team in data.teams):
        raise ValueError("The league has no populated rosters yet")
    values = fetch_values(float(config.get("dynasty_weight", 0)))
    rankings = power_rankings(data, week, values)
    if week > 1:
        rankings = add_weekly_change(rankings, power_rankings(data, week - 1, values))
    standings = projected_standings(data, rankings, week)
    luck = luck_index(data, week)
    playoffs = None
    regular_weeks = int(data.league["settings"].get("playoff_week_start", 15)) - 1
    if 5 <= week < regular_weeks:
        playoffs = playoff_probabilities(data, rankings, week, int(config.get("simulations", 100000)), config.get("random_seed"))
    ai_enabled = bool(config.get("ai_recap", False)) and not args.skip_ai
    summary = generate_summary(data, week, os.getenv("OPENAI_MODEL", "gpt-5-mini")) if ai_enabled else None
    return render_site(output=args.output, title=config.get("title", f'{data.league["name"]} Power Rankings'), league_name=data.league["name"], season=data.league["season"], week=week, rankings=rankings, summary=summary, playoffs=playoffs, standings=standings, luck=luck)


def main(argv=None) -> int:
    try:
        path = run(parser().parse_args(argv))
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    print(f"Site written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
