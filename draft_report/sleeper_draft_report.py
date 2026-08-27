import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

try:
    from .report import write_report_atomic
except ImportError:  # Support direct execution from the draft_report directory.
    from report import write_report_atomic


SLEEPER_API = "https://api.sleeper.app/v1"
FANTASYPROS_KICKERS = "https://www.fantasypros.com/nfl/projections/k.php?week=draft"
FANTASYPROS_DEFENSES = "https://www.fantasypros.com/nfl/projections/dst.php?week=draft"
RADAR_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
FLEX_ELIGIBILITY = {
    "FLEX": {"RB", "WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
    "REC_FLEX": {"WR", "TE"},
    "WRRB_FLEX": {"WR", "RB"},
}
POSITION_ALIASES = {"DEF": "DST"}
DEFAULT_VOR_BASELINE = {"QB": 13, "RB": 35, "WR": 36, "TE": 13, "K": 8, "DST": 3}
DEFAULT_AI_MODEL = "gpt-5-mini"
LOCAL_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
NFL_TEAM_CODES = {
    "arizona cardinals": "ARI", "atlanta falcons": "ATL", "baltimore ravens": "BAL",
    "buffalo bills": "BUF", "carolina panthers": "CAR", "chicago bears": "CHI",
    "cincinnati bengals": "CIN", "cleveland browns": "CLE", "dallas cowboys": "DAL",
    "denver broncos": "DEN", "detroit lions": "DET", "green bay packers": "GB",
    "houston texans": "HOU", "indianapolis colts": "IND", "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC", "las vegas raiders": "LV", "los angeles chargers": "LAC",
    "los angeles rams": "LAR", "miami dolphins": "MIA", "minnesota vikings": "MIN",
    "new england patriots": "NE", "new orleans saints": "NO", "new york giants": "NYG",
    "new york jets": "NYJ", "philadelphia eagles": "PHI", "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF", "seattle seahawks": "SEA", "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN", "washington commanders": "WAS",
}
PLAYER_NAME_ALIASES = {
    "chigokonkwo": "chigoziemokonkwo",
    "kennygainwell": "kennethgainwell",
}


@dataclass(frozen=True)
class PlayerProjection:
    name: str
    position: str
    team: str
    points: float
    points_vor: float
    vor_rank: int


def api_get(path, *, session=requests):
    response = session.get(f"{SLEEPER_API}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_name(value):
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", value.lower())
    normalized = re.sub(r"[^a-z0-9]", "", value)
    return PLAYER_NAME_ALIASES.get(normalized, normalized)


def normalize_position(value):
    return POSITION_ALIASES.get(str(value).upper(), str(value).upper())


def run_ffanalytics(*, season, scoring_settings, output_path, rscript="Rscript"):
    script = Path(__file__).with_name("ffanalytics_projections.R")
    scoring_path = Path(output_path).with_suffix(".scoring.json")
    scoring_path.write_text(json.dumps(scoring_settings), encoding="utf-8")
    try:
        subprocess.run([rscript, str(script), str(season), str(scoring_path), str(output_path)], check=True)
    finally:
        scoring_path.unlink(missing_ok=True)


def load_ffanalytics(path):
    frame = pd.read_csv(path)
    required = {"first_name", "last_name", "team", "position", "points", "points_vor", "rank"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ffanalytics output is missing columns: {', '.join(sorted(missing))}")
    frame["name"] = (frame["first_name"].fillna("") + " " + frame["last_name"].fillna("")).str.strip()
    frame["position"] = frame["position"].map(normalize_position)
    return frame[["name", "team", "position", "points", "points_vor", "rank"]]


def fetch_fantasypros_position(position, *, session=requests):
    url = FANTASYPROS_KICKERS if position == "K" else FANTASYPROS_DEFENSES
    response = session.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    table = next((item for item in tables if any("Player" in str(column) for column in item.columns)), None)
    if table is None:
        raise ValueError(f"FantasyPros {position} projections table was not found")
    table.columns = [column[-1] if isinstance(column, tuple) else column for column in table.columns]
    player_column = next(column for column in table.columns if "Player" in str(column))
    points_column = next((column for column in table.columns if str(column).strip().upper() in {"FPTS", "POINTS"}), None)
    if points_column is None:
        raise ValueError(f"FantasyPros {position} projections did not contain a points column")
    extracted = table[player_column].astype(str).str.extract(r"^(.*?)\s+([A-Z]{2,3})$", expand=True)
    result = pd.DataFrame({
        "name": extracted[0].fillna(table[player_column]).str.strip(),
        "team": extracted[1].fillna(""),
        "position": position,
        "points": pd.to_numeric(table[points_column], errors="coerce"),
    }).dropna(subset=["points"])
    return result


def add_supplemental_vor_and_rerank(projections, supplemental, baseline=DEFAULT_VOR_BASELINE):
    supplemental = supplemental.copy()
    additions = []
    existing_positions = set(projections["position"].map(normalize_position))
    for position, group in supplemental.groupby("position"):
        position = normalize_position(position)
        if position in existing_positions and position != "K":
            continue
        group = group.sort_values("points", ascending=False).reset_index(drop=True)
        replacement_index = min(baseline[position], len(group)) - 1
        replacement = group.iloc[replacement_index]["points"] if replacement_index >= 0 else 0
        group["points_vor"] = group["points"] - replacement
        additions.append(group)
    combined = pd.concat(
        [projections.drop(columns=["rank"], errors="ignore"), *additions], ignore_index=True,
    )
    combined["rank"] = combined["points_vor"].rank(method="dense", ascending=False).astype(int)
    return combined


def add_kicker_vor_and_rerank(projections, kickers, baseline=DEFAULT_VOR_BASELINE):
    return add_supplemental_vor_and_rerank(projections, kickers, baseline)


def projection_index(frame):
    index = {}
    for row in frame.itertuples(index=False):
        item = PlayerProjection(
            name=str(row.name), position=normalize_position(row.position), team=str(row.team),
            points=float(row.points), points_vor=float(row.points_vor), vor_rank=int(row.rank),
        )
        index[(normalize_name(item.name), item.position)] = item
        if item.position == "DST" and item.name.lower() in NFL_TEAM_CODES:
            index[(normalize_name(NFL_TEAM_CODES[item.name.lower()]), item.position)] = item
        if item.position == "DST" and item.team and item.team.lower() != "nan":
            index[(normalize_name(item.team), item.position)] = item
    return index


def generate_dummy_picks(frame, *, teams, rounds, seed):
    if teams < 2 or rounds < 1:
        raise ValueError("Dummy drafts require at least two teams and one round")
    pool = frame.dropna(subset=["name", "position", "points", "rank"]).copy()
    pool = pool[pool["position"].map(normalize_position).isin(RADAR_POSITIONS)]
    required = teams * rounds
    if len(pool) < required:
        raise ValueError(f"Only {len(pool)} projected players are available for {required} draft picks")
    random = np.random.default_rng(seed)
    # Mostly preserve VOR order while allowing plausible draft-day variation.
    pool["dummy_order"] = pd.to_numeric(pool["rank"]) + random.normal(0, 12, len(pool))
    pool = pool.sort_values(["dummy_order", "rank", "name"]).head(required).reset_index(drop=True)
    picks = []
    for overall, row in enumerate(pool.itertuples(index=False), 1):
        round_number = (overall - 1) // teams + 1
        slot_in_round = (overall - 1) % teams
        draft_slot = slot_in_round + 1 if round_number % 2 else teams - slot_in_round
        first_name, _, last_name = str(row.name).partition(" ")
        picks.append({
            "player_id": f"dummy-{overall}", "roster_id": draft_slot,
            "round": round_number, "draft_slot": draft_slot, "pick_no": overall,
            "metadata": {
                "first_name": first_name, "last_name": last_name,
                "position": normalize_position(row.position), "team": str(row.team),
            },
        })
    return picks


def write_dummy_draft(path, *, league_id, season, teams, rounds, seed, picks):
    payload = {
        "league_id": str(league_id), "season": int(season), "type": "snake",
        "teams": int(teams), "rounds": int(rounds), "seed": int(seed), "picks": picks,
    }
    write_report_atomic(Path(path), json.dumps(payload, indent=2) + "\n")


def pick_name_and_position(pick):
    metadata = pick.get("metadata") or {}
    name = metadata.get("first_name", "") + " " + metadata.get("last_name", "")
    if not name.strip():
        name = metadata.get("full_name") or metadata.get("player_id") or pick.get("player_id", "")
    return name.strip(), normalize_position(metadata.get("position", ""))


def find_projection(index, pick, name, position):
    projection = index.get((normalize_name(name), position))
    if projection is None and position == "DST":
        team = (pick.get("metadata") or {}).get("team", "")
        projection = index.get((normalize_name(team), position))
    return projection


def eligible(player_position, slot):
    slot = normalize_position(slot)
    return player_position == slot or player_position in FLEX_ELIGIBILITY.get(slot, set())


def optimize_lineup(players, roster_positions):
    slots = [slot for slot in roster_positions if slot not in {"BN", "IR", "TAXI"}]
    states = {0: (0.0, [])}
    for player in players:
        next_states = dict(states)
        for mask, (score, selected) in states.items():
            for slot_index, slot in enumerate(slots):
                bit = 1 << slot_index
                if not mask & bit and eligible(player.position, slot):
                    candidate = (score + player.points, selected + [(slot, player)])
                    if candidate[0] > next_states.get(mask | bit, (-math.inf, []))[0]:
                        next_states[mask | bit] = candidate
        states = next_states
    target = (1 << len(slots)) - 1
    if target not in states:
        missing = len(slots) - max((bin(mask).count("1") for mask in states), default=0)
        raise ValueError(f"Unable to fill {missing} starting lineup slot(s) from projected drafted players")
    return states[target]


def team_name(roster_id, rosters, users):
    roster = next(item for item in rosters if int(item["roster_id"]) == int(roster_id))
    user = users.get(str(roster.get("owner_id")), {})
    metadata = user.get("metadata") or {}
    return metadata.get("team_name") or user.get("display_name") or f"Roster {roster_id}"


def league_context(league):
    settings = league.get("settings") or {}
    scoring = league.get("scoring_settings") or {}
    reception_points = float(scoring.get("rec", 0) or 0)
    if reception_points == 1:
        reception_scoring = "full PPR"
    elif reception_points == 0.5:
        reception_scoring = "half PPR"
    elif reception_points == 0:
        reception_scoring = "standard/non-PPR"
    else:
        reception_scoring = f"{reception_points:g} points per reception"
    return {
        "format": "best ball" if int(settings.get("best_ball", 0) or 0) else "managed lineup",
        "reception_scoring": reception_scoring,
        "starting_lineup_slots": [
            normalize_position(position)
            for position in league.get("roster_positions", [])
            if position not in {"BN", "IR", "TAXI"}
        ],
    }


def build_team_results(*, league, rosters, users, picks, projections, player_catalog=None):
    drafted_by_roster = {}
    unmatched = []
    for pick in picks:
        name, position = pick_name_and_position(pick)
        projection = find_projection(projections, pick, name, position)
        if projection is None:
            unmatched.append(f"{name} ({position or 'unknown'})")
            continue
        drafted_by_roster.setdefault(int(pick["roster_id"]), []).append((pick, projection))
    results = []
    for roster in rosters:
        roster_id = int(roster["roster_id"])
        drafted = drafted_by_roster.get(roster_id, [])
        roster_players = [item[1] for item in drafted]
        if player_catalog is not None:
            roster_players = []
            for player_id in roster.get("players") or []:
                data = player_catalog.get(str(player_id), {})
                name = data.get("full_name") or f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
                position = normalize_position(data.get("position", ""))
                projection = projections.get((normalize_name(name), position))
                if projection is None and position == "DST":
                    projection = projections.get((normalize_name(data.get("team", player_id)), position))
                if projection is None:
                    unmatched.append(f"{name or player_id} ({position or 'unknown'})")
                else:
                    roster_players.append(projection)
        score, starters = optimize_lineup(roster_players, league["roster_positions"])
        position_totals = {position: 0.0 for position in RADAR_POSITIONS}
        for _, player in starters:
            if player.position in position_totals:
                position_totals[player.position] += player.points
        deltas = [(int(pick["pick_no"]), player, player.vor_rank - int(pick["pick_no"])) for pick, player in drafted]
        results.append({
            "roster_id": roster_id,
            "team": team_name(roster_id, rosters, users),
            "projected_points": score,
            "position_totals": position_totals,
            "roster_construction": dict(sorted(Counter(player.position for player in roster_players).items())),
            "roster": [
                {
                    "name": player.name,
                    "position": player.position,
                    "season_projection": round(player.points, 1),
                }
                for player in sorted(roster_players, key=lambda player: (player.position, -player.points, player.name))
            ],
            "reach": max(deltas, key=lambda item: item[2]) if deltas else None,
            "value": min(deltas, key=lambda item: item[2]) if deltas else None,
        })
    results.sort(key=lambda item: item["projected_points"])
    for rank, result in enumerate(reversed(results), 1):
        result["rank"] = rank
    unmatched = list(dict.fromkeys(unmatched))
    return results, unmatched


def radar_positions_for_league(roster_positions):
    league_positions = {normalize_position(position) for position in roster_positions}
    return tuple(
        position
        for position in RADAR_POSITIONS
        if position not in {"K", "DST"} or position in league_positions
    )


def rank_radar_values(results, positions=RADAR_POSITIONS):
    team_count = len(results)
    for item in results:
        item["team_count"] = team_count
        item["radar_positions"] = tuple(positions)
    for position in positions:
        values = pd.Series([item["position_totals"][position] for item in results])
        if values.max() == 0:
            for item in results:
                item.setdefault("position_ranks", {})[position] = None
                item.setdefault("radar", {})[position] = 0
            continue
        ranks = values.rank(method="min", ascending=False).astype(int)
        for item, rank in zip(results, ranks):
            item.setdefault("position_ranks", {})[position] = int(rank)
            item.setdefault("radar", {})[position] = team_count + 1 - int(rank)


def render_radar(result, path):
    positions = result.get("radar_positions", RADAR_POSITIONS)
    labels = [
        f'{position}\n#{result["position_ranks"][position]}'
        if result["position_ranks"][position] is not None
        else f"{position}\nN/A"
        for position in positions
    ]
    values = [result["radar"][position] for position in positions]
    team_count = result["team_count"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    figure, axis = plt.subplots(figsize=(4, 4), subplot_kw={"polar": True})
    axis.plot(angles, values, color="#2563eb", linewidth=2)
    axis.fill(angles, values, color="#60a5fa", alpha=0.35)
    axis.set_xticks(angles[:-1], labels)
    axis.set_ylim(0, team_count)
    axis.set_yticklabels([])
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=130, bbox_inches="tight", transparent=True)
    plt.close(figure)


def pick_summary(item):
    if item is None:
        return "N/A"
    pick_no, player, difference = item
    return f"{player.name} — pick {pick_no}, VOR rank {player.vor_rank} ({difference:+d})"


def response_markdown_with_citations(response):
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) != "output_text":
                continue
            text = content.text
            citations = []
            for annotation in getattr(content, "annotations", []):
                if getattr(annotation, "type", None) != "url_citation":
                    continue
                citations.append((
                    annotation.start_index,
                    annotation.end_index,
                    annotation.title or "source",
                    annotation.url,
                ))
            for start, end, title, url in sorted(citations, reverse=True):
                text = f"{text[:start]}[{title}]({url}){text[end:]}"
            return text.strip()
    return response.output_text.strip()


def generate_ai_commentary(*, league, results, api_key, model, client=None):
    if client is None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when --ai-commentary is enabled")
        client = OpenAI(api_key=api_key)
    instructions = (
        "Write as an energetic NFL draft commentator analyzing a fantasy football roster after the draft. "
        "Act as an analyst, not a standings reader: do not recite the projected point total or overall league "
        "rank, because the report already displays them. Research the roster's players using current, reputable "
        "fantasy football sources such as ESPN, FantasyPros, CBS Sports, and official NFL or team coverage. In "
        "4-6 punchy sentences, explain where the roster construction thrives, where it falls short, and which "
        "players or position groups drive that verdict. Explicitly account for the supplied league format and "
        "scoring rules; best ball roster construction and risk tolerance differ from a managed-lineup league. "
        "Use the projections, positional ranks, biggest value, and biggest reach as evidence rather than merely "
        "repeating them. Be engaging, opinionated, colorful, and a little bombastic—celebrate sharp drafting and "
        "call out questionable decisions. Distinguish sourced facts from your analysis, never invent facts, and "
        "include inline citations for web-derived claims. Return prose without a heading or bullet list."
    )
    for result in results:
        statistics = {
            "league": league["name"],
            "league_context": league_context(league),
            "team": result["team"],
            "overall_rank": result["rank"],
            "league_size": result["team_count"],
            "projected_starter_points": round(result["projected_points"], 1),
            "roster_construction": result.get("roster_construction", {}),
            "roster": result.get("roster", []),
            "position_ranks": {
                position: rank
                for position, rank in result["position_ranks"].items()
                if rank is not None
            },
            "biggest_reach": pick_summary(result["reach"]),
            "biggest_value": pick_summary(result["value"]),
        }
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=json.dumps(statistics),
            store=False,
            text={"verbosity": "low"},
            tools=[{"type": "web_search"}],
            tool_choice="auto",
        )
        commentary = response_markdown_with_citations(response)
        if not commentary:
            raise ValueError(f"OpenAI returned empty commentary for {result['team']}")
        result["commentary"] = commentary

def render_report(*, league, results):
    sections = [
        "+++", f'title = "{league["season"]} Post-Draft Rankings"', f'date = "{date.today()}"',
        "draft = false", "+++", "", f"# {league['name']} Post-Draft Rankings", "",
    ]
    for result in results:
        image_name = f"team-{result['roster_id']}-radar.png"
        sections.extend([
            f"## #{result['rank']} {result['team']}", "",
            f"**Projected starter points:** {result['projected_points']:.1f}", "",
            f"![{result['team']} positional strength radar chart]({image_name})", "",
            f"**Biggest Reach:** {pick_summary(result['reach'])}", "",
            f"**Biggest Value:** {pick_summary(result['value'])}", "",
        ])
        if result.get("commentary"):
            sections.extend([result["commentary"], ""])
    return "\n".join(sections)


def parse_args(argv=None):
    load_dotenv(LOCAL_ENV_FILE)
    parser = argparse.ArgumentParser(description="Generate a Sleeper post-draft rankings report.")
    parser.add_argument("league_id", help="Sleeper league ID")
    parser.add_argument("--output", type=Path, help="Output directory (defaults to the website content directory)")
    parser.add_argument("--projections", type=Path, help="Reuse an existing ffanalytics projection CSV")
    parser.add_argument("--dummy-draft", type=Path, help="Use draft picks from a dummy draft JSON file")
    parser.add_argument("--rscript", default="Rscript", help="Rscript executable")
    parser.add_argument("--ai-commentary", action="store_true", help="Add OpenAI-generated commentary for each team")
    parser.add_argument("--ai-model", default=os.getenv("OPENAI_MODEL", DEFAULT_AI_MODEL), help="OpenAI model used for commentary")
    return parser.parse_args(argv)


def run(args):
    league = api_get(f"/league/{args.league_id}")
    dummy = None
    if args.dummy_draft:
        dummy = json.loads(args.dummy_draft.read_text(encoding="utf-8"))
        if str(dummy.get("league_id")) != str(args.league_id):
            raise ValueError("Dummy draft league ID does not match the requested Sleeper league")
        picks = dummy.get("picks") or []
    else:
        drafts = api_get(f"/league/{args.league_id}/drafts")
        complete = [draft for draft in drafts if draft.get("status") == "complete"]
        if not complete:
            raise ValueError("Sleeper league has no completed draft; use --dummy-draft for a preview")
        draft = max(complete, key=lambda item: item.get("last_picked") or item.get("start_time") or 0)
        picks = api_get(f"/draft/{draft['draft_id']}/picks")
    rosters = api_get(f"/league/{args.league_id}/rosters")
    users = {str(user["user_id"]): user for user in api_get(f"/league/{args.league_id}/users")}
    player_catalog = None if args.dummy_draft else api_get("/players/nfl")
    season = int(league["season"])
    with tempfile.TemporaryDirectory() as temporary:
        projection_path = args.projections
        if projection_path is None:
            projection_path = Path(temporary) / "ffanalytics.csv"
            run_ffanalytics(
                season=season, scoring_settings=league.get("scoring_settings", {}),
                output_path=projection_path, rscript=args.rscript,
            )
        base = load_ffanalytics(projection_path)
        if dummy is not None and not picks:
            picks = generate_dummy_picks(
                base, teams=int(dummy.get("teams", len(rosters))),
                rounds=int(dummy.get("rounds", len(league["roster_positions"]))),
                seed=int(dummy.get("seed", 2026)),
            )
        supplemental = pd.concat([
            fetch_fantasypros_position("K"), fetch_fantasypros_position("DST"),
        ], ignore_index=True)
        projections = projection_index(add_supplemental_vor_and_rerank(base, supplemental))
    results, unmatched = build_team_results(
        league=league, rosters=rosters, users=users, picks=picks, projections=projections,
        player_catalog=player_catalog,
    )
    rank_radar_values(results, radar_positions_for_league(league["roster_positions"]))
    if args.ai_commentary:
        generate_ai_commentary(
            league=league, results=results, api_key=os.getenv("OPENAI_API_KEY"), model=args.ai_model,
        )
    output_dir = (
        args.output or Path(__file__).resolve().parent / "reports" / str(season)
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        render_radar(result, output_dir / f"team-{result['roster_id']}-radar.png")
    write_report_atomic(output_dir / "index.md", render_report(league=league, results=results))
    if unmatched:
        print(f"Warning: {len(unmatched)} drafted player(s) had no projection: {', '.join(unmatched)}", file=sys.stderr)
    print(f"Report written to {output_dir / 'index.md'}")
    return output_dir / "index.md"


def main(argv=None):
    try:
        run(parse_args(argv))
    except (OSError, subprocess.CalledProcessError, requests.RequestException, ValueError, OpenAIError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
