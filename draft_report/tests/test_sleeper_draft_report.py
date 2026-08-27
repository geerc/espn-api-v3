import os
from types import SimpleNamespace

import pandas as pd
import pytest

import draft_report.sleeper_draft_report as draft_report
from draft_report.sleeper_draft_report import (
    PlayerProjection,
    add_kicker_vor_and_rerank,
    build_team_results,
    generate_ai_commentary,
    generate_dummy_picks,
    normalize_name,
    rank_radar_values,
    optimize_lineup,
    pick_summary,
    projection_index,
    render_report,
)


def player(name, position, points, rank=1):
    return PlayerProjection(name, position, "NFL", points, points - 100, rank)


def test_parse_args_loads_local_environment(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_MODEL=test-model\n", encoding="utf-8")
    monkeypatch.setattr(draft_report, "LOCAL_ENV_FILE", env_file)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    args = draft_report.parse_args(["123"])

    assert args.ai_model == "test-model"
    assert os.environ["OPENAI_MODEL"] == "test-model"


def test_normalize_name_handles_suffixes_and_punctuation():
    assert normalize_name("D'Andre Swift Jr.") == "dandreswift"


def test_normalize_name_handles_sleeper_projection_aliases():
    assert normalize_name("Kenny Gainwell") == normalize_name("Kenneth Gainwell")
    assert normalize_name("Chig Okonkwo") == normalize_name("Chigoziem Okonkwo")


def test_optimize_lineup_maximizes_legal_flex_lineup():
    players = [
        player("QB", "QB", 300), player("RB1", "RB", 250), player("RB2", "RB", 200),
        player("WR1", "WR", 240), player("WR2", "WR", 230), player("TE", "TE", 180),
    ]

    score, starters = optimize_lineup(players, ["QB", "RB", "WR", "TE", "FLEX", "BN"])

    assert score == 1200
    assert {item.name for _, item in starters} == {"QB", "RB1", "WR1", "WR2", "TE"}


def test_optimize_lineup_rejects_incomplete_roster():
    with pytest.raises(ValueError, match="Unable to fill"):
        optimize_lineup([player("Only QB", "QB", 10)], ["QB", "RB"])


def test_kicker_vor_is_merged_before_overall_reranking():
    base = pd.DataFrame([
        {"name": "Quarterback", "team": "BUF", "position": "QB", "points": 300, "points_vor": 50, "rank": 1},
    ])
    kickers = pd.DataFrame([
        {"name": "K One", "team": "DAL", "position": "K", "points": 140},
        {"name": "K Two", "team": "BAL", "position": "K", "points": 120},
    ])

    result = add_kicker_vor_and_rerank(base, kickers, baseline={"K": 2})

    assert result.loc[result["name"] == "K One", "points_vor"].item() == 20
    assert result.loc[result["name"] == "Quarterback", "rank"].item() == 1
    assert result.loc[result["name"] == "K One", "rank"].item() == 2


def test_team_results_are_worst_to_best_and_reach_value_use_actual_pick():
    frame = pd.DataFrame([
        {"name": "Alpha QB", "team": "BUF", "position": "QB", "points": 300, "points_vor": 10, "rank": 10},
        {"name": "Alpha RB", "team": "DAL", "position": "RB", "points": 200, "points_vor": 5, "rank": 20},
        {"name": "Beta QB", "team": "KC", "position": "QB", "points": 350, "points_vor": 30, "rank": 1},
        {"name": "Beta RB", "team": "SF", "position": "RB", "points": 250, "points_vor": 20, "rank": 2},
    ])
    picks = [
        {"roster_id": 1, "pick_no": 1, "metadata": {"first_name": "Alpha", "last_name": "QB", "position": "QB"}},
        {"roster_id": 1, "pick_no": 30, "metadata": {"first_name": "Alpha", "last_name": "RB", "position": "RB"}},
        {"roster_id": 2, "pick_no": 2, "metadata": {"first_name": "Beta", "last_name": "QB", "position": "QB"}},
        {"roster_id": 2, "pick_no": 20, "metadata": {"first_name": "Beta", "last_name": "RB", "position": "RB"}},
    ]
    rosters = [{"roster_id": 1, "owner_id": "a"}, {"roster_id": 2, "owner_id": "b"}]
    users = {
        "a": {"display_name": "Alpha Manager", "metadata": {"team_name": "Alpha Team"}},
        "b": {"display_name": "Beta Manager", "metadata": {}},
    }
    league = {"roster_positions": ["QB", "RB"]}

    results, unmatched = build_team_results(
        league=league, rosters=rosters, users=users, picks=picks, projections=projection_index(frame),
    )

    assert unmatched == []
    assert [item["team"] for item in results] == ["Alpha Team", "Beta Manager"]
    assert [item["rank"] for item in results] == [2, 1]
    assert results[0]["reach"][1].name == "Alpha QB"
    assert results[0]["reach"][2] == 9
    assert results[0]["value"][1].name == "Alpha RB"
    assert results[0]["value"][2] == -10


def test_team_results_use_full_sleeper_roster():
    frame = pd.DataFrame([
        {"name": "Kept QB", "team": "BUF", "position": "QB", "points": 300, "points_vor": 20, "rank": 2},
        {"name": "Drafted RB", "team": "DAL", "position": "RB", "points": 200, "points_vor": 10, "rank": 10},
    ])
    rosters = [{"roster_id": 1, "owner_id": "a", "players": ["kept", "drafted"]}]
    catalog = {
        "kept": {"full_name": "Kept QB", "position": "QB", "team": "BUF"},
        "drafted": {"full_name": "Drafted RB", "position": "RB", "team": "DAL"},
    }
    picks = [{
        "roster_id": 1, "pick_no": 5,
        "metadata": {"first_name": "Drafted", "last_name": "RB", "position": "RB"},
    }]

    results, unmatched = build_team_results(
        league={"roster_positions": ["QB", "RB"]}, rosters=rosters,
        users={"a": {"display_name": "Manager"}}, picks=picks,
        projections=projection_index(frame), player_catalog=catalog,
    )

    assert unmatched == []
    assert results[0]["projected_points"] == 500
    assert results[0]["reach"][1].name == "Drafted RB"


def test_dummy_draft_is_reproducible_randomized_snake():
    frame = pd.DataFrame([
        {
            "name": f"Player {index}", "team": "NFL", "position": "RB",
            "points": 300 - index, "points_vor": 100 - index, "rank": index,
        }
        for index in range(1, 25)
    ])

    first = generate_dummy_picks(frame, teams=4, rounds=3, seed=17)
    second = generate_dummy_picks(frame, teams=4, rounds=3, seed=17)
    different = generate_dummy_picks(frame, teams=4, rounds=3, seed=18)

    assert first == second
    assert [pick["draft_slot"] for pick in first] == [1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4]
    assert [pick["metadata"] for pick in first] != [pick["metadata"] for pick in different]


def test_radar_uses_positional_rank_among_teams():
    results = [
        {"position_totals": {position: 100 for position in ("QB", "RB", "WR", "TE", "K", "DST")}},
        {"position_totals": {position: 200 for position in ("QB", "RB", "WR", "TE", "K", "DST")}},
    ]
    rank_radar_values(results)
    assert set(results[0]["position_ranks"].values()) == {2}
    assert set(results[1]["position_ranks"].values()) == {1}
    assert set(results[0]["radar"].values()) == {1}
    assert set(results[1]["radar"].values()) == {2}


def test_radar_all_zero_position_stays_at_zero():
    totals = {position: 100 for position in ("QB", "RB", "WR", "TE", "K", "DST")}
    totals["K"] = 0
    results = [{"position_totals": totals}]

    rank_radar_values(results)

    assert results[0]["position_ranks"]["K"] is None
    assert results[0]["radar"]["K"] == 0
    assert results[0]["position_ranks"]["QB"] == 1
    assert results[0]["radar"]["QB"] == 1


def test_report_contains_only_structured_rankings_and_statistics():
    item = {
        "rank": 1, "roster_id": 7, "team": "Champions", "projected_points": 1234.56,
        "reach": (3, player("Reach", "WR", 100, rank=12), 9),
        "value": (30, player("Value", "RB", 100, rank=10), -20),
    }
    content = render_report(league={"season": "2026", "name": "League"}, results=[item])

    assert "## #1 Champions" in content
    assert "Projected starter points:** 1234.6" in content
    assert "team-7-radar.png" in content
    assert pick_summary(item["reach"]) in content


def test_ai_commentary_is_opt_in_and_uses_only_team_statistics():
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            statistics = __import__("json").loads(kwargs["input"])
            return SimpleNamespace(output_text=f'{statistics["team"]} is ranked #{statistics["overall_rank"]}.')

    client = SimpleNamespace(responses=FakeResponses())
    results = [{
        "team": "Alpha",
        "rank": 2,
        "team_count": 12,
        "projected_points": 1900.25,
        "position_ranks": {"QB": 3, "RB": 8, "K": None},
        "reach": (3, player("Reach", "WR", 100, rank=12), 9),
        "value": (30, player("Value", "RB", 100, rank=10), -20),
    }]

    generate_ai_commentary(
        league={"name": "League"}, results=results, api_key=None,
        model="test-model", client=client,
    )

    assert results[0]["commentary"] == "Alpha is ranked #2."
    assert calls[0]["model"] == "test-model"
    assert calls[0]["store"] is False
    assert __import__("json").loads(calls[0]["input"])["position_ranks"] == {"QB": 3, "RB": 8}


def test_ai_commentary_requires_api_key_without_injected_client():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        generate_ai_commentary(
            league={"name": "League"}, results=[], api_key=None, model="test-model",
        )


def test_report_includes_commentary_only_when_present():
    item = {
        "rank": 1, "roster_id": 7, "team": "Champions", "projected_points": 1234.56,
        "reach": None, "value": None, "commentary": "A concise statistical assessment.",
    }

    content = render_report(league={"season": "2026", "name": "League"}, results=[item])

    assert "A concise statistical assessment." in content
