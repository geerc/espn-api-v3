import argparse
import sys
from pathlib import Path

try:
    from .sleeper_draft_report import generate_dummy_picks, load_ffanalytics, write_dummy_draft
except ImportError:  # Support direct execution from the draft_report directory.
    from sleeper_draft_report import generate_dummy_picks, load_ffanalytics, write_dummy_draft


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create a randomized snake draft from ffanalytics rankings.")
    parser.add_argument("league_id", help="Sleeper league ID recorded in the dummy file")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--projections", type=Path, required=True, help="ffanalytics projection CSV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teams", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        projections = load_ffanalytics(args.projections)
        picks = generate_dummy_picks(
            projections, teams=args.teams, rounds=args.rounds, seed=args.seed,
        )
        write_dummy_draft(
            args.output, league_id=args.league_id, season=args.season, teams=args.teams,
            rounds=args.rounds, seed=args.seed, picks=picks,
        )
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"Dummy draft written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
