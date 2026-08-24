import argparse
from datetime import datetime
import sys

from espn_api.football import League

from config import AppConfig
from player_values import build_session, merge_values, scrape_rankings, write_csv_atomic


REDRAFT_URL = "https://keeptradecut.com/fantasy-rankings?page={}&filters=QB|WR|RB|TE|DST|PK&format=1"
DYNASTY_URL = "https://keeptradecut.com/dynasty-rankings?page={}&filters=QB|WR|RB|TE&format=1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Scrape weekly KeepTradeCut player values.")
    parser.add_argument("--year", type=int, default=datetime.now().year)
    parser.add_argument("--week", type=int, help="Output week (defaults to ESPN's last completed week)")
    parser.add_argument("--dynasty-weight", type=float, default=0.8)
    return parser.parse_args(argv)


def run(args):
    if not 0 <= args.dynasty_weight <= 1:
        raise ValueError("--dynasty-weight must be between 0 and 1")
    config = AppConfig.from_env(year=args.year, week=args.week)
    league = League(config.league_id, config.year, config.espn_s2, config.swid)
    week = config.week if config.week is not None else league.nfl_week - 1
    if week < 1:
        raise ValueError("No completed fantasy week is available; pass --week explicitly if appropriate")

    with build_session() as session:
        redraft = scrape_rankings(session, REDRAFT_URL, 8, "redraft")
        dynasty = scrape_rankings(session, DYNASTY_URL, 10, "dynasty")
    values = merge_values(redraft, dynasty, args.dynasty_weight)
    destination = config.player_values_dir / f"KTC_values_week{week}.csv"
    write_csv_atomic(values, destination)
    print(f"Wrote {len(values)} player values to {destination}")


def main(argv=None):
    try:
        run(parse_args(argv))
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
