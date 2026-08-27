"""Hosted ESPN entry point: resolve week once, preserve KTC, generate reviewed output."""
import argparse

from espn_api.football import League

import main as report
import scrape_values
from config import AppConfig


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-ai", action="store_true")
    args = parser.parse_args(argv)
    if args.overwrite and args.week is None:
        raise ValueError("Corrections require an explicit week")
    config = AppConfig.from_env(year=args.year, week=args.week)
    league = League(config.league_id, config.year, config.espn_s2, config.swid)
    week = args.week if args.week is not None else league.nfl_week - 1
    if week < 1:
        print("No completed week; nothing to generate.")
        return
    snapshot = config.player_values_dir / f"KTC_values_week{week}.csv"
    destination = config.report_root / f"{config.year}Week{week}" / "index.md"
    if destination.exists() and not args.overwrite:
        print("Report already saved; scheduled run leaves it unchanged.")
        return
    if args.overwrite and not snapshot.exists():
        raise ValueError(f"Correction requires the saved KTC snapshot: {snapshot}")
    if not snapshot.exists():
        scrape_values.run(scrape_values.parse_args(["--year", str(config.year), "--week", str(week)]))
    command = ["--year", str(config.year), "--week", str(week)]
    if args.overwrite:
        command.append("--overwrite")
    if args.skip_ai:
        command.append("--skip-ai")
    return report.run(report.parse_args(command))


if __name__ == "__main__":
    run()
