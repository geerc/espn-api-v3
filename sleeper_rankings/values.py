from __future__ import annotations

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup


REDRAFT_URL = "https://keeptradecut.com/fantasy-rankings?page={}&filters=QB|WR|RB|TE&format=1"
DYNASTY_URL = "https://keeptradecut.com/dynasty-rankings?page={}&filters=QB|WR|RB|TE&format=1"


def _scrape(session: requests.Session, template: str, pages: int) -> pd.DataFrame:
    records = []
    for page in range(pages):
        response = session.get(template.format(page), timeout=(5, 30))
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for row in soup.select("div.onePlayer"):
            name = row.select_one("div.player-name a")
            team = row.select_one("div.player-name span.player-team")
            position = row.select_one("p.position")
            value = row.select_one("div.value")
            if all((name, team, position, value)):
                records.append({
                    "name": name.get_text(strip=True),
                    "position": re.sub(r"\d+$", "", position.get_text(strip=True)),
                    "nfl_team": team.get_text(strip=True),
                    "value": pd.to_numeric(value.get_text(strip=True).replace(",", ""), errors="coerce"),
                })
    frame = pd.DataFrame(records).dropna(subset=["name", "value"]).drop_duplicates("name")
    if frame.empty:
        raise ValueError("KeepTradeCut returned no player values")
    return frame


def fetch_values(dynasty_weight: float = 0.0) -> pd.DataFrame:
    session = requests.Session()
    session.headers["User-Agent"] = "sleeper-power-rankings/0.1"
    redraft = _scrape(session, REDRAFT_URL, 8)
    if dynasty_weight <= 0:
        return redraft
    dynasty = _scrape(session, DYNASTY_URL, 10)
    merged = redraft.merge(dynasty[["name", "value"]], on="name", how="outer", suffixes=("_redraft", "_dynasty"))
    merged["value"] = merged["value_redraft"].fillna(0) * (1 - dynasty_weight) + merged["value_dynasty"].fillna(0) * dynasty_weight
    metadata = pd.concat([redraft, dynasty]).drop_duplicates("name")[["name", "position", "nfl_team"]]
    return merged[["name", "value"]].merge(metadata, on="name", how="left")

