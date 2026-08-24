import os
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


COLUMNS = ["Player Name", "Pos", "Value", "NFL_Team"]


def build_session():
    session = requests.Session()
    retries = Retry(total=3, connect=3, read=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers["User-Agent"] = "jtown-fantasy-rankings/1.0"
    return session


def scrape_rankings(session, url_template, pages, label):
    players = []
    for page in range(pages):
        response = session.get(url_template.format(page), timeout=(5, 30))
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for row in soup.find_all("div", class_="onePlayer"):
            name_block = row.find("div", class_="player-name")
            name = name_block.find("a") if name_block else None
            team = name_block.find("span", class_="player-team") if name_block else None
            position = row.find("p", class_="position")
            value = row.find("div", class_="value")
            if not all((name, team, position, value)):
                continue
            players.append({"Player Name": name.get_text(strip=True), "Pos": position.get_text(strip=True), "Value": value.get_text(strip=True), "NFL_Team": team.get_text(strip=True)})
    if not players:
        raise ValueError(f"{label} scraper returned no players; refusing to replace existing data")
    frame = pd.DataFrame(players, columns=COLUMNS).drop_duplicates("Player Name")
    frame["Value"] = pd.to_numeric(frame["Value"].str.replace(r"[$,]", "", regex=True), errors="coerce")
    frame = frame.dropna(subset=["Player Name", "Value"])
    if frame.empty:
        raise ValueError(f"{label} scraper returned no valid player values")
    return frame


def merge_values(redraft, dynasty, dynasty_weight=0.8):
    dynasty = dynasty.copy()
    dynasty["Value"] = (dynasty["Value"] * dynasty_weight).round()
    combined = pd.concat([redraft, dynasty[~dynasty["Player Name"].isin(redraft["Player Name"])]], ignore_index=True)
    return combined[COLUMNS].drop_duplicates("Player Name")


def write_csv_atomic(frame, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            frame.to_csv(output, index=False)
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
