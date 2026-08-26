from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://api.sleeper.app/v1"


class SleeperAPIError(RuntimeError):
    pass


@dataclass
class SleeperClient:
    timeout: tuple[int, int] = (5, 30)

    def __post_init__(self) -> None:
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers["User-Agent"] = "sleeper-power-rankings/0.1"

    def get(self, path: str) -> Any:
        response = self.session.get(f"{BASE_URL}{path}", timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            raise SleeperAPIError(f"Sleeper request failed for {path}: {error}") from error
        return response.json()

    def league(self, league_id: str) -> dict:
        return self.get(f"/league/{league_id}")

    def rosters(self, league_id: str) -> list[dict]:
        return self.get(f"/league/{league_id}/rosters")

    def users(self, league_id: str) -> list[dict]:
        return self.get(f"/league/{league_id}/users")

    def matchups(self, league_id: str, week: int) -> list[dict]:
        return self.get(f"/league/{league_id}/matchups/{week}")

    def players(self) -> dict[str, dict]:
        return self.get("/players/nfl")

