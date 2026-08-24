from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    league_id: int
    swid: Optional[str]
    espn_s2: Optional[str]
    openai_api_key: Optional[str]
    year: int
    week: Optional[int]
    player_values_dir: Path
    names_file: Path
    report_root: Path
    simulations: int = 100_000
    random_seed: Optional[int] = None
    ai_model: str = "gpt-4o"

    @classmethod
    def from_env(
        cls,
        *,
        year: Optional[int] = None,
        week: Optional[int] = None,
        simulations: int = 100_000,
        random_seed: Optional[int] = None,
    ) -> "AppConfig":
        load_dotenv(SCRIPT_DIR / ".env")
        raw_league_id = os.getenv("league_id")
        if not raw_league_id:
            raise ValueError("Missing required league_id in espn-api-v3/.env")

        report_root = Path(
            os.getenv(
                "REPORT_ROOT",
                SCRIPT_DIR.parent / "jtown-dynasty" / "content" / "power_rankings",
            )
        ).expanduser()

        return cls(
            league_id=int(raw_league_id),
            swid=os.getenv("swid"),
            espn_s2=os.getenv("espn_s2"),
            openai_api_key=os.getenv("OPEN_AI_KEY"),
            year=year or datetime.now().year,
            week=week,
            player_values_dir=SCRIPT_DIR / "player_values",
            names_file=SCRIPT_DIR / "fantasy_pros_names.csv",
            report_root=report_root,
            simulations=simulations,
            random_seed=random_seed,
            ai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        )
