import pandas as pd

from summary import _fantasypros_url, _player_metadata


def test_fantasypros_fallback_uses_search_instead_of_a_guessed_slug():
    assert _fantasypros_url("Josh Allen") == "https://www.fantasypros.com/nfl/players/?q=Josh+Allen"


def test_player_metadata_prefers_an_explicit_canonical_url(tmp_path):
    names_path = tmp_path / "names.csv"
    values_path = tmp_path / "values.csv"
    canonical = "https://www.fantasypros.com/nfl/players/josh-allen-qb.php"
    pd.DataFrame([
        {"Name": "Josh Allen", "Team": "BUF", "URL": canonical},
        {"Name": "Christian McCaffrey", "Team": "SF", "URL": None},
    ]).to_csv(names_path, index=False)
    pd.DataFrame([
        {"Player Name": "Josh Allen", "Value": 100},
        {"Player Name": "Christian McCaffrey", "Value": 90},
    ]).to_csv(values_path, index=False)

    urls, _ = _player_metadata(names_path, values_path)

    assert urls["Josh Allen"] == canonical
    assert urls["Christian McCaffrey"] == "https://www.fantasypros.com/nfl/players/?q=Christian+McCaffrey"
