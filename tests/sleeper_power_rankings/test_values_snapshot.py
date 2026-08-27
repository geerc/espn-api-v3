import pandas as pd
import pytest

from sleeper_rankings import values


def test_correction_uses_saved_values_without_fetching(tmp_path, monkeypatch):
    path = tmp_path / "values.csv"
    pd.DataFrame([dict(name="A", position="QB", nfl_team="BUF", value=123)]).to_csv(path, index=False)
    original = path.read_bytes()
    monkeypatch.setattr(values, "fetch_values", lambda *a: pytest.fail("must not fetch"))
    assert values.weekly_values(path, overwrite=True).iloc[0]["value"] == 123
    assert path.read_bytes() == original


def test_correction_with_missing_snapshot_fails(tmp_path):
    with pytest.raises(ValueError, match="refusing to substitute"):
        values.weekly_values(tmp_path / "missing.csv", overwrite=True)
