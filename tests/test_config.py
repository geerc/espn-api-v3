from config import AppConfig


def test_config_validates_league_id(monkeypatch):
    monkeypatch.delenv("league_id", raising=False)
    monkeypatch.setattr("config.load_dotenv", lambda *args, **kwargs: None)

    try:
        AppConfig.from_env()
    except ValueError as error:
        assert "league_id" in str(error)
    else:
        raise AssertionError("Expected missing league_id to fail")


def test_config_accepts_runtime_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("league_id", "123")
    monkeypatch.setenv("REPORT_ROOT", str(tmp_path))
    monkeypatch.setattr("config.load_dotenv", lambda *args, **kwargs: None)

    config = AppConfig.from_env(year=2024, week=7, simulations=25, random_seed=9)

    assert config.league_id == 123
    assert config.year == 2024
    assert config.week == 7
    assert config.simulations == 25
    assert config.random_seed == 9
    assert config.report_root == tmp_path
