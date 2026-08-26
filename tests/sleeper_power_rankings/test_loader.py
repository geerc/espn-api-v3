from sleeper_rankings.loader import completed_week, team_name
from sleeper_rankings.render import render_preseason


def test_completed_week_defaults_to_zero():
    assert completed_week({"settings": {}}) == 0


def test_team_name_falls_back_to_display_name():
    assert team_name({"display_name": "Chris", "metadata": {}}, 1) == "Chris"


def test_preseason_page_can_be_deployed(tmp_path):
    output = render_preseason(output=tmp_path, title="SYPIP Power Rankings", league_name="SYPIP", season="2026")
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "Something is coming" in content
    assert "Power Rankings" not in content
