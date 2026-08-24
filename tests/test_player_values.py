import pandas as pd
import pytest
import requests

from player_values import merge_values, scrape_rankings, write_csv_atomic


class FakeResponse:
    def __init__(self, html, status=200):
        self.content = html.encode()
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        return next(self.responses)


def test_scraper_parses_valid_rows_once_per_page():
    html = """<div class='onePlayer'><div class='player-name'><a>Josh Allen</a><span class='player-team'>BUF</span></div><p class='position'>QB1</p><div class='value'>$9,999</div></div>"""
    session = FakeSession([FakeResponse(html)])

    result = scrape_rankings(session, "https://example.test/{}", 1, "sample")

    assert result.to_dict("records") == [{"Player Name": "Josh Allen", "Pos": "QB1", "Value": 9999, "NFL_Team": "BUF"}]
    assert session.calls == [("https://example.test/0", (5, 30))]


def test_scraper_rejects_malformed_or_empty_html():
    with pytest.raises(ValueError, match="returned no players"):
        scrape_rankings(FakeSession([FakeResponse("<html>blocked</html>")]), "https://example.test/{}", 1, "sample")


def test_scraper_propagates_http_errors():
    with pytest.raises(requests.HTTPError):
        scrape_rankings(FakeSession([FakeResponse("error", status=503)]), "https://example.test/{}", 1, "sample")


def test_merge_values_prefers_redraft_and_adds_dynasty_only_players():
    redraft = pd.DataFrame([{"Player Name": "A", "Pos": "RB", "Value": 100, "NFL_Team": "BUF"}])
    dynasty = pd.DataFrame([
        {"Player Name": "A", "Pos": "RB", "Value": 500, "NFL_Team": "BUF"},
        {"Player Name": "B", "Pos": "WR", "Value": 200, "NFL_Team": "KC"},
    ])

    result = merge_values(redraft, dynasty, dynasty_weight=0.5).set_index("Player Name")

    assert result.loc["A", "Value"] == 100
    assert result.loc["B", "Value"] == 100


def test_atomic_csv_write_replaces_destination(tmp_path):
    destination = tmp_path / "nested" / "values.csv"
    frame = pd.DataFrame([{"Player Name": "A", "Pos": "RB", "Value": 100, "NFL_Team": "BUF"}])

    write_csv_atomic(frame, destination)

    assert pd.read_csv(destination).to_dict("records") == frame.to_dict("records")
    assert list(destination.parent.glob(".values.csv.*")) == []
