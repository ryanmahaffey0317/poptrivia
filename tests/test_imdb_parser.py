from __future__ import annotations

from pathlib import Path

from poptrivia.prep.sources.imdb import _parse_items

FIXTURES = Path(__file__).parent / "fixtures" / "imdb"


def test_modern_layout_parsing() -> None:
    html = (FIXTURES / "modern.html").read_text()
    items = _parse_items(html, "imdb_trivia")
    assert len(items) == 3
    assert items[0].source == "imdb_trivia"
    assert items[0].text.startswith("Stanley Kubrick")
    # The "X of Y found this interesting" tail must be stripped.
    assert "found this interesting" not in items[0].text
    assert items[2].text.startswith("Jack Nicholson improvised")


def test_legacy_layout_fallback() -> None:
    html = (FIXTURES / "legacy.html").read_text()
    items = _parse_items(html, "imdb_goofs")
    assert len(items) == 2
    assert items[0].source == "imdb_goofs"
    assert "Steadicam" in items[0].text


def test_empty_page_returns_empty_list() -> None:
    items = _parse_items("<html><body></body></html>", "imdb_trivia")
    assert items == []
