from __future__ import annotations

import pytest

from poptrivia.models import (
    Movie,
    MovieStatus,
    TautulliEvent,
    TriviaCard,
)


def test_tautulli_event_brief_template() -> None:
    raw = {
        "event": "play",
        "username": "alice",
        "media_type": "movie",
        "title": "The Shining",
        "year": "1980",
        "imdb_id": "tt0081505",
        "tmdb_id": "694",
        "plex_guid": "plex://movie/abc",
        "file": "/media/movies/The Shining.mkv",
        "duration_ms": 8460000,
        "session_key": "42",
        "view_offset_ms": 0,
    }
    ev = TautulliEvent.from_raw(raw)
    assert ev.username == "alice"
    assert ev.year == 1980  # coerced from str
    assert ev.duration_ms == 8460000
    assert ev.session_key == "42"
    assert ev.view_offset_ms == 0


def test_tautulli_event_session_key_int_to_str() -> None:
    raw = {
        "event": "play",
        "username": "alice",
        "media_type": "movie",
        "title": "x",
        "plex_guid": "plex://movie/x",
        "session_key": 99,
    }
    ev = TautulliEvent.from_raw(raw)
    assert ev.session_key == "99"


def test_tautulli_event_camelcase_variant() -> None:
    raw = {
        "action": "play",
        "user": "alice",
        "mediaType": "movie",
        "title": "x",
        "guid": "plex://movie/x",
        "sessionKey": "7",
        "viewOffset": 12345,
        "imdbId": "tt0000001",
    }
    ev = TautulliEvent.from_raw(raw)
    assert ev.event == "play"
    assert ev.username == "alice"
    assert ev.media_type == "movie"
    assert ev.plex_guid == "plex://movie/x"
    assert ev.session_key == "7"
    assert ev.view_offset_ms == 12345
    assert ev.imdb_id == "tt0000001"


def test_tautulli_event_empty_year_becomes_none() -> None:
    raw = {
        "event": "play",
        "username": "alice",
        "media_type": "movie",
        "title": "x",
        "plex_guid": "plex://movie/x",
        "session_key": "1",
        "year": "",
    }
    ev = TautulliEvent.from_raw(raw)
    assert ev.year is None


def test_movie_status_enum_roundtrip() -> None:
    m = Movie(plex_guid="g", title="t", status=MovieStatus.READY)
    assert m.status is MovieStatus.READY
    assert m.status.value == "ready"


def test_trivia_card_interest_level_bounds() -> None:
    TriviaCard(
        id="c1",
        timestamp_ms=10000,
        text="x",
        category="production",
        interest_level=3,
        source_fact_id="f1",
    )
    with pytest.raises(Exception):
        TriviaCard(
            id="c2",
            timestamp_ms=10000,
            text="x",
            category="production",
            interest_level=6,
            source_fact_id="f1",
        )
