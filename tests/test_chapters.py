from __future__ import annotations

from poptrivia.prep.chapters import Chapter, _parse_chapters, _parse_duration


def test_parse_duration_from_format_block() -> None:
    data = {"format": {"duration": "5637.84"}}
    assert _parse_duration(data) == 5_637_840


def test_parse_chapters_extracts_named_chapters() -> None:
    data = {
        "chapters": [
            {
                "start_time": "0.000",
                "end_time": "600.500",
                "tags": {"title": "Opening"},
            },
            {
                "start_time": "600.500",
                "end_time": "1800.000",
                "tags": {"title": "Climax"},
            },
        ]
    }
    out = _parse_chapters(data)
    assert len(out) == 2
    assert out[0] == Chapter(0, 600_500, "Opening")
    assert out[1].title == "Climax"
    assert out[1].duration_ms == 1_199_500


def test_parse_chapters_falls_back_to_default_title() -> None:
    data = {
        "chapters": [
            {"start_time": "0", "end_time": "100"},
            {"start_time": "100", "end_time": "200", "tags": {}},
        ]
    }
    out = _parse_chapters(data)
    assert [c.title for c in out] == ["Chapter 1", "Chapter 2"]


def test_parse_chapters_skips_malformed() -> None:
    data = {
        "chapters": [
            {"start_time": "0", "end_time": "100"},
            {"start_time": "broken"},  # malformed — should be skipped
            {"start_time": "100", "end_time": "200"},
        ]
    }
    out = _parse_chapters(data)
    assert len(out) == 2


def test_parse_chapters_empty_returns_empty_list() -> None:
    assert _parse_chapters({}) == []
    assert _parse_chapters({"chapters": []}) == []


def test_chapter_duration_property() -> None:
    assert Chapter(0, 600_000, "x").duration_ms == 600_000
    assert Chapter(100, 50, "negative").duration_ms == 0  # clamped to 0
