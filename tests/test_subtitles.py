from __future__ import annotations

from poptrivia.prep.subtitles import (
    SubtitleEntry,
    _parse_srt_text,
    windowize,
)

SRT = """1
00:00:01,000 --> 00:00:03,500
Hello, world.

2
00:00:04,000 --> 00:00:06,000
[ARGUMENT]
Wendy: This is fine.

3
00:00:50,000 --> 00:00:52,000
Forty seconds later.

4
00:00:55,000 --> 00:00:57,000
And again.
"""


def test_parse_srt_strips_brackets_and_keeps_dialogue() -> None:
    entries = _parse_srt_text(SRT)
    assert len(entries) == 4
    assert entries[0].start_ms == 1000
    assert entries[0].end_ms == 3500
    assert entries[0].text == "Hello, world."
    # [ARGUMENT] should be stripped, "Wendy: This is fine." kept.
    assert "ARGUMENT" not in entries[1].text
    assert "This is fine." in entries[1].text


def test_windowize_groups_close_entries() -> None:
    entries = [
        SubtitleEntry(start_ms=0, end_ms=2000, text="a"),
        SubtitleEntry(start_ms=5000, end_ms=7000, text="b"),
        SubtitleEntry(start_ms=10_000, end_ms=12_000, text="c"),
        # 50s later — new window
        SubtitleEntry(start_ms=60_000, end_ms=62_000, text="d"),
    ]
    windows = windowize(entries, window_seconds=30)
    assert len(windows) == 2
    assert windows[0].text == "a b c"
    assert windows[0].start_ms == 0
    assert windows[1].text == "d"
    assert windows[1].start_ms == 60_000


def test_windowize_empty_input() -> None:
    assert windowize([], window_seconds=30) == []
