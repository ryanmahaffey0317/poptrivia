from __future__ import annotations

from pathlib import Path

from poptrivia.models import RawFactExtracted, TriviaCard
from poptrivia.prep.llm.stage2_align import (
    _assign_facts_to_windows,
    _build_fact_index,
    _cap_and_renumber,
    _dedupe_by_fact,
    _format_ts,
    _split_subtitle_windows,
    write_track_file,
)
from poptrivia.prep.subtitles import SubtitleEntry


def _card(
    cid: str,
    ts: int,
    fact_id: str = "f001",
    interest: int = 3,
) -> TriviaCard:
    return TriviaCard(
        id=cid,
        timestamp_ms=ts,
        text="x",
        category="production",
        interest_level=interest,
        source_fact_id=fact_id,
    )


def test_dedupe_keeps_highest_interest_per_fact() -> None:
    cards = [
        _card("a", 1000, fact_id="f1", interest=2),
        _card("b", 2000, fact_id="f1", interest=4),
        _card("c", 3000, fact_id="f2", interest=3),
    ]
    out = _dedupe_by_fact(cards)
    assert len(out) == 2
    f1 = next(c for c in out if c.source_fact_id == "f1")
    assert f1.interest_level == 4


def test_cap_and_renumber_drops_lowest_and_orders_by_time() -> None:
    cards = [
        _card("a", 30_000, fact_id="f1", interest=2),
        _card("b", 10_000, fact_id="f2", interest=5),
        _card("c", 20_000, fact_id="f3", interest=4),
        _card("d", 5_000, fact_id="f4", interest=1),
    ]
    out = _cap_and_renumber(cards, max_cards=2)
    assert len(out) == 2
    assert [c.id for c in out] == ["card_001", "card_002"]
    assert [c.source_fact_id for c in out] == ["f2", "f3"]
    assert out[0].timestamp_ms < out[1].timestamp_ms


def test_split_subtitle_windows_with_duration() -> None:
    subs = [
        SubtitleEntry(start_ms=0, end_ms=2000, text="a"),         # 0:00
        SubtitleEntry(start_ms=910_000, end_ms=912_000, text="b"),  # 15:10
        SubtitleEntry(start_ms=1_810_000, end_ms=1_812_000, text="c"),  # 30:10
    ]
    windows = _split_subtitle_windows(subs, movie_duration_ms=2_000_000)  # 33:20
    # 0-15m, 15-30m, 30-33:20
    assert len(windows) == 3
    assert [w[0] for w in windows] == [0, 900_000, 1_800_000]
    assert windows[0][2][0].text == "a"
    assert windows[1][2][0].text == "b"
    assert windows[2][2][0].text == "c"


def test_split_subtitle_windows_no_duration_uses_last_sub() -> None:
    subs = [SubtitleEntry(start_ms=0, end_ms=2000, text="a")]
    windows = _split_subtitle_windows(subs, movie_duration_ms=None)
    assert len(windows) == 1
    assert windows[0][1] == 2000  # end = last subtitle end


def test_build_fact_index_assigns_sequential_ids() -> None:
    facts = [
        RawFactExtracted(
            fact="a", source="imdb_trivia", specificity="high",
            anchors=[], category="production",
        ),
        RawFactExtracted(
            fact="b", source="wikipedia", specificity="medium",
            anchors=[], category="casting",
        ),
    ]
    idx = _build_fact_index(facts)
    assert list(idx.keys()) == ["f001", "f002"]


def test_format_ts() -> None:
    assert _format_ts(0) == "00:00"
    assert _format_ts(90_500) == "01:30"
    assert _format_ts(3_600_000) == "60:00"


def _raw_fact(
    text: str,
    *,
    anchors: list[str] | None = None,
    specificity: str = "high",
    category: str = "production",
) -> RawFactExtracted:
    return RawFactExtracted(
        fact=text,
        source="imdb_trivia",
        specificity=specificity,  # type: ignore[arg-type]
        anchors=anchors or [],
        category=category,  # type: ignore[arg-type]
    )


def test_assign_facts_anchor_match_goes_only_to_matching_windows() -> None:
    """An anchored fact never lands in a window where the anchor doesn't
    appear, regardless of load."""
    facts = [
        _raw_fact("Bat fact", anchors=["baseball bat"]),
    ]
    windows = [
        (0, 900_000, [SubtitleEntry(0, 5000, "Hello world")]),
        (900_000, 1_800_000, [SubtitleEntry(0, 5000, "She picks up the baseball bat.")]),
        (1_800_000, 2_700_000, [SubtitleEntry(0, 5000, "Generic dialogue")]),
    ]
    assigned = _assign_facts_to_windows(facts, windows)
    assert assigned[0] == []
    assert len(assigned[1]) == 1
    assert assigned[2] == []


def test_assign_facts_balances_when_anchor_matches_many_windows() -> None:
    """The common-anchor problem (character names appear everywhere):
    facts must spread, not pile up in window 0."""
    facts = [
        _raw_fact(f"Annie fact {i}", anchors=["Annie"])
        for i in range(6)
    ]
    windows = [
        (i * 900_000, (i + 1) * 900_000, [SubtitleEntry(0, 5000, "Annie says hi.")])
        for i in range(3)
    ]
    assigned = _assign_facts_to_windows(facts, windows)
    counts = [len(w) for w in assigned]
    # 6 facts / 3 matching windows -> exactly 2 each, not 6/0/0.
    assert counts == [2, 2, 2], counts


def test_assign_facts_anchorless_round_robin_distributes_across_windows() -> None:
    """Anchor-less facts must spread across all windows."""
    facts = [
        _raw_fact(f"Fact {i}", anchors=[], specificity="medium")
        for i in range(9)
    ]
    windows = [
        (i * 900_000, (i + 1) * 900_000, [SubtitleEntry(0, 1, "dialogue")])
        for i in range(3)
    ]
    assigned = _assign_facts_to_windows(facts, windows)
    counts = [len(w) for w in assigned]
    # 9 facts / 3 windows -> exactly 3 each
    assert counts == [3, 3, 3]


def test_assign_facts_unmatched_anchor_falls_through_to_round_robin() -> None:
    """A fact with anchors that don't appear anywhere should still be placed."""
    facts = [
        _raw_fact("Wedding fact", anchors=["nonexistent wedding"]),
    ]
    windows = [
        (0, 900_000, [SubtitleEntry(0, 1, "morning coffee")]),
        (900_000, 1_800_000, [SubtitleEntry(0, 1, "office scene")]),
    ]
    assigned = _assign_facts_to_windows(facts, windows)
    # Round-robin starts at index 0 for the leftover.
    assert sum(len(w) for w in assigned) == 1
    assert len(assigned[0]) == 1


def test_assign_facts_anchorless_high_specificity_distributed_first() -> None:
    """High-specificity anchor-less facts spread across windows before
    low-specificity ones fill remaining slots."""
    facts = [
        _raw_fact("Low 1", specificity="low"),
        _raw_fact("Low 2", specificity="low"),
        _raw_fact("High A", specificity="high"),
        _raw_fact("High B", specificity="high"),
    ]
    windows = [
        (0, 900_000, [SubtitleEntry(0, 1, "x")]),
        (900_000, 1_800_000, [SubtitleEntry(0, 1, "y")]),
    ]
    assigned = _assign_facts_to_windows(facts, windows)
    # First two slots get high-specificity (one per window), then lows.
    assert {assigned[0][0].fact, assigned[1][0].fact} == {"High A", "High B"}
    assert {assigned[0][1].fact, assigned[1][1].fact} == {"Low 1", "Low 2"}


def test_assign_facts_handles_empty_windows_list() -> None:
    assert _assign_facts_to_windows([_raw_fact("x")], []) == []


def test_write_track_file_roundtrip(tmp_path: Path) -> None:
    cards = [_card("card_001", 1000), _card("card_002", 2000)]
    out = write_track_file(
        tracks_dir=tmp_path,
        plex_guid="plex://movie/abc",
        imdb_id="tt0081505",
        tmdb_id="694",
        title="The Shining",
        year=1980,
        model="qwen2.5:32b",
        cards=cards,
    )
    assert out.exists()
    assert out.name == "plex___movie_abc.json"
    text = out.read_text()
    assert '"title": "The Shining"' in text
    assert '"cards": [' in text
