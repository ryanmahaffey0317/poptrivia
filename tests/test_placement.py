from __future__ import annotations

import pytest

from poptrivia.models import RawFactExtracted
from poptrivia.prep.chapters import Chapter, MovieMetadata
from poptrivia.prep.placement import (
    _allocate_largest_remainder,
    _enforce_min_spacing,
    _place_evenly,
    _place_in_chapters,
    place_facts,
)


def _fact(name: str, specificity: str = "high") -> RawFactExtracted:
    return RawFactExtracted(
        fact=name,
        source="imdb_trivia",
        specificity=specificity,  # type: ignore[arg-type]
        anchors=[],
        category="production",
    )


def _meta(*, duration_ms: int, chapters: list[Chapter] | None = None) -> MovieMetadata:
    return MovieMetadata(duration_ms=duration_ms, chapters=tuple(chapters or []))


# ─── even-spacing fallback ────────────────────────────────────────────────


def test_place_evenly_distributes_across_useful_range() -> None:
    # 100-minute movie (6_000_000 ms). 5% trim → useful range [300_000, 5_700_000].
    ts = _place_evenly(n_facts=4, duration_ms=6_000_000)
    assert len(ts) == 4
    # All inside the trimmed range
    assert all(300_000 <= t <= 5_700_000 for t in ts)
    # Strictly increasing
    assert ts == sorted(ts)


def test_place_evenly_zero_facts() -> None:
    assert _place_evenly(n_facts=0, duration_ms=6_000_000) == []


def test_place_evenly_short_movie_returns_empty() -> None:
    assert _place_evenly(n_facts=5, duration_ms=0) == []


# ─── chapter-based placement ──────────────────────────────────────────────


def test_allocate_largest_remainder_sums_to_n() -> None:
    chapters = (
        Chapter(0, 600_000, "A"),       # 10 min
        Chapter(600_000, 1_800_000, "B"),  # 20 min
        Chapter(1_800_000, 3_000_000, "C"),  # 20 min
    )
    out = _allocate_largest_remainder(10, chapters)
    assert sum(out) == 10
    # Chapter B and C are 2× the duration of A, so should get more facts.
    assert out[1] >= out[0] and out[2] >= out[0]


def test_place_in_chapters_facts_land_inside_chapters() -> None:
    chapters = (
        Chapter(0, 600_000, "A"),
        Chapter(600_000, 1_200_000, "B"),
    )
    ts = _place_in_chapters(n_facts=4, chapters=chapters)
    assert len(ts) == 4
    assert all(0 <= t < 1_200_000 for t in ts)
    # In sorted order
    assert ts == sorted(ts)


def test_place_in_chapters_zero_duration_chapter_skipped() -> None:
    chapters = (
        Chapter(0, 0, "Empty"),  # degenerate; nothing should land here
        Chapter(0, 600_000, "B"),
    )
    ts = _place_in_chapters(n_facts=2, chapters=chapters)
    assert all(t > 0 for t in ts)


# ─── min-spacing ──────────────────────────────────────────────────────────


def test_enforce_min_spacing_drops_close_neighbors() -> None:
    out = _enforce_min_spacing([0, 10_000, 100_000, 110_000, 300_000])
    # 0 kept; 10_000 too close to 0; 100_000 kept; 110_000 too close;
    # 300_000 kept.
    assert out == [0, 100_000, 300_000]


def test_enforce_min_spacing_empty() -> None:
    assert _enforce_min_spacing([]) == []


# ─── end-to-end place_facts ───────────────────────────────────────────────


def test_place_facts_produces_cards_in_timestamp_order() -> None:
    facts = [_fact(f"fact_{i}") for i in range(5)]
    metadata = _meta(duration_ms=6_000_000)
    cards = place_facts(facts=facts, metadata=metadata, max_cards=70)
    assert len(cards) == 5
    timestamps = [c.timestamp_ms for c in cards]
    assert timestamps == sorted(timestamps)
    assert all(c.id.startswith("card_") for c in cards)


def test_place_facts_respects_max_cards_cap() -> None:
    facts = [_fact(f"fact_{i}") for i in range(100)]
    metadata = _meta(duration_ms=12_000_000)
    cards = place_facts(facts=facts, metadata=metadata, max_cards=20)
    assert len(cards) <= 20


def test_place_facts_with_chapters_evidence_mentions_chapters() -> None:
    facts = [_fact("a"), _fact("b")]
    metadata = _meta(
        duration_ms=1_200_000,
        chapters=[
            Chapter(0, 600_000, "Opening"),
            Chapter(600_000, 1_200_000, "Climax"),
        ],
    )
    cards = place_facts(facts=facts, metadata=metadata, max_cards=70)
    assert all("chapters" in (c.anchor_evidence or "").lower() for c in cards)


def test_place_facts_without_chapters_evidence_mentions_even_spacing() -> None:
    facts = [_fact("a"), _fact("b")]
    metadata = _meta(duration_ms=1_200_000)
    cards = place_facts(facts=facts, metadata=metadata, max_cards=70)
    assert all("even spacing" in (c.anchor_evidence or "").lower() for c in cards)


def test_place_facts_empty_input() -> None:
    metadata = _meta(duration_ms=6_000_000)
    assert place_facts(facts=[], metadata=metadata, max_cards=70) == []


def test_place_facts_specificity_to_interest_mapping() -> None:
    facts = [
        _fact("low", specificity="low"),
        _fact("med", specificity="medium"),
        _fact("high", specificity="high"),
    ]
    metadata = _meta(duration_ms=6_000_000)
    cards = place_facts(facts=facts, metadata=metadata, max_cards=70)
    # Same order as facts in -> cards out (since timestamps go ascending and
    # we placed them in order).
    by_text = {c.text: c.interest_level for c in cards}
    assert by_text["low"] == 2
    assert by_text["med"] == 3
    assert by_text["high"] == 4
