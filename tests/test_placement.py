from __future__ import annotations

import pytest

from poptrivia.models import RawFactExtracted
from poptrivia.prep.chapters import Chapter, MovieMetadata
from poptrivia.prep.placement import (
    _allocate_largest_remainder,
    _enforce_min_spacing,
    _exclude_credits_chapters,
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
    facts = [_fact(f"f{i}") for i in range(4)]
    pairs, matched = _place_in_chapters(facts, chapters)
    assert len(pairs) == 4
    timestamps = [ts for ts, _f in pairs]
    assert all(0 <= t < 1_200_000 for t in timestamps)
    assert timestamps == sorted(timestamps)
    # No anchors on these facts → matched count is 0
    assert matched == 0


def test_place_in_chapters_zero_duration_chapter_skipped() -> None:
    chapters = (
        Chapter(0, 0, "Empty"),  # degenerate; nothing should land here
        Chapter(0, 600_000, "B"),
    )
    facts = [_fact(f"f{i}") for i in range(2)]
    pairs, _matched = _place_in_chapters(facts, chapters)
    assert all(ts > 0 for ts, _f in pairs)


# ─── anchor → chapter-title matching ─────────────────────────────────────


def _afact(text: str, anchors: list[str], specificity: str = "high"):
    return RawFactExtracted(
        fact=text,
        source="imdb_trivia",
        specificity=specificity,  # type: ignore[arg-type]
        anchors=anchors,
        category="production",
    )


def test_anchor_matched_fact_routes_to_matching_chapter() -> None:
    """A fact whose anchor appears in a chapter title should be placed
    inside that chapter, not somewhere arbitrary."""
    chapters = (
        Chapter(0, 600_000, "The Setup"),
        Chapter(600_000, 1_200_000, "Wedding Reception"),
        Chapter(1_200_000, 1_800_000, "The Aftermath"),
    )
    facts = [_afact("Wilson Phillips played the wedding", ["wedding"])]
    pairs, matched = _place_in_chapters(facts, chapters)
    assert matched == 1
    ts, _fact = pairs[0]
    # Must be inside the Wedding chapter
    assert 600_000 <= ts < 1_200_000


def test_anchor_matched_load_balances_across_matching_chapters() -> None:
    """If anchor matches multiple chapters, facts spread across them
    rather than piling into the first one."""
    chapters = (
        Chapter(0, 600_000, "Annie at the Coffee Shop"),
        Chapter(600_000, 1_200_000, "Annie at the Bakery"),
        Chapter(1_200_000, 1_800_000, "Annie at the Wedding"),
    )
    facts = [_afact(f"fact_{i}", ["Annie"]) for i in range(6)]
    pairs, matched = _place_in_chapters(facts, chapters)
    assert matched == 6
    # 2 per chapter — load-balanced.
    counts_per_chapter = [
        sum(1 for ts, _f in pairs if c.start_ms <= ts < c.end_ms)
        for c in chapters
    ]
    assert counts_per_chapter == [2, 2, 2]


def test_unmatched_anchor_falls_through_to_round_robin() -> None:
    """A fact with anchors but no matching chapter title should still
    be placed somewhere, via the round-robin leftover path."""
    chapters = (
        Chapter(0, 600_000, "Chapter A"),
        Chapter(600_000, 1_200_000, "Chapter B"),
    )
    facts = [_afact("x", ["something not in any chapter title"])]
    pairs, matched = _place_in_chapters(facts, chapters)
    assert len(pairs) == 1     # placed
    assert matched == 0        # not anchor-matched


def test_mixed_anchored_and_anchorless_facts() -> None:
    """Anchored facts route via title match; anchorless / unmatched facts
    spread across remaining capacity."""
    chapters = (
        Chapter(0, 600_000, "Opening"),
        Chapter(600_000, 1_200_000, "Wedding"),
        Chapter(1_200_000, 1_800_000, "Finale"),
    )
    facts = [
        _afact("Wedding venue fact", ["wedding"]),        # → chapter 1
        _afact("Production fact A", []),                  # → leftover
        _afact("Production fact B", []),                  # → leftover
        _afact("Production fact C", []),                  # → leftover
    ]
    pairs, matched = _place_in_chapters(facts, chapters)
    assert len(pairs) == 4
    assert matched == 1
    # The wedding fact should be inside the wedding chapter.
    wedding_pair = next(p for p in pairs if p[1].fact == "Wedding venue fact")
    assert 600_000 <= wedding_pair[0] < 1_200_000


def test_short_anchor_ignored() -> None:
    """Anchors shorter than 3 chars don't trigger matching (avoids
    false positives on common words like 'is', 'on', 'to')."""
    chapters = (
        Chapter(0, 600_000, "Is on To"),  # title with short words
        Chapter(600_000, 1_200_000, "Real Scene"),
    )
    facts = [_afact("x", ["is", "on"])]
    _pairs, matched = _place_in_chapters(facts, chapters)
    assert matched == 0  # Short anchors don't count


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


# ─── credits-chapter filter ──────────────────────────────────────────────


def test_exclude_credits_chapters_by_title() -> None:
    duration = 7_200_000  # 2 hours
    chapters = (
        Chapter(0, 120_000, "Opening Credits"),
        Chapter(120_000, 1_200_000, "Act One"),
        Chapter(1_200_000, 6_900_000, "Main Story"),
        Chapter(6_900_000, 7_200_000, "End Titles"),
    )
    out = _exclude_credits_chapters(chapters, duration)
    titles = [c.title for c in out]
    assert "Act One" in titles
    assert "Main Story" in titles
    assert "Opening Credits" not in titles
    assert "End Titles" not in titles


def test_exclude_credits_chapters_various_title_patterns() -> None:
    duration = 7_200_000
    chapters = (
        Chapter(0, 100_000, "Main Title Sequence"),
        Chapter(100_000, 200_000, "Logo"),  # studio logo
        Chapter(200_000, 300_000, "Closing Credits"),
        Chapter(300_000, 400_000, "Final Credits"),
        Chapter(400_000, 500_000, "Titles"),
        Chapter(500_000, 600_000, "Some Real Chapter"),
    )
    # Bias the duration so position-based exclusion isn't the trigger.
    out = _exclude_credits_chapters(chapters, duration)
    assert len(out) == 1
    assert out[0].title == "Some Real Chapter"


def test_exclude_credits_chapters_by_position() -> None:
    """A chapter entirely inside the first 5% of runtime is excluded,
    even if its title doesn't match the credits regex."""
    duration = 7_200_000  # 2h -> 5% = 360_000ms
    chapters = (
        Chapter(0, 240_000, "Chapter 1"),  # entirely in first 5% — excluded
        Chapter(240_000, 1_200_000, "Chapter 2"),  # spans into useful range — kept
        Chapter(6_900_000, 7_200_000, "Chapter 9"),  # entirely in last 5% — excluded
        Chapter(6_500_000, 7_000_000, "Chapter 8"),  # spans out of edge — kept
    )
    out = _exclude_credits_chapters(chapters, duration)
    titles = [c.title for c in out]
    assert "Chapter 1" not in titles
    assert "Chapter 9" not in titles
    assert "Chapter 2" in titles
    assert "Chapter 8" in titles


def test_exclude_credits_chapters_keeps_normal_chapters() -> None:
    duration = 7_200_000
    chapters = (
        Chapter(0, 1_800_000, "Act One"),
        Chapter(1_800_000, 3_600_000, "Act Two"),
        Chapter(3_600_000, 7_200_000, "Act Three"),
    )
    out = _exclude_credits_chapters(chapters, duration)
    assert len(out) == 3


def test_exclude_credits_chapters_empty_input() -> None:
    assert _exclude_credits_chapters((), 7_200_000) == ()


def test_place_facts_falls_back_to_even_spacing_when_all_chapters_excluded() -> None:
    """If every chapter looks like credits, even-spacing kicks in instead
    of returning zero cards."""
    facts = [_fact(f"fact_{i}") for i in range(3)]
    metadata = MovieMetadata(
        duration_ms=600_000,
        chapters=(
            Chapter(0, 200_000, "Opening Credits"),
            Chapter(200_000, 600_000, "End Titles"),
        ),
    )
    cards = place_facts(facts=facts, metadata=metadata, max_cards=70)
    # Should still get cards via the even-spacing fallback.
    assert len(cards) == 3
    assert all("even spacing" in (c.anchor_evidence or "").lower() for c in cards)


def test_place_facts_with_mix_of_credits_and_real_chapters() -> None:
    """Credits chapters excluded, facts distributed across the rest."""
    facts = [_fact(f"fact_{i}") for i in range(4)]
    metadata = MovieMetadata(
        duration_ms=7_200_000,
        chapters=(
            Chapter(0, 120_000, "Opening Credits"),
            Chapter(120_000, 3_600_000, "Act One"),
            Chapter(3_600_000, 6_900_000, "Act Two"),
            Chapter(6_900_000, 7_200_000, "End Titles"),
        ),
    )
    cards = place_facts(facts=facts, metadata=metadata, max_cards=70)
    # No card should be inside the credits chapters.
    for c in cards:
        assert not (0 <= c.timestamp_ms < 120_000), (
            f"card landed in opening credits: {c.timestamp_ms}ms"
        )
        assert not (6_900_000 <= c.timestamp_ms < 7_200_000), (
            f"card landed in end titles: {c.timestamp_ms}ms"
        )
    # Anchor evidence mentions credit-skip metadata
    assert any("skipped 2 credits" in (c.anchor_evidence or "") for c in cards)


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
