from __future__ import annotations

import logging
import re

from poptrivia.models import RawFactExtracted, TriviaCard
from poptrivia.prep.chapters import Chapter, MovieMetadata

log = logging.getLogger("poptrivia.prep.placement")


# Map RawFactExtracted.specificity → TriviaCard.interest_level (1-5 scale).
_SPECIFICITY_TO_INTEREST = {"high": 4, "medium": 3, "low": 2}

# Minimum gap between consecutive cards. Stage 2's old LLM-based prompt
# tried to enforce this; algorithmic placement enforces it directly.
_MIN_CARD_SPACING_MS = 90_000

# Skip the very start and end of the runtime when placing facts — credits,
# opening logos, etc. 5% on each side trims ~6 minutes off a 2-hour film.
_RUNTIME_EDGE_TRIM = 0.05

# Chapter titles that almost certainly mark credits / no-trivia regions.
# Matches "Opening Credits", "End Titles", "Main Title Sequence", etc.
# Case-insensitive; ignores trailing punctuation / numbering.
_CREDITS_TITLE_RE = re.compile(
    r"^\s*(opening|end|closing|main|final|start)?\s*"
    r"(title\s*sequence|titles?|credits?|logos?)\b",
    re.IGNORECASE,
)


def place_facts(
    *,
    facts: list[RawFactExtracted],
    metadata: MovieMetadata,
    max_cards: int,
) -> list[TriviaCard]:
    """Distribute facts across the movie's timeline.

    Chapters present: allocate facts to each chapter proportionally to
    chapter duration, then evenly space within each chapter.

    No chapters: even-space across (5%-95% of) the full runtime.

    Order in the output is by timestamp ascending. Fact ids are assigned
    sequentially as fNNN; card ids as card_NNN.
    """
    if not facts:
        return []
    if metadata.duration_ms <= 0:
        log.warning("Movie duration is non-positive; no placement possible")
        return []

    facts = facts[:max_cards]

    placeable_chapters = (
        _exclude_credits_chapters(metadata.chapters, metadata.duration_ms)
        if metadata.chapters
        else ()
    )
    if metadata.chapters and not placeable_chapters:
        log.info(
            "All %d chapters look like credits / edge-trim — falling back "
            "to even spacing",
            len(metadata.chapters),
        )

    if placeable_chapters:
        skipped = len(metadata.chapters) - len(placeable_chapters)
        placed_pairs, matched_count = _place_in_chapters(facts, placeable_chapters)
        timestamps = [ts for ts, _f in placed_pairs]
        facts = [f for _ts, f in placed_pairs]
        scheme = (
            f"{len(placeable_chapters)} chapters"
            + (f", {matched_count}/{len(facts)} anchor-matched"
               if matched_count else "")
            + (f", skipped {skipped} credits" if skipped else "")
        )
    else:
        timestamps = _place_evenly(len(facts), metadata.duration_ms)
        scheme = "even spacing"

    # Defensive spacing pass: facts pulled into the same minute by a small
    # chapter still need 90s minimum gap. If a collision forces fewer
    # timestamps than facts, trim the lowest-priority facts.
    timestamps = _enforce_min_spacing(timestamps)
    if len(timestamps) < len(facts):
        log.info(
            "Spacing constraints dropped %d fact(s) from the placement",
            len(facts) - len(timestamps),
        )
        facts = facts[: len(timestamps)]

    cards: list[TriviaCard] = []
    for i, (fact, ts) in enumerate(zip(facts, timestamps)):
        cards.append(
            TriviaCard(
                id=f"card_{i + 1:03d}",
                timestamp_ms=ts,
                text=fact.fact,
                category=fact.category,
                interest_level=_SPECIFICITY_TO_INTEREST.get(fact.specificity, 3),
                source_fact_id=f"f{i + 1:03d}",
                anchor_evidence=f"Algorithmic placement #{i + 1}/{len(facts)} ({scheme}).",
            )
        )

    log.info(
        "Placed %d card(s) across %dms using %s",
        len(cards),
        metadata.duration_ms,
        scheme,
    )
    return cards


# ─── credits-chapter filter ──────────────────────────────────────────────


def _exclude_credits_chapters(
    chapters: tuple[Chapter, ...], duration_ms: int
) -> tuple[Chapter, ...]:
    """Drop chapters that almost certainly cover credits / logos / no-trivia.

    Two rules, each independent:
      1. Title-based: matches _CREDITS_TITLE_RE (e.g. "Opening Credits",
         "End Titles", "Main Titles").
      2. Position-based: chapter is entirely inside the first or last
         _RUNTIME_EDGE_TRIM fraction of the runtime. Catches credit
         chapters that don't follow predictable naming.
    """
    edge_lo = int(duration_ms * _RUNTIME_EDGE_TRIM)
    edge_hi = int(duration_ms * (1.0 - _RUNTIME_EDGE_TRIM))

    out: list[Chapter] = []
    for c in chapters:
        if _CREDITS_TITLE_RE.match(c.title):
            log.info("Excluding credits-titled chapter %r", c.title)
            continue
        if c.end_ms <= edge_lo or c.start_ms >= edge_hi:
            log.info(
                "Excluding edge-position chapter %r (%dms-%dms outside "
                "5%%-95%% useful range %dms-%dms)",
                c.title,
                c.start_ms,
                c.end_ms,
                edge_lo,
                edge_hi,
            )
            continue
        out.append(c)
    return tuple(out)


# ─── chapter-based placement ──────────────────────────────────────────────


def _place_in_chapters(
    facts: list,
    chapters: tuple[Chapter, ...],
) -> tuple[list[tuple[int, object]], int]:
    """Assign each fact to one chapter and choose a timestamp within it.

    Returns (placed_pairs, anchor_match_count).
    placed_pairs is a list of (timestamp_ms, fact) tuples sorted by timestamp.

    Assignment is anchor-aware: if any of a fact's anchors appears (case-
    insensitive substring) in a chapter title, route the fact there. Common
    anchors that match multiple chapters go to the least-loaded one. Facts
    with no anchor-title match (or no anchors at all) round-robin-fill the
    least-loaded chapters globally — same distribution behavior we had
    before, just preceded by the anchor-matching pass.

    Within each chapter we then space the assigned facts at
    chapter_start + k * (chapter_duration / (allocation + 1)).
    """
    if not chapters or not facts:
        return [], 0

    assignment, matched_count = _assign_by_anchor(facts, chapters)
    placed: list[tuple[int, object]] = []
    for chapter, ch_facts in zip(chapters, assignment):
        if not ch_facts or chapter.duration_ms <= 0:
            continue
        n = len(ch_facts)
        for k, fact in enumerate(ch_facts, start=1):
            offset = chapter.duration_ms * k // (n + 1)
            placed.append((chapter.start_ms + offset, fact))
    placed.sort(key=lambda p: p[0])
    return placed, matched_count


def _assign_by_anchor(
    facts: list,
    chapters: tuple[Chapter, ...],
) -> tuple[list[list], int]:
    """For each fact, pick a chapter.

    1. If any anchor word (>=3 chars) appears in a chapter title, that
       chapter is a candidate. Among matching chapters, pick the least-
       loaded one (ties → lowest index). Counts as an anchor match.
    2. Otherwise (no anchors, or no anchor matches any title), the fact
       is queued for round-robin assignment to the globally-emptiest
       chapter. Sorted by specificity desc so high-quality facts spread
       first.

    Returns (assignment_per_chapter, count_of_anchor_matched_facts).
    """
    n_ch = len(chapters)
    assignment: list[list] = [[] for _ in range(n_ch)]
    title_lower = [c.title.lower() for c in chapters]

    def _least_loaded(candidates: list[int]) -> int:
        return min(candidates, key=lambda i: (len(assignment[i]), i))

    leftover: list = []
    anchor_matched = 0

    for fact in facts:
        anchors = [
            a.lower().strip()
            for a in getattr(fact, "anchors", []) or []
            if isinstance(a, str) and len(a.strip()) >= 3
        ]
        if not anchors:
            leftover.append(fact)
            continue
        matching = [
            i for i, t in enumerate(title_lower)
            if any(anchor in t for anchor in anchors)
        ]
        if not matching:
            leftover.append(fact)
            continue
        assignment[_least_loaded(matching)].append(fact)
        anchor_matched += 1
        log.info(
            "Anchor-matched fact to chapter %r via anchors %s",
            chapters[matching[0]].title,
            anchors,
        )

    # Round-robin leftover facts to least-loaded chapter overall, sorted
    # by specificity so the strongest facts spread first.
    leftover.sort(
        key=lambda f: -_SPECIFICITY_TO_INTEREST.get(
            getattr(f, "specificity", "medium"), 0
        )
    )
    all_idxs = list(range(n_ch))
    for fact in leftover:
        assignment[_least_loaded(all_idxs)].append(fact)

    return assignment, anchor_matched


def _allocate_largest_remainder(
    n: int, chapters: tuple[Chapter, ...]
) -> list[int]:
    """Distribute n items across chapters proportionally to their duration.

    Largest-remainder method: take the integer share, then hand out the
    leftover one-by-one to whichever chapter had the largest fractional
    remainder. Guarantees sum(allocations) == n.
    """
    total = sum(c.duration_ms for c in chapters)
    if total <= 0:
        # All chapters are zero-length somehow; spread evenly.
        even = n // max(1, len(chapters))
        result = [even] * len(chapters)
        for i in range(n - sum(result)):
            result[i % len(chapters)] += 1
        return result

    raw = [n * c.duration_ms / total for c in chapters]
    floors = [int(x) for x in raw]
    remainder = [(i, raw[i] - floors[i]) for i in range(len(chapters))]
    remainder.sort(key=lambda x: -x[1])
    leftover = n - sum(floors)
    for i in range(leftover):
        floors[remainder[i][0]] += 1
    return floors


# ─── no-chapters fallback: even spacing ──────────────────────────────────


def _place_evenly(n_facts: int, duration_ms: int) -> list[int]:
    """Even-space n_facts inside the [5%, 95%] portion of the movie."""
    start = int(duration_ms * _RUNTIME_EDGE_TRIM)
    end = int(duration_ms * (1.0 - _RUNTIME_EDGE_TRIM))
    span = end - start
    if span <= 0 or n_facts <= 0:
        return []
    step = span // (n_facts + 1)
    return [start + (i + 1) * step for i in range(n_facts)]


# ─── min-spacing guarantee (defensive) ───────────────────────────────────


def _enforce_min_spacing(timestamps: list[int]) -> list[int]:
    """Drop timestamps that fall within _MIN_CARD_SPACING_MS of the previous
    kept one. Preserves order; later collisions lose to earlier ones."""
    if not timestamps:
        return timestamps
    kept: list[int] = [timestamps[0]]
    for ts in timestamps[1:]:
        if ts - kept[-1] >= _MIN_CARD_SPACING_MS:
            kept.append(ts)
    return kept
