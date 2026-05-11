from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from poptrivia.models import RawFactExtracted, TrackFile, TriviaCard
from poptrivia.prep.llm.client import OllamaClient
from poptrivia.prep.subtitles import SubtitleEntry

log = logging.getLogger("poptrivia.prep.llm.stage2")

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# 15-minute windows per the brief.
_WINDOW_MS = 15 * 60 * 1000

# Concurrent Stage 2 windows in flight. Same caveat as Stage 1: only
# actually parallel if Ollama is run with OLLAMA_NUM_PARALLEL >= this.
_STAGE2_CONCURRENCY = 3

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "timestamp_ms": {"type": "integer"},
                    "text": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "production",
                            "casting",
                            "cinematography",
                            "historical_context",
                            "easter_egg",
                            "cultural_impact",
                            "goof",
                            "cut_content",
                            "cast_biography",
                            "score_music",
                        ],
                    },
                    "interest_level": {"type": "integer", "minimum": 1, "maximum": 5},
                    "source_fact_id": {"type": "string"},
                    "anchor_evidence": {"type": "string"},
                },
                "required": [
                    "id",
                    "timestamp_ms",
                    "text",
                    "category",
                    "interest_level",
                    "source_fact_id",
                ],
            },
        }
    },
    "required": ["cards"],
}


def _load_prompts() -> tuple[str, str]:
    system = (_PROMPTS_DIR / "stage2_system.txt").read_text(encoding="utf-8")
    examples = (_PROMPTS_DIR / "stage2_examples.txt").read_text(encoding="utf-8")
    return system, examples


async def align_facts(
    *,
    movie_title: str,
    movie_year: int | None,
    movie_duration_ms: int | None,
    facts: list[RawFactExtracted],
    subtitles: list[SubtitleEntry],
    llm: OllamaClient,
    max_cards: int,
    num_ctx: int | None = None,
) -> list[TriviaCard]:
    """Place facts onto a per-window timeline.

    Architecture: we pre-assign each fact to exactly one window in Python
    (anchored facts to the first window where their anchor appears, anchor-
    less facts round-robin), then ask the LLM only to choose the precise
    *timestamp* within that window. The model is good at "when within these
    15 minutes" but bad at "which of the 9 windows" — taking the global
    coordination problem out of its hands fixes the distribution issue we
    saw on Bridesmaids (9 cards clustered in 30-53 min, rest of the movie
    empty).
    """
    system, examples = _load_prompts()
    fact_index = _build_fact_index(facts)
    fact_id_by_object = {id(f): fid for fid, f in fact_index.items()}

    windows = _split_subtitle_windows(subtitles, movie_duration_ms)
    log.info("Stage 2: %d facts across %d windows", len(facts), len(windows))

    assignment = _assign_facts_to_windows(facts, windows)
    for i, win_facts in enumerate(assignment):
        log.info(
            "Stage 2 window %d/%d: %d fact(s) pre-assigned",
            i + 1,
            len(windows),
            len(win_facts),
        )

    sem = asyncio.Semaphore(_STAGE2_CONCURRENCY)

    async def _process_window(
        i: int,
        start_ms: int,
        end_ms: int,
        window_subs: list[SubtitleEntry],
        window_facts: list[RawFactExtracted],
    ) -> list[TriviaCard]:
        if not window_facts:
            log.info(
                "Stage 2 window %d/%d (%.1fm-%.1fm): no facts assigned",
                i + 1,
                len(windows),
                start_ms / 60_000,
                end_ms / 60_000,
            )
            return []

        # Fact ids span ALL facts. The prompt only sees the assigned subset,
        # but the ids match the global index so _parse_cards finds them.
        local_block = _format_facts_subset(window_facts, fact_id_by_object, fact_index)
        user_prompt = _build_user_prompt(
            movie_title=movie_title,
            movie_year=movie_year,
            examples=examples,
            facts_block=local_block,
            window_index=i,
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            subtitles=window_subs,
        )
        async with sem:
            try:
                result = await llm.generate(
                    user_prompt,
                    system=system,
                    schema=_OUTPUT_SCHEMA,
                    temperature=0.3,
                    num_ctx=num_ctx,
                )
            except Exception as e:
                log.warning("Stage 2 window %d/%d failed: %s", i + 1, len(windows), e)
                return []
            if not isinstance(result, dict):
                log.warning("Stage 2 window %d returned non-dict: %r", i + 1, result)
                return []

        cards = _parse_cards(result, fact_index=fact_index)
        log.info(
            "Stage 2 window %d/%d (%.1fm-%.1fm): %d/%d placed",
            i + 1,
            len(windows),
            start_ms / 60_000,
            end_ms / 60_000,
            len(cards),
            len(window_facts),
        )
        return cards

    window_results = await asyncio.gather(
        *(
            _process_window(i, start_ms, end_ms, window_subs, assignment[i])
            for i, (start_ms, end_ms, window_subs) in enumerate(windows)
        )
    )
    all_cards: list[TriviaCard] = [c for batch in window_results for c in batch]

    # Pre-assignment means each fact lands in exactly one window, so the
    # old _dedupe_by_fact step is a no-op now. We still cap+renumber in
    # case the model placed multiple cards per fact (shouldn't, but safe).
    deduped = _dedupe_by_fact(all_cards)
    capped = _cap_and_renumber(deduped, max_cards=max_cards)
    log.info(
        "Stage 2 totals: %d raw -> %d deduped -> %d after cap (max %d)",
        len(all_cards),
        len(deduped),
        len(capped),
        max_cards,
    )
    return capped


# ─── fact -> window assignment ────────────────────────────────────────────


def _assign_facts_to_windows(
    facts: list[RawFactExtracted],
    windows: list[tuple[int, int, list[SubtitleEntry]]],
) -> list[list[RawFactExtracted]]:
    """Decide which window each fact belongs to, with load balancing.

    - Anchored facts: scanned against every window's subtitle blob. We
      collect ALL matching windows, then assign to whichever has the
      fewest facts so far (ties broken by lower index = earlier in the
      film). This prevents character-name anchors like "Annie" from
      dumping every related fact into window 1.
    - Anchor-less facts (and anchored facts that matched nowhere): also
      assigned to the least-loaded window, sorted by specificity desc so
      high-quality facts spread first.

    Result: even distribution across the timeline regardless of how
    common the anchor terms are.
    """
    if not windows:
        return []
    n_windows = len(windows)
    assignment: list[list[RawFactExtracted]] = [[] for _ in range(n_windows)]

    window_blobs = [
        " ".join(s.text for s in subs).lower() for (_a, _b, subs) in windows
    ]

    def _least_loaded_among(candidate_indices: list[int]) -> int:
        return min(candidate_indices, key=lambda i: (len(assignment[i]), i))

    leftover: list[RawFactExtracted] = []
    for fact in facts:
        anchors = [a for a in fact.anchors if a and a.strip()]
        if not anchors:
            leftover.append(fact)
            continue
        matching = [
            idx
            for idx, blob in enumerate(window_blobs)
            if any(a.lower() in blob for a in anchors)
        ]
        if not matching:
            leftover.append(fact)
            continue
        assignment[_least_loaded_among(matching)].append(fact)

    # Round-robin distribute leftover facts to the globally-emptiest
    # window. Sort by specificity desc so the strong stuff is spread
    # first.
    leftover.sort(key=lambda f: -_SPECIFICITY_RANK.get(f.specificity, 0))
    all_windows = list(range(n_windows))
    for fact in leftover:
        assignment[_least_loaded_among(all_windows)].append(fact)

    return assignment


_SPECIFICITY_RANK = {"high": 3, "medium": 2, "low": 1}


def _format_facts_subset(
    window_facts: list[RawFactExtracted],
    fact_id_by_object: dict[int, str],
    fact_index: dict[str, RawFactExtracted],
) -> str:
    """Render just the facts assigned to one window, preserving their
    globally-assigned fXXX ids so _parse_cards finds them in fact_index."""
    lines: list[str] = []
    for fact in window_facts:
        fid = fact_id_by_object.get(id(fact))
        if fid is None:
            # Fallback — shouldn't happen, but degrade gracefully.
            continue
        anchors = ", ".join(fact.anchors) if fact.anchors else "(no anchors)"
        lines.append(
            f"- id={fid} ({fact.category}, anchors=[{anchors}]):\n    {fact.fact}"
        )
    return "\n".join(lines)


def write_track_file(
    *,
    tracks_dir: Path,
    plex_guid: str,
    imdb_id: str | None,
    tmdb_id: str | None,
    title: str,
    year: int | None,
    model: str,
    cards: list[TriviaCard],
) -> Path:
    track = TrackFile(
        plex_guid=plex_guid,
        imdb_id=imdb_id,
        tmdb_id=tmdb_id,
        title=title,
        year=year,
        generated_at=datetime.now(timezone.utc),
        model=model,
        cards=cards,
    )
    safe = plex_guid.replace("/", "_").replace(":", "_")
    path = tracks_dir / f"{safe}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(track.model_dump_json(indent=2), encoding="utf-8")
    return path


# ─── prompt assembly ──────────────────────────────────────────────────────


def _build_user_prompt(
    *,
    movie_title: str,
    movie_year: int | None,
    examples: str,
    facts_block: str,
    window_index: int,
    window_start_ms: int,
    window_end_ms: int,
    subtitles: list[SubtitleEntry],
) -> str:
    year_str = f" ({movie_year})" if movie_year else ""
    return (
        f"FILM: {movie_title}{year_str}\n\n"
        f"--- FEW-SHOT EXAMPLES ---\n{examples}\n\n"
        f"--- FACTS ---\n{facts_block}\n\n"
        f"--- SUBTITLE WINDOW ({_format_ts(window_start_ms)} to "
        f"{_format_ts(window_end_ms)}) ---\n"
        f"{_format_subs(subtitles)}\n\n"
        "Respond with a JSON object {\"cards\": [...]}. Only include facts "
        "that actually fit this window. Use the placement rules strictly."
    )


def _build_fact_index(facts: list[RawFactExtracted]) -> dict[str, RawFactExtracted]:
    """Assign and index sequential IDs to each fact, mutating in place."""
    out: dict[str, RawFactExtracted] = {}
    for idx, fact in enumerate(facts):
        fid = f"f{idx + 1:03d}"
        out[fid] = fact
    return out


def _format_facts(facts: list[RawFactExtracted]) -> str:
    lines: list[str] = []
    for idx, f in enumerate(facts):
        fid = f"f{idx + 1:03d}"
        anchors = ", ".join(f.anchors) if f.anchors else "(no anchors)"
        lines.append(f"- id={fid} ({f.category}, anchors=[{anchors}]):\n    {f.fact}")
    return "\n".join(lines)


def _format_subs(entries: list[SubtitleEntry]) -> str:
    if not entries:
        return "(no subtitles in this window)"
    return "\n".join(
        f"[{_format_ts(e.start_ms)}] {e.text}" for e in entries
    )


def _format_ts(ms: int) -> str:
    total_sec, _ = divmod(ms, 1000)
    minutes, seconds = divmod(total_sec, 60)
    return f"{minutes:02d}:{seconds:02d}"


# ─── windowing ────────────────────────────────────────────────────────────


def _split_subtitle_windows(
    subtitles: list[SubtitleEntry], movie_duration_ms: int | None
) -> list[tuple[int, int, list[SubtitleEntry]]]:
    """Partition subtitles into 15-minute chunks.

    Returns (start_ms, end_ms, entries_in_window) tuples. Windows are
    inclusive of their start and exclusive of their end. An empty window is
    still emitted if no subtitles exist there but the movie's duration
    extends into that range — facts with no subtitle anchor may still want
    to land there.
    """
    if not subtitles and not movie_duration_ms:
        return []

    last_ms = movie_duration_ms or subtitles[-1].end_ms
    windows: list[tuple[int, int, list[SubtitleEntry]]] = []

    start = 0
    sub_idx = 0
    while start < last_ms:
        end = min(start + _WINDOW_MS, last_ms)
        bucket: list[SubtitleEntry] = []
        while sub_idx < len(subtitles) and subtitles[sub_idx].start_ms < end:
            bucket.append(subtitles[sub_idx])
            sub_idx += 1
        windows.append((start, end, bucket))
        start = end
    return windows


# ─── parsing / merging ────────────────────────────────────────────────────


def _parse_cards(
    result: dict, *, fact_index: dict[str, RawFactExtracted]
) -> list[TriviaCard]:
    out: list[TriviaCard] = []
    cards_raw = result.get("cards", [])
    if not isinstance(cards_raw, list):
        log.warning("Stage 2: 'cards' was not a list: %r", cards_raw)
        return out
    for c in cards_raw:
        if not isinstance(c, dict):
            continue
        try:
            card = TriviaCard.model_validate(c)
        except ValidationError as e:
            log.warning("Stage 2: dropping invalid card: %s", e)
            continue
        if card.source_fact_id not in fact_index:
            log.warning(
                "Stage 2: card refers to unknown fact_id=%s — dropping",
                card.source_fact_id,
            )
            continue
        out.append(card)
    return out


def _dedupe_by_fact(cards: list[TriviaCard]) -> list[TriviaCard]:
    """Same fact placed in multiple windows? Keep highest interest_level."""
    best: dict[str, TriviaCard] = {}
    for c in cards:
        prior = best.get(c.source_fact_id)
        if prior is None or c.interest_level > prior.interest_level:
            best[c.source_fact_id] = c
    return list(best.values())


def _cap_and_renumber(cards: list[TriviaCard], *, max_cards: int) -> list[TriviaCard]:
    # Sort by timestamp for the final output, but pick the top max_cards by
    # interest_level first if we're over budget.
    if len(cards) > max_cards:
        cards = sorted(cards, key=lambda c: -c.interest_level)[:max_cards]
    ordered = sorted(cards, key=lambda c: c.timestamp_ms)
    for i, c in enumerate(ordered):
        c.id = f"card_{i + 1:03d}"
    return ordered
