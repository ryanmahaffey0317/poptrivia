from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import ValidationError

from poptrivia.models import RawFactExtracted, RawSourceItem
from poptrivia.prep.llm.client import OllamaClient

log = logging.getLogger("poptrivia.prep.llm.stage1")

_PROMPTS_DIR = Path(__file__).parent / "prompts"


# Rough char-per-token estimate; we don't ship a tokenizer dependency for this.
_APPROX_CHARS_PER_TOKEN = 4
_CHUNK_TARGET_TOKENS = 3000

# JSON schema we pass to Ollama for structured output. Mirrors RawFactExtracted.
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "source": {
                        "type": "string",
                        "enum": ["imdb_trivia", "imdb_goofs", "wikipedia"],
                    },
                    "specificity": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "anchors": {"type": "array", "items": {"type": "string"}},
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
                },
                "required": [
                    "fact",
                    "source",
                    "specificity",
                    "anchors",
                    "category",
                ],
            },
        }
    },
    "required": ["facts"],
}


def _load_prompts() -> tuple[str, str]:
    system = (_PROMPTS_DIR / "stage1_system.txt").read_text(encoding="utf-8")
    examples = (_PROMPTS_DIR / "stage1_examples.txt").read_text(encoding="utf-8")
    return system, examples


async def extract_facts(
    *,
    movie_title: str,
    movie_year: int | None,
    source_items: list[RawSourceItem],
    llm: OllamaClient,
    target_count: int,
) -> list[RawFactExtracted]:
    """Run stage 1 extraction over all source material.

    Steps:
      1. Chunk inputs to ~3000 tokens each, preserving source/section metadata.
      2. Ask Ollama for facts per chunk.
      3. Dedupe by normalized fact text.
      4. Rank by specificity; trim to target_count * 1.5.
    """
    system, examples = _load_prompts()

    raw_facts: list[RawFactExtracted] = []
    chunks = list(_iter_chunks(source_items))
    log.info(
        "Stage 1: %d source items -> %d chunks (%s)",
        len(source_items),
        len(chunks),
        movie_title,
    )

    for i, (chunk, source, section) in enumerate(chunks):
        user_prompt = _build_user_prompt(
            movie_title=movie_title,
            movie_year=movie_year,
            examples=examples,
            chunk_text=chunk,
            source=source,
            section=section,
        )
        try:
            result = await llm.generate(
                user_prompt, system=system, schema=_OUTPUT_SCHEMA, temperature=0.2
            )
        except Exception as e:
            log.warning("Stage 1 chunk %d/%d failed: %s", i + 1, len(chunks), e)
            continue
        if not isinstance(result, dict):
            log.warning("Stage 1 chunk %d returned non-dict: %r", i + 1, result)
            continue

        parsed = _parse_facts(result, source)
        log.info(
            "Stage 1 chunk %d/%d (%s%s): %d facts",
            i + 1,
            len(chunks),
            source,
            f"/{section}" if section else "",
            len(parsed),
        )
        raw_facts.extend(parsed)

    deduped = _dedupe(raw_facts)
    ranked = _rank_and_trim(deduped, target_count=target_count)
    log.info(
        "Stage 1 totals: %d raw -> %d deduped -> %d after trim (target %d)",
        len(raw_facts),
        len(deduped),
        len(ranked),
        target_count,
    )
    return ranked


# ─── prompt assembly ──────────────────────────────────────────────────────


def _build_user_prompt(
    *,
    movie_title: str,
    movie_year: int | None,
    examples: str,
    chunk_text: str,
    source: str,
    section: str,
) -> str:
    year_str = f" ({movie_year})" if movie_year else ""
    section_str = f", section={section}" if section else ""
    return (
        f"FILM: {movie_title}{year_str}\n\n"
        f"--- FEW-SHOT EXAMPLES ---\n{examples}\n"
        "--- INPUT ---\n"
        f"INPUT ({source}{section_str}):\n{chunk_text}\n\n"
        "Respond with a JSON object of the form {\"facts\": [...]} matching "
        "the schema. Return an empty array if no facts qualify."
    )


# ─── chunking ─────────────────────────────────────────────────────────────


def _iter_chunks(items: list[RawSourceItem]):
    """Yield (text, source, section) tuples sized to ~_CHUNK_TARGET_TOKENS."""
    target_chars = _CHUNK_TARGET_TOKENS * _APPROX_CHARS_PER_TOKEN
    for item in items:
        text = item.text.strip()
        if not text:
            continue
        if len(text) <= target_chars:
            yield text, item.source, item.section
            continue
        # Paragraph-aware split.
        paragraphs = re.split(r"\n\s*\n", text)
        buf: list[str] = []
        buf_len = 0
        for para in paragraphs:
            p = para.strip()
            if not p:
                continue
            if buf_len + len(p) > target_chars and buf:
                yield "\n\n".join(buf), item.source, item.section
                buf, buf_len = [], 0
            buf.append(p)
            buf_len += len(p) + 2
        if buf:
            yield "\n\n".join(buf), item.source, item.section


# ─── parsing / dedupe / ranking ───────────────────────────────────────────


def _parse_facts(result: dict, source: str) -> list[RawFactExtracted]:
    out: list[RawFactExtracted] = []
    facts_raw = result.get("facts", [])
    if not isinstance(facts_raw, list):
        log.warning("Stage 1: 'facts' was not a list: %r", facts_raw)
        return out
    for f in facts_raw:
        if not isinstance(f, dict):
            continue
        # Force the source label to match what we sent the model; the model
        # sometimes echoes the wrong one when given a wikipedia chunk.
        f["source"] = source if source != "wikipedia" else "wikipedia"
        try:
            out.append(RawFactExtracted.model_validate(f))
        except ValidationError as e:
            log.warning("Stage 1: dropping invalid fact: %s", e)
    return out


_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text.lower()).strip()


# Common English stopwords + filler that don't help distinguish facts.
# Trimmed to the high-frequency ones that would otherwise pollute Jaccard
# overlap calculations.
_STOPWORDS: frozenset[str] = frozenset(
    """
    the a an and or but of for to in on at by with from as is was were be
    been being this that these those it its his her their he she they
    you we us our your i had has have having do did does done would could
    should will shall may might can must than then so if while during
    after before because since until when where which who whom whose
    what why how also too just very much many more most some any all
    each every other another such only own same about over under into
    out up down off out across against between within without through
    such whether either neither
    """.split()
)

# Two facts are considered duplicates when their significant-token sets
# overlap by more than this fraction. We use the *overlap coefficient*
# (|A∩B| / min(|A|, |B|)) rather than Jaccard because cross-source
# paraphrases tend to use synonyms — Jaccard's union grows with synonym
# variance and depresses the score below useful thresholds. Overlap
# coefficient is invariant to the longer fact's extra unique tokens.
_DEDUPE_OVERLAP_THRESHOLD = 0.55

_SPECIFICITY_RANK = {"high": 3, "medium": 2, "low": 1}


def _significant_tokens(text: str) -> set[str]:
    """Lowercase tokens, alphanumeric, stopwords + short words dropped.

    The remaining set is the 'fingerprint' of the fact — proper nouns,
    numbers, specific verbs and rare words dominate.
    """
    return {
        w for w in _TOKEN_RE.findall(text.lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def _dedupe(facts: list[RawFactExtracted]) -> list[RawFactExtracted]:
    """Drop facts that overlap heavily with a kept fact.

    Catches both verbatim duplicates and cross-source paraphrases (IMDB
    and Wikipedia describing the same anecdote in different words). When
    two facts collide, we keep the higher-specificity one.

    O(n²) over kept-facts; fine for n in the low hundreds.
    """
    kept: list[tuple[RawFactExtracted, set[str]]] = []
    for f in facts:
        toks = _significant_tokens(f.fact)
        if not toks:
            continue
        f_rank = _SPECIFICITY_RANK.get(f.specificity, 0)
        dup_index: int | None = None
        for i, (_existing, existing_toks) in enumerate(kept):
            inter = len(toks & existing_toks)
            denom = min(len(toks), len(existing_toks))
            if denom and inter / denom >= _DEDUPE_OVERLAP_THRESHOLD:
                dup_index = i
                break
        if dup_index is None:
            kept.append((f, toks))
            continue
        # Collision: keep the better one.
        existing, _ = kept[dup_index]
        existing_rank = _SPECIFICITY_RANK.get(existing.specificity, 0)
        if f_rank > existing_rank:
            kept[dup_index] = (f, toks)
    return [f for (f, _t) in kept]


def _rank_and_trim(
    facts: list[RawFactExtracted], *, target_count: int
) -> list[RawFactExtracted]:
    # Stable sort by specificity desc, then by fact length asc (shorter facts
    # generally read better as pop-ups).
    sorted_facts = sorted(
        facts,
        key=lambda f: (-_SPECIFICITY_RANK.get(f.specificity, 0), len(f.fact)),
    )
    limit = int(target_count * 1.5)
    return sorted_facts[:limit]
