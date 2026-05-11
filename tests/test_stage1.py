from __future__ import annotations

from poptrivia.models import RawFactExtracted, RawSourceItem
from poptrivia.prep.llm.stage1_extract import (
    _dedupe,
    _iter_chunks,
    _rank_and_trim,
)


def _fact(text: str, specificity: str = "high") -> RawFactExtracted:
    return RawFactExtracted(
        fact=text,
        source="imdb_trivia",
        specificity=specificity,  # type: ignore[arg-type]
        anchors=[],
        category="production",
    )


def test_dedupe_drops_normalized_duplicates() -> None:
    facts = [
        _fact("Kubrick shot 127 takes of the baseball bat scene."),
        # Exact duplicate, different punctuation/case.
        _fact("Kubrick shot 127 takes of the baseball-bat scene!"),
        _fact("The Colorado Lounge was the largest set ever built at Elstree."),
    ]
    out = _dedupe(facts)
    assert len(out) == 2
    assert out[0].fact.startswith("Kubrick shot 127")
    assert out[1].fact.startswith("The Colorado")


def test_rank_and_trim_prefers_high_specificity() -> None:
    facts = [
        _fact("low one", "low"),
        _fact("medium one", "medium"),
        _fact("high A", "high"),
        _fact("high B", "high"),
    ]
    out = _rank_and_trim(facts, target_count=2)  # limit = 3
    assert [f.fact for f in out] == ["high A", "high B", "medium one"]


def test_iter_chunks_splits_long_items_on_paragraphs() -> None:
    long_para = "Paragraph A. " * 500
    second_para = "Paragraph B. " * 500
    item = RawSourceItem(
        source="wikipedia",
        section="Production",
        text=long_para + "\n\n" + second_para,
    )
    chunks = list(_iter_chunks([item]))
    # Two chunks: roughly one per paragraph.
    assert len(chunks) >= 2
    for text, source, section in chunks:
        assert source == "wikipedia"
        assert section == "Production"


def test_iter_chunks_skips_empty_items() -> None:
    items = [
        RawSourceItem(source="imdb_trivia", text=""),
        RawSourceItem(source="imdb_trivia", text="  "),
        RawSourceItem(source="imdb_trivia", text="something"),
    ]
    chunks = list(_iter_chunks(items))
    assert len(chunks) == 1
    assert chunks[0][0] == "something"
