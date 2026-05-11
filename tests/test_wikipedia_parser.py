from __future__ import annotations

from poptrivia.prep.sources.wikipedia import _filter_preferred, _split_sections

SAMPLE = """
The Shining is a 1980 psychological horror film produced and directed by Stanley Kubrick.

== Plot ==
A family heads to an isolated hotel for the winter.

== Production ==
Kubrick shot the film at EMI Elstree Studios over fifty weeks.

=== Casting ===
Jack Nicholson was Kubrick's first choice.

== Filming ==
Steadicam operator Garrett Brown pioneered new techniques.

== Reception ==
The film was met with mixed reviews on initial release.

== Trivia ==
Random unrelated content.
""".strip()


def test_split_sections_extracts_named_blocks() -> None:
    out = _split_sections(SAMPLE)
    assert "(lead)" in out
    assert "Plot" in out
    assert "Production" in out
    assert "Filming" in out
    assert "Reception" in out
    assert "Kubrick shot the film" in out["Production"]


def test_filter_preferred_drops_unwanted_sections() -> None:
    out = _filter_preferred(_split_sections(SAMPLE))
    assert "(lead)" in out
    assert "Production" in out
    assert "Filming" in out
    assert "Reception" in out
    # Plot and Trivia aren't in the preferred list.
    assert "Plot" not in out
    assert "Trivia" not in out
