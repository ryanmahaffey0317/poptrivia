from __future__ import annotations

from pathlib import Path

import pytest

from poptrivia.tracked import append_to_tracked, read_tracked


def test_append_to_tracked_creates_file_if_missing(tmp_path: Path) -> None:
    p = tmp_path / "tracked.txt"
    append_to_tracked(p, "tt0081505", comment="The Shining (1980)")
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "tt0081505" in text
    assert "The Shining (1980)" in text


def test_append_to_tracked_is_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "tracked.txt"
    append_to_tracked(p, "tt0081505")
    append_to_tracked(p, "tt0081505")  # second call is a no-op
    entries = read_tracked(p)
    assert entries == {"tt0081505"}


def test_append_to_tracked_preserves_existing_content(tmp_path: Path) -> None:
    p = tmp_path / "tracked.txt"
    p.write_text(
        "# header comment\ntt9999999   # existing movie\n",
        encoding="utf-8",
    )
    append_to_tracked(p, "tt0081505", comment="The Shining")
    text = p.read_text(encoding="utf-8")
    assert "# header comment" in text  # preserved
    assert "tt9999999" in text         # preserved
    assert "tt0081505" in text         # new
    assert read_tracked(p) == {"tt9999999", "tt0081505"}


def test_append_to_tracked_rejects_empty(tmp_path: Path) -> None:
    p = tmp_path / "tracked.txt"
    with pytest.raises(ValueError):
        append_to_tracked(p, "")
    with pytest.raises(ValueError):
        append_to_tracked(p, "   ")


def test_append_to_tracked_atomic_via_tempfile(tmp_path: Path) -> None:
    """The function should write through a .tmp file and rename, leaving
    no stray tempfile on success."""
    p = tmp_path / "tracked.txt"
    append_to_tracked(p, "tt0081505")
    assert p.exists()
    assert not p.with_suffix(p.suffix + ".tmp").exists()
