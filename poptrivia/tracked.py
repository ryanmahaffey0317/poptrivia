from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("poptrivia.tracked")


# Default template written on first run so the user has something to edit.
_TEMPLATE = """\
# poptrivia tracked-movies list.
#
# One movie per line, in either of these forms:
#
#   tt0081505               # IMDB id (preferred — unambiguous)
#   plex://movie/abc123     # Plex GUID (also fine)
#
# Anything after a '#' is treated as a comment. Blank lines are ignored.
# Edit this file at any time — poptrivia polls it every 30 minutes and
# also consults it on every playback event.
#
# Behavior:
#   - Movies in this list get prep queued (Wikipedia + TMDB + IMDB sources,
#     LLM fact extraction, timestamp alignment, written to a track file).
#   - When a track exists, playing that movie in Plex fires trivia cards
#     to your Discord channel.
#   - Movies NOT in this list are ignored entirely — no prep, no cards.
#
# Examples (uncomment / replace with your own):
# tt0081505               # The Shining (1980)
# tt1478338               # Bridesmaids (2011)
"""


def read_tracked(path: Path) -> set[str]:
    """Parse the tracked-list file into a set of normalized identifiers.

    Lines are stripped, lowercased, and have inline '#' comments removed.
    Returns an empty set if the file doesn't exist.
    """
    if not path.exists():
        return set()
    out: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        out.add(line.lower())
    return out


def ensure_template(path: Path) -> None:
    """Write the default template if the file doesn't exist yet."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE, encoding="utf-8")
    log.info("Created default tracked list at %s", path)


def is_tracked(
    *,
    tracked: set[str],
    imdb_id: str | None,
    plex_guid: str | None,
) -> bool:
    """Check whether a movie matches any identifier in the tracked set."""
    if imdb_id and imdb_id.lower() in tracked:
        return True
    if plex_guid and plex_guid.lower() in tracked:
        return True
    return False


def remove_from_tracked(path: Path, identifier: str) -> bool:
    """Atomically remove an identifier line from the tracked-list file.

    Match is case-insensitive against the line's identifier (the bit before
    any inline `#` comment). Lines that don't contain the identifier are
    preserved verbatim — including comments, blank lines, and the file's
    header block.

    Returns True if a line was removed, False if the identifier wasn't
    present (no-op). Writes through a tempfile + rename for atomicity.
    """
    identifier = identifier.strip()
    if not identifier:
        raise ValueError("identifier cannot be empty")
    if not path.exists():
        return False

    target = identifier.lower()
    kept_lines: list[str] = []
    removed = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        token = raw_line.split("#", 1)[0].strip().lower()
        if token == target:
            removed = True
            continue
        kept_lines.append(raw_line)

    if not removed:
        return False

    body = "\n".join(kept_lines)
    if body and not body.endswith("\n"):
        body += "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)
    log.info("Removed %s from tracked list", identifier)
    return True


def append_to_tracked(
    path: Path,
    identifier: str,
    *,
    comment: str | None = None,
) -> None:
    """Atomically append a new identifier line to the tracked-list file.

    Used by the Discord bot when a user approves "Generate trivia for this."
    Writes through a tempfile + rename so a partial write during a poller
    read can't corrupt the file. Idempotent — if the identifier is already
    present, this is a no-op.
    """
    identifier = identifier.strip()
    if not identifier:
        raise ValueError("identifier cannot be empty")

    existing = read_tracked(path) if path.exists() else set()
    if identifier.lower() in existing:
        log.info("Tracked list already contains %s; no-op", identifier)
        return

    body = path.read_text(encoding="utf-8") if path.exists() else _TEMPLATE
    if body and not body.endswith("\n"):
        body += "\n"
    suffix = f"   # {comment}" if comment else ""
    body += f"{identifier}{suffix}\n"

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)
    log.info("Appended %s to tracked list", identifier)
