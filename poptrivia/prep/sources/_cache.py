from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("poptrivia.prep.cache")


def cache_path(cache_dir: Path, source: str, key: str, ext: str) -> Path:
    """Resolve a per-source cache file path.

    We don't sanitize aggressively — the inputs are IMDB IDs, TMDB IDs, and
    URL-encoded titles, so the worst we'd see is colons in plex GUIDs (which
    we don't use here).
    """
    safe = key.replace("/", "_").replace(":", "_")
    p = cache_dir / source / f"{safe}.{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    log.debug("cache hit: %s", path)
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    log.debug("cache wrote: %s (%d bytes)", path, len(content))


def read_bytes(path: Path) -> bytes | None:
    if not path.exists():
        return None
    return path.read_bytes()


def write_bytes(path: Path, content: bytes) -> None:
    path.write_bytes(content)
