from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("poptrivia.prep.chapters")


@dataclass(frozen=True)
class Chapter:
    """One chapter in a movie. Times are in milliseconds from the start."""

    start_ms: int
    end_ms: int
    title: str

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass(frozen=True)
class MovieMetadata:
    """Lightweight movie info derived from a single ffprobe call.

    No stream decoding involved — this is purely a metadata read, so it
    works in milliseconds even on 80 GB Remuxes.
    """

    duration_ms: int
    chapters: tuple[Chapter, ...]


class ChapterProbeError(RuntimeError):
    pass


_FFPROBE_TIMEOUT_SECONDS = 30


async def probe_metadata(file_path: Path) -> MovieMetadata:
    """Run ffprobe -show_chapters -show_format and parse the result.

    Returns duration in ms (from the format block) and the chapter list
    (empty tuple when the file has no chapter markers).
    """
    if not file_path.exists():
        raise ChapterProbeError(f"Media file not found: {file_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_chapters",
        "-show_format",
        "-of",
        "json",
        str(file_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_FFPROBE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        raise ChapterProbeError(
            f"ffprobe timed out after {_FFPROBE_TIMEOUT_SECONDS}s for {file_path}"
        )

    if proc.returncode != 0 or not stdout:
        raise ChapterProbeError(
            f"ffprobe failed (rc={proc.returncode}): "
            f"{stderr.decode('utf-8', errors='replace')[:500]}"
        )

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise ChapterProbeError(f"ffprobe returned non-JSON: {e}")

    duration_ms = _parse_duration(data)
    chapters = _parse_chapters(data)
    log.info(
        "ffprobe: duration=%dms (%.1fm), %d chapter(s) for %s",
        duration_ms,
        duration_ms / 60_000,
        len(chapters),
        file_path.name,
    )
    return MovieMetadata(duration_ms=duration_ms, chapters=tuple(chapters))


def _parse_duration(data: dict) -> int:
    fmt = data.get("format") or {}
    raw = fmt.get("duration")
    if raw is None:
        raise ChapterProbeError("ffprobe format block has no duration")
    try:
        return int(float(raw) * 1000)
    except (TypeError, ValueError) as e:
        raise ChapterProbeError(f"unparseable duration {raw!r}: {e}")


def _parse_chapters(data: dict) -> list[Chapter]:
    out: list[Chapter] = []
    for i, c in enumerate(data.get("chapters") or []):
        try:
            start = int(float(c["start_time"]) * 1000)
            end = int(float(c["end_time"]) * 1000)
        except (KeyError, TypeError, ValueError):
            log.warning("Skipping malformed chapter %d: %r", i, c)
            continue
        title = (c.get("tags") or {}).get("title") or f"Chapter {i + 1}"
        out.append(Chapter(start_ms=start, end_ms=end, title=title))
    return out
