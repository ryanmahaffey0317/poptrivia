from __future__ import annotations

import asyncio
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pysrt

log = logging.getLogger("poptrivia.prep.subtitles")


@dataclass
class SubtitleEntry:
    start_ms: int
    end_ms: int
    text: str


@dataclass
class SubtitleWindow:
    """A coalesced window of subtitle entries, suitable for LLM input."""

    start_ms: int
    end_ms: int
    text: str


class SubtitleExtractError(RuntimeError):
    pass


async def extract(file_path: Path) -> list[SubtitleEntry]:
    """Get subtitles for a movie file.

    Tries in order:
      1. ffmpeg embedded-subtitle extraction (first English stream we find).
      2. Sidecar .srt next to the file (foo.srt, foo.en.srt, foo.en.eng.srt).
      3. faster-whisper transcription with large-v3 (slow, opt-in heavy dep).
    """
    if not file_path.exists():
        raise SubtitleExtractError(f"Media file not found: {file_path}")

    embedded = await _try_ffmpeg(file_path)
    if embedded:
        log.info("Subtitles via embedded stream (%d entries)", len(embedded))
        return embedded

    sidecar = _try_sidecar(file_path)
    if sidecar:
        log.info("Subtitles via sidecar SRT (%d entries)", len(sidecar))
        return sidecar

    log.warning("No embedded or sidecar subs for %s — falling back to Whisper (slow!)", file_path)
    transcribed = await _try_whisper(file_path)
    log.info("Subtitles via Whisper transcription (%d entries)", len(transcribed))
    return transcribed


# ─── ffmpeg ───────────────────────────────────────────────────────────────


async def _try_ffmpeg(file_path: Path) -> list[SubtitleEntry] | None:
    if shutil.which("ffmpeg") is None:
        log.warning("ffmpeg not on PATH; skipping embedded subtitle extraction")
        return None

    # We try the first three subtitle streams in order. Picking a *specifically*
    # English stream requires probing first, which we skip — most movies'
    # stream 0:s:0 is English, and we fall back to sidecar/Whisper if not.
    for stream_index in range(3):
        out = await _run_ffmpeg_to_srt(file_path, stream_index)
        if out is None:
            continue
        entries = _parse_srt_text(out)
        if entries:
            return entries
    return None


async def _run_ffmpeg_to_srt(file_path: Path, stream_index: int) -> str | None:
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(file_path),
        "-map",
        f"0:s:{stream_index}",
        "-c:s",
        "srt",
        "-f",
        "srt",
        "-",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0 or not stdout:
        # Most common case: stream index doesn't exist. Don't log loudly.
        log.debug(
            "ffmpeg subtitle stream %d failed (rc=%s): %s",
            stream_index,
            proc.returncode,
            stderr.decode("utf-8", errors="replace")[:200],
        )
        return None
    return stdout.decode("utf-8", errors="replace")


# ─── sidecar ──────────────────────────────────────────────────────────────


_SIDECAR_PATTERNS = (".srt", ".en.srt", ".eng.srt", ".en.eng.srt", ".english.srt")


def _try_sidecar(file_path: Path) -> list[SubtitleEntry] | None:
    base = file_path.with_suffix("")
    for suffix in _SIDECAR_PATTERNS:
        candidate = Path(str(base) + suffix)
        if candidate.exists():
            log.info("Sidecar subtitle: %s", candidate)
            return _parse_srt_text(candidate.read_text(encoding="utf-8", errors="replace"))
    return None


def _parse_srt_text(text: str) -> list[SubtitleEntry]:
    try:
        items = pysrt.from_string(text)
    except Exception as e:  # pysrt raises a few different exception types
        log.warning("Failed to parse SRT: %s", e)
        return []
    return [
        SubtitleEntry(
            start_ms=_to_ms(item.start),
            end_ms=_to_ms(item.end),
            text=_clean(item.text_without_tags),
        )
        for item in items
        if (item.text_without_tags or "").strip()
    ]


def _to_ms(t: pysrt.SubRipTime) -> int:
    return (
        t.hours * 3600_000
        + t.minutes * 60_000
        + t.seconds * 1_000
        + t.milliseconds
    )


_WS_RE = re.compile(r"\s+")
_BRACKETS_RE = re.compile(r"\[[^\]]+\]|\([A-Z ]{3,}\)")  # [SCREAMS], (LAUGHTER)


def _clean(text: str) -> str:
    text = text.replace("\n", " ")
    text = _BRACKETS_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


# ─── Whisper fallback ─────────────────────────────────────────────────────


async def _try_whisper(file_path: Path) -> list[SubtitleEntry]:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise SubtitleExtractError(
            "No embedded/sidecar subs and faster-whisper is not installed. "
            "Install with the 'whisper' extra to enable transcription fallback."
        ) from e

    log.warning("Loading Whisper large-v3 — this will take a while…")
    model = await asyncio.to_thread(WhisperModel, "large-v3", device="auto", compute_type="auto")

    def _run() -> list[SubtitleEntry]:
        segments, _info = model.transcribe(str(file_path), language="en", vad_filter=True)
        out: list[SubtitleEntry] = []
        for seg in segments:
            text = _clean(seg.text)
            if not text:
                continue
            out.append(
                SubtitleEntry(
                    start_ms=int(seg.start * 1000),
                    end_ms=int(seg.end * 1000),
                    text=text,
                )
            )
        return out

    return await asyncio.to_thread(_run)


# ─── windowing ────────────────────────────────────────────────────────────


def windowize(entries: list[SubtitleEntry], window_seconds: int = 30) -> list[SubtitleWindow]:
    """Coalesce subtitle entries into ~window_seconds chunks for LLM input.

    Entries that fall entirely within the same window are concatenated. We
    don't split entries themselves — the boundary is whichever entry closes
    after the window deadline.
    """
    if not entries:
        return []

    window_ms = window_seconds * 1000
    windows: list[SubtitleWindow] = []
    cur_start = entries[0].start_ms
    cur_end = entries[0].end_ms
    cur_lines: list[str] = [entries[0].text]

    for entry in entries[1:]:
        if entry.start_ms - cur_start < window_ms:
            cur_lines.append(entry.text)
            cur_end = max(cur_end, entry.end_ms)
        else:
            windows.append(
                SubtitleWindow(start_ms=cur_start, end_ms=cur_end, text=" ".join(cur_lines))
            )
            cur_start = entry.start_ms
            cur_end = entry.end_ms
            cur_lines = [entry.text]

    windows.append(
        SubtitleWindow(start_ms=cur_start, end_ms=cur_end, text=" ".join(cur_lines))
    )
    return windows
