from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import pysrt

log = logging.getLogger("poptrivia.prep.subtitles")


@dataclass
class SubtitleEntry:
    start_ms: int
    end_ms: int
    text: str


@dataclass
class SubtitleWindow:
    start_ms: int
    end_ms: int
    text: str


class SubtitleExtractError(RuntimeError):
    pass


# Subtitle codecs ffmpeg can convert directly to SRT. Anything else
# (PGS/VOBSUB/DVB) is image-based and would require OCR.
_TEXT_SUB_CODECS: set[str] = {
    "subrip",
    "srt",
    "ass",
    "ssa",
    "mov_text",
    "webvtt",
    "text",
    "microdvd",
    "jacosub",
    "realtext",
    "subviewer",
}

_FFMPEG_TIMEOUT_SECONDS = 60
_FFPROBE_TIMEOUT_SECONDS = 20
_AUDIO_EXTRACT_TIMEOUT_SECONDS = 600  # ~10 min ceiling for audio rip


async def extract(
    file_path: Path,
    *,
    whisper_url: str = "",
    whisper_model: str = "Systran/faster-whisper-large-v3",
    whisper_timeout: int = 3600,
) -> list[SubtitleEntry]:
    """Get subtitles for a movie file.

    Strategy in order:
      1. ffprobe the file. Skip image-based subtitle streams (PGS/VOBSUB)
         and try only text streams via ffmpeg.
      2. Sidecar .srt next to the file.
      3. Remote Whisper (if `whisper_url` is set) — extracts low-bitrate
         audio and POSTs it to an OpenAI-compatible /v1/audio/transcriptions
         endpoint.
      4. Hard error.
    """
    if not file_path.exists():
        raise SubtitleExtractError(f"Media file not found: {file_path}")

    embedded = await _try_ffmpeg(file_path)
    if embedded:
        log.info("Subtitles via embedded text stream (%d entries)", len(embedded))
        return embedded

    sidecar = _try_sidecar(file_path)
    if sidecar:
        log.info("Subtitles via sidecar SRT (%d entries)", len(sidecar))
        return sidecar

    if whisper_url:
        log.warning(
            "No text subs in %s — falling back to remote Whisper at %s",
            file_path.name,
            whisper_url,
        )
        return await _try_remote_whisper(
            file_path,
            whisper_url=whisper_url,
            model=whisper_model,
            timeout=whisper_timeout,
        )

    raise SubtitleExtractError(
        f"No text subs in {file_path.name} and WHISPER_URL is not set. "
        "Either drop a sidecar .srt next to the file or configure a remote "
        "Whisper service in .env."
    )


# ─── ffmpeg (text-stream extraction) ──────────────────────────────────────


async def _try_ffmpeg(file_path: Path) -> list[SubtitleEntry] | None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        log.warning("ffmpeg/ffprobe not on PATH; skipping embedded subs")
        return None

    text_streams = await _probe_text_subtitle_streams(file_path)
    if not text_streams:
        log.info("ffprobe found no text-based subtitle streams in %s", file_path.name)
        return None

    log.info(
        "ffprobe found %d text subtitle stream(s) at indices %s",
        len(text_streams),
        text_streams,
    )

    for stream_index in text_streams:
        log.info("Extracting embedded subtitle stream 0:s:%d", stream_index)
        out = await _run_ffmpeg_to_srt(file_path, stream_index)
        if out is None:
            continue
        entries = _parse_srt_text(out)
        if entries:
            return entries
    return None


async def _probe_text_subtitle_streams(file_path: Path) -> list[int]:
    """Return subtitle-namespace indices of text-based streams (English preferred).

    ffprobe gives us absolute stream indices via `stream=index`, but ffmpeg's
    `-map 0:s:N` uses an index within the subtitle namespace. We track both
    and return the subtitle-namespace index.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "s",
        "-show_entries",
        "stream=index,codec_name:stream_tags=language,title",
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
        stdout, _stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_FFPROBE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        log.warning("ffprobe timed out for %s", file_path)
        return []

    if proc.returncode != 0 or not stdout:
        log.warning("ffprobe failed (rc=%s) for %s", proc.returncode, file_path)
        return []

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        log.warning("ffprobe returned non-JSON: %s", e)
        return []

    streams = data.get("streams") or []
    # Build (sub_namespace_index, codec_name, language, title) tuples.
    enriched: list[tuple[int, str, str, str]] = []
    for sub_idx, s in enumerate(streams):
        codec = (s.get("codec_name") or "").lower()
        tags = s.get("tags") or {}
        lang = (tags.get("language") or "").lower()
        title = (tags.get("title") or "").lower()
        enriched.append((sub_idx, codec, lang, title))

    # Drop image-based codecs. Keep text codecs.
    text_streams = [
        (idx, codec, lang, title)
        for (idx, codec, lang, title) in enriched
        if codec in _TEXT_SUB_CODECS
    ]
    if not text_streams:
        return []

    # Prefer English non-commentary streams.
    def _rank(item: tuple[int, str, str, str]) -> tuple[int, int]:
        _idx, _codec, lang, title = item
        eng = 0 if lang in ("eng", "en", "english") else 1
        commentary = 1 if "commentary" in title or "sdh" in title else 0
        return (eng, commentary)

    text_streams.sort(key=_rank)
    return [idx for (idx, _codec, _lang, _title) in text_streams]


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
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_FFMPEG_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        log.warning(
            "ffmpeg stream %d timed out after %ds — killing",
            stream_index,
            _FFMPEG_TIMEOUT_SECONDS,
        )
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return None
    if proc.returncode != 0 or not stdout:
        log.info("ffmpeg stream %d failed (rc=%s)", stream_index, proc.returncode)
        log.debug(
            "ffmpeg stream %d stderr: %s",
            stream_index,
            stderr.decode("utf-8", errors="replace")[:500],
        )
        return None
    log.info("ffmpeg stream %d extracted (%d bytes)", stream_index, len(stdout))
    return stdout.decode("utf-8", errors="replace")


# ─── sidecar SRT ──────────────────────────────────────────────────────────


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
    except Exception as e:
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
_BRACKETS_RE = re.compile(r"\[[^\]]+\]|\([A-Z ]{3,}\)")


def _clean(text: str) -> str:
    text = text.replace("\n", " ")
    text = _BRACKETS_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


# ─── remote Whisper fallback ──────────────────────────────────────────────


async def _try_remote_whisper(
    file_path: Path,
    *,
    whisper_url: str,
    model: str,
    timeout: int,
) -> list[SubtitleEntry]:
    """Extract a low-bitrate audio file and POST to an OpenAI-compatible
    /v1/audio/transcriptions endpoint. Speaches and faster-whisper-server
    both implement this."""
    audio_path = await _extract_audio_for_whisper(file_path)
    try:
        log.info(
            "Whisper: uploading %d MB of audio to %s (model=%s)",
            audio_path.stat().st_size // 1_000_000,
            whisper_url,
            model,
        )
        srt_text = await _post_audio_to_whisper(
            audio_path, whisper_url=whisper_url, model=model, timeout=timeout
        )
    finally:
        audio_path.unlink(missing_ok=True)

    entries = _parse_srt_text(srt_text)
    if not entries:
        raise SubtitleExtractError(
            "Remote Whisper returned 0 subtitle entries — check the server logs."
        )
    log.info("Whisper transcription complete: %d entries", len(entries))
    return entries


async def _extract_audio_for_whisper(file_path: Path) -> Path:
    """Rip the first audio track as 24kbps mono Opus. Whisper is fine with
    aggressive compression at speech bandwidth, and 24k Opus keeps a 2-hour
    movie under ~25 MB for the upload."""
    out_path = Path(
        tempfile.mktemp(prefix="poptrivia_", suffix=".opus", dir="/tmp")
    )
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(file_path),
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libopus",
        "-b:a",
        "24k",
        str(out_path),
    ]
    log.info("Extracting audio for Whisper from %s", file_path.name)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_AUDIO_EXTRACT_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        out_path.unlink(missing_ok=True)
        raise SubtitleExtractError(
            f"Audio extraction for Whisper timed out after "
            f"{_AUDIO_EXTRACT_TIMEOUT_SECONDS}s"
        )
    if proc.returncode != 0:
        out_path.unlink(missing_ok=True)
        raise SubtitleExtractError(
            f"Audio extraction failed (rc={proc.returncode}): "
            f"{stderr.decode('utf-8', errors='replace')[:500]}"
        )
    log.info(
        "Audio extracted: %s (%d bytes)", out_path, out_path.stat().st_size
    )
    return out_path


async def _post_audio_to_whisper(
    audio_path: Path, *, whisper_url: str, model: str, timeout: int
) -> str:
    base = whisper_url.rstrip("/")
    url = f"{base}/v1/audio/transcriptions"
    audio_bytes = audio_path.read_bytes()

    files = {"file": (audio_path.name, audio_bytes, "audio/opus")}
    data = {
        "model": model,
        "response_format": "srt",
        "language": "en",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as c:
        r = await c.post(url, files=files, data=data)
    if r.status_code != 200:
        raise SubtitleExtractError(
            f"Whisper HTTP {r.status_code}: {r.text[:500]}"
        )
    return r.text


# ─── windowing (unchanged) ────────────────────────────────────────────────


def windowize(entries: list[SubtitleEntry], window_seconds: int = 30) -> list[SubtitleWindow]:
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
