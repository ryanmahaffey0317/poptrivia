#!/usr/bin/env python3
"""Pretty-print a generated trivia track in the terminal.

Usage:
    python scripts/inspect_track.py plex://movie/abc
    python scripts/inspect_track.py /config/tracks/plex___movie_abc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poptrivia.config import get_settings  # noqa: E402
from poptrivia.discord_client import pretty_category  # noqa: E402


# 256-color ANSI codes that vaguely match the Discord category palette.
_CATEGORY_ANSI = {
    "production": 36,         # cyan
    "casting": 161,           # pink
    "cinematography": 99,     # purple
    "historical_context": 130,  # brown/orange
    "easter_egg": 220,        # gold
    "cultural_impact": 196,   # red
    "goof": 208,              # orange
    "cut_content": 67,        # slate
    "cast_biography": 135,    # violet
    "score_music": 35,        # green
}


def _color(code: int, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\x1b[38;5;{code}m{text}\x1b[0m"


def _fmt_ts(ms: int) -> str:
    total_sec, _ = divmod(ms, 1000)
    minutes, seconds = divmod(total_sec, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a generated trivia track.")
    parser.add_argument("target", help="Plex GUID or path to the track JSON file")
    args = parser.parse_args()

    path = _resolve_path(args.target)
    if path is None:
        print(f"Could not find track for {args.target!r}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    title = data.get("title", "?")
    year = data.get("year")
    cards = data.get("cards") or []

    header = f"{title}" + (f" ({year})" if year else "")
    print(f"\n{header}")
    print(f"track: {path}")
    print(f"model: {data.get('model', '?')}")
    print(f"cards: {len(cards)}")
    print()

    for c in cards:
        ts = _fmt_ts(c.get("timestamp_ms", 0))
        cat = c.get("category", "")
        cat_label = pretty_category(cat)
        ansi = _CATEGORY_ANSI.get(cat, 244)
        interest = c.get("interest_level", 0)
        marker = "★" * interest + "·" * (5 - interest)
        cid = c.get("id", "?")
        text = c.get("text", "")
        line = (
            f"  {_color(244, ts)}  "
            f"{_color(ansi, cat_label.ljust(20))} "
            f"{_color(244, marker)}  "
            f"{_color(244, cid)}"
        )
        print(line)
        print(f"      {text}")
        anchor = c.get("anchor_evidence") or ""
        if anchor:
            print(_color(244, f"      ↳ {anchor}"))
        print()

    return 0


def _resolve_path(target: str) -> Path | None:
    p = Path(target)
    if p.exists():
        return p
    settings = get_settings()
    safe = target.replace("/", "_").replace(":", "_")
    candidate = settings.tracks_dir / f"{safe}.json"
    return candidate if candidate.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
