from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ─── Tautulli webhook ─────────────────────────────────────────────────────


class TautulliEvent(BaseModel):
    """Parsed Tautulli webhook payload.

    The handler accepts a few field-name variants because Tautulli has
    renamed variables across versions. Anything we genuinely depend on is
    required; everything else is best-effort.
    """

    event: str
    username: str
    media_type: str
    title: str
    year: int | None = None
    imdb_id: str | None = None
    tmdb_id: str | None = None
    plex_guid: str
    file: str | None = None
    duration_ms: int | None = None
    session_key: str
    view_offset_ms: int = 0

    @field_validator("year", mode="before")
    @classmethod
    def _year_to_int(cls, v: Any) -> Any:
        if v in (None, "", "None"):
            return None
        return int(v)

    @field_validator("duration_ms", mode="before")
    @classmethod
    def _duration_to_int(cls, v: Any) -> Any:
        if v in (None, "", "None"):
            return None
        return int(v)

    @field_validator("view_offset_ms", mode="before")
    @classmethod
    def _view_offset_to_int(cls, v: Any) -> Any:
        if v in (None, "", "None"):
            return 0
        return int(v)

    @field_validator("session_key", mode="before")
    @classmethod
    def _session_key_to_str(cls, v: Any) -> Any:
        return str(v) if v is not None else v

    @field_validator("imdb_id", "tmdb_id", mode="before")
    @classmethod
    def _strip_external_id(cls, v: Any) -> Any:
        if v in (None, "", "None"):
            return None
        return str(v).strip()

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "TautulliEvent":
        """Build from a raw webhook dict, tolerating common field-name variants."""

        def pick(*names: str) -> Any:
            for n in names:
                if n in raw and raw[n] not in (None, ""):
                    return raw[n]
            return None

        normalized = {
            "event": pick("event", "action") or "",
            "username": pick("username", "user") or "",
            "media_type": pick("media_type", "mediaType") or "",
            "title": pick("title", "movie_name", "full_title") or "",
            "year": pick("year"),
            "imdb_id": pick("imdb_id", "imdbId"),
            "tmdb_id": pick("tmdb_id", "themoviedb_id", "tmdbId"),
            "plex_guid": pick("plex_guid", "guid") or "",
            "file": pick("file", "filename"),
            "duration_ms": pick("duration_ms", "duration"),
            "session_key": pick("session_key", "sessionKey") or "",
            "view_offset_ms": pick("view_offset_ms", "view_offset", "viewOffset"),
        }
        return cls.model_validate(normalized)


# ─── Movie / prep status ──────────────────────────────────────────────────


class MovieStatus(StrEnum):
    NOT_STARTED = "not_started"
    QUEUED = "queued"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Movie(BaseModel):
    plex_guid: str
    imdb_id: str | None = None
    tmdb_id: str | None = None
    title: str
    year: int | None = None
    file_path: str | None = None
    status: MovieStatus = MovieStatus.NOT_STARTED
    track_path: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class PrepJob(BaseModel):
    id: int
    plex_guid: str
    status: JobStatus
    attempts: int = 0
    last_error: str | None = None
    manual_sources: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


# ─── Trivia data ──────────────────────────────────────────────────────────


Category = Literal[
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
]


class RawFactExtracted(BaseModel):
    """Output of stage 1 — a single fact pulled from source material."""

    fact: str
    source: Literal["imdb_trivia", "imdb_goofs", "wikipedia"]
    specificity: Literal["high", "medium", "low"]
    anchors: list[str] = Field(default_factory=list)
    category: Category


class RawSourceItem(BaseModel):
    """A pre-LLM chunk of scraped source material.

    Each scraper produces a list of these. Stage 1 walks them and produces
    RawFactExtracted entries.
    """

    source: Literal["imdb_trivia", "imdb_goofs", "wikipedia"]
    section: str = ""  # e.g. wikipedia section name; empty for IMDB items
    text: str


class TriviaCard(BaseModel):
    """Output of stage 2 — a fact placed at a specific timestamp."""

    id: str
    timestamp_ms: int
    text: str
    category: Category
    interest_level: int = Field(ge=1, le=5)
    source_fact_id: str
    anchor_evidence: str = ""


class TrackFile(BaseModel):
    """JSON blob written to {TRACKS_DIR}/{guid_safe}.json."""

    plex_guid: str
    imdb_id: str | None = None
    tmdb_id: str | None = None
    title: str
    year: int | None = None
    generated_at: datetime
    model: str
    cards: list[TriviaCard]
