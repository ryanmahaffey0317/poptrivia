from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from poptrivia.models import JobStatus, Movie, MovieStatus, PrepJob

log = logging.getLogger("poptrivia.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    plex_guid     TEXT PRIMARY KEY,
    imdb_id       TEXT,
    tmdb_id       TEXT,
    title         TEXT NOT NULL,
    year          INTEGER,
    file_path     TEXT,
    status        TEXT NOT NULL,
    track_path    TEXT,
    error_message TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prep_jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    plex_guid   TEXT NOT NULL,
    status      TEXT NOT NULL,
    attempts    INTEGER DEFAULT 0,
    last_error  TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    FOREIGN KEY (plex_guid) REFERENCES movies(plex_guid)
);

CREATE INDEX IF NOT EXISTS idx_prep_jobs_status ON prep_jobs(status, id);
"""


class Database:
    """Thin async wrapper around an aiosqlite connection.

    One connection, one writer at a time. SQLite's WAL mode is enabled so the
    queue worker and the request handlers don't trip on each other.
    """

    def __init__(self, path: Path):
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        log.info("SQLite ready at %s", self.path)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database.connect() was not called")
        return self._conn

    # ─── movies ─────────────────────────────────────────────────────

    async def get_movie(self, plex_guid: str) -> Movie | None:
        async with self.conn.execute(
            "SELECT * FROM movies WHERE plex_guid = ?", (plex_guid,)
        ) as cur:
            row = await cur.fetchone()
        return _row_to_movie(row) if row else None

    async def list_movies(self) -> list[Movie]:
        """Return every movie row. Cheap at our scale; we have at most ~hundreds."""
        async with self.conn.execute("SELECT * FROM movies") as cur:
            rows = await cur.fetchall()
        return [_row_to_movie(r) for r in rows]

    async def upsert_movie(
        self,
        *,
        plex_guid: str,
        title: str,
        year: int | None = None,
        imdb_id: str | None = None,
        tmdb_id: str | None = None,
        file_path: str | None = None,
        status: MovieStatus | None = None,
        track_path: str | None = None,
        error_message: str | None = None,
    ) -> Movie:
        existing = await self.get_movie(plex_guid)
        if existing is None:
            await self.conn.execute(
                """
                INSERT INTO movies
                    (plex_guid, imdb_id, tmdb_id, title, year, file_path,
                     status, track_path, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plex_guid,
                    imdb_id,
                    tmdb_id,
                    title,
                    year,
                    file_path,
                    (status or MovieStatus.NOT_STARTED).value,
                    track_path,
                    error_message,
                ),
            )
        else:
            # Only overwrite optional fields when callers actually pass them;
            # this lets a webhook update title/file without clobbering status.
            updates: dict[str, Any] = {
                "title": title,
                "imdb_id": imdb_id if imdb_id is not None else existing.imdb_id,
                "tmdb_id": tmdb_id if tmdb_id is not None else existing.tmdb_id,
                "year": year if year is not None else existing.year,
                "file_path": file_path if file_path is not None else existing.file_path,
                "status": (status.value if status is not None else existing.status.value),
                "track_path": track_path if track_path is not None else existing.track_path,
                "error_message": (
                    error_message if error_message is not None else existing.error_message
                ),
            }
            await self.conn.execute(
                """
                UPDATE movies
                   SET title=?, imdb_id=?, tmdb_id=?, year=?, file_path=?,
                       status=?, track_path=?, error_message=?,
                       updated_at=CURRENT_TIMESTAMP
                 WHERE plex_guid=?
                """,
                (
                    updates["title"],
                    updates["imdb_id"],
                    updates["tmdb_id"],
                    updates["year"],
                    updates["file_path"],
                    updates["status"],
                    updates["track_path"],
                    updates["error_message"],
                    plex_guid,
                ),
            )
        await self.conn.commit()
        movie = await self.get_movie(plex_guid)
        assert movie is not None
        return movie

    async def set_movie_status(
        self,
        plex_guid: str,
        status: MovieStatus,
        *,
        track_path: str | None = None,
        error_message: str | None = None,
    ) -> None:
        await self.conn.execute(
            """
            UPDATE movies
               SET status=?,
                   track_path=COALESCE(?, track_path),
                   error_message=?,
                   updated_at=CURRENT_TIMESTAMP
             WHERE plex_guid=?
            """,
            (status.value, track_path, error_message, plex_guid),
        )
        await self.conn.commit()

    # ─── prep_jobs ──────────────────────────────────────────────────

    async def enqueue_job(self, plex_guid: str) -> int:
        cur = await self.conn.execute(
            "INSERT INTO prep_jobs (plex_guid, status) VALUES (?, ?)",
            (plex_guid, JobStatus.PENDING.value),
        )
        await self.conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    async def has_active_job(self, plex_guid: str) -> bool:
        async with self.conn.execute(
            """
            SELECT 1 FROM prep_jobs
             WHERE plex_guid=? AND status IN (?, ?)
             LIMIT 1
            """,
            (plex_guid, JobStatus.PENDING.value, JobStatus.RUNNING.value),
        ) as cur:
            row = await cur.fetchone()
        return row is not None

    async def next_pending_job(self) -> PrepJob | None:
        async with self.conn.execute(
            "SELECT * FROM prep_jobs WHERE status=? ORDER BY id LIMIT 1",
            (JobStatus.PENDING.value,),
        ) as cur:
            row = await cur.fetchone()
        return _row_to_job(row) if row else None

    async def mark_job_running(self, job_id: int) -> None:
        await self.conn.execute(
            """
            UPDATE prep_jobs
               SET status=?, attempts=attempts+1, started_at=CURRENT_TIMESTAMP
             WHERE id=?
            """,
            (JobStatus.RUNNING.value, job_id),
        )
        await self.conn.commit()

    async def mark_job_succeeded(self, job_id: int) -> None:
        await self.conn.execute(
            """
            UPDATE prep_jobs
               SET status=?, finished_at=CURRENT_TIMESTAMP, last_error=NULL
             WHERE id=?
            """,
            (JobStatus.SUCCEEDED.value, job_id),
        )
        await self.conn.commit()

    async def mark_job_failed(self, job_id: int, error: str) -> None:
        await self.conn.execute(
            """
            UPDATE prep_jobs
               SET status=?, finished_at=CURRENT_TIMESTAMP, last_error=?
             WHERE id=?
            """,
            (JobStatus.FAILED.value, job_id, error),
        )
        await self.conn.commit()

    async def requeue_job(self, job_id: int, error: str) -> None:
        """Push the job back to PENDING so the worker retries on next tick."""
        await self.conn.execute(
            """
            UPDATE prep_jobs
               SET status=?, last_error=?
             WHERE id=?
            """,
            (JobStatus.PENDING.value, error, job_id),
        )
        await self.conn.commit()


# ─── row -> model ──────────────────────────────────────────────────────────


def _row_to_movie(row: aiosqlite.Row) -> Movie:
    return Movie(
        plex_guid=row["plex_guid"],
        imdb_id=row["imdb_id"],
        tmdb_id=row["tmdb_id"],
        title=row["title"],
        year=row["year"],
        file_path=row["file_path"],
        status=MovieStatus(row["status"]),
        track_path=row["track_path"],
        error_message=row["error_message"],
        created_at=_parse_ts(row["created_at"]),
        updated_at=_parse_ts(row["updated_at"]),
    )


def _row_to_job(row: aiosqlite.Row) -> PrepJob:
    return PrepJob(
        id=row["id"],
        plex_guid=row["plex_guid"],
        status=JobStatus(row["status"]),
        attempts=row["attempts"] or 0,
        last_error=row["last_error"],
        created_at=_parse_ts(row["created_at"]),
        started_at=_parse_ts(row["started_at"]),
        finished_at=_parse_ts(row["finished_at"]),
    )


def _parse_ts(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v).replace(" ", "T"))
    except ValueError:
        return None
