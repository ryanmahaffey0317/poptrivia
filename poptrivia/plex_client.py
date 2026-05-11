from __future__ import annotations

import logging
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import httpx

log = logging.getLogger("poptrivia.plex")


# Plex stream types:
#   1 = video, 2 = audio, 3 = subtitle


# Text-based subtitle codecs Plex reports. PGS / VOBSUB are image-based.
_TEXT_CODECS = {"srt", "subrip", "ass", "ssa", "mov_text", "webvtt", "vtt"}


@dataclass
class PlexSubtitleStream:
    stream_id: int
    codec: str
    language: str
    title: str
    is_default: bool


class PlexError(RuntimeError):
    pass


class PlexClient:
    """Minimal Plex Media Server client.

    Just enough to: find a movie by Plex GUID and pull subtitle streams.
    Plex auto-downloads SRTs (via its OpenSubtitles agent) and exposes them
    via the same `/library/streams/{id}` endpoint as embedded subs, so we
    don't need to care where they live on disk.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        client: httpx.AsyncClient | None = None,
    ):
        if not base_url or not token:
            raise ValueError("PlexClient requires base_url and token")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._client = client or httpx.AsyncClient(timeout=20.0)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def find_subtitle_streams(
        self, plex_guid: str
    ) -> list[PlexSubtitleStream]:
        """Return all subtitle streams for the movie matching `plex_guid`.

        English / forced-default streams come first.
        """
        rating_key = await self._resolve_rating_key(plex_guid)
        if rating_key is None:
            log.warning("Plex: no movie found for guid=%s", plex_guid)
            return []

        return await self._fetch_streams(rating_key)

    async def download_subtitle(self, stream_id: int) -> str:
        """Fetch the subtitle stream content (typically SRT text).

        Plex requires the format extension (`.srt`, `.vtt`) in the URL for
        externally-downloaded subtitles, otherwise it 501s. We try the
        extension forms first and fall back to the bare endpoint last
        (which works for some embedded subs).
        """
        last_error: str | None = None
        for url in (
            f"{self.base_url}/library/streams/{stream_id}.srt",
            f"{self.base_url}/library/streams/{stream_id}.vtt",
            f"{self.base_url}/library/streams/{stream_id}",
        ):
            try:
                r = await self._client.get(url, headers=self._auth_headers())
            except httpx.HTTPError as e:
                last_error = f"{type(e).__name__}: {e}"
                continue
            if r.status_code == 200 and r.text:
                log.debug("Plex stream %s downloaded via %s", stream_id, url)
                return r.text
            last_error = (
                f"{url.rsplit('/', 1)[-1]} -> HTTP {r.status_code}"
            )
        raise PlexError(
            f"All Plex stream download URLs failed for stream {stream_id} "
            f"(last: {last_error})"
        )

    # ─── internals ──────────────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        return {
            "X-Plex-Token": self.token,
            "Accept": "application/xml",
        }

    async def _resolve_rating_key(self, plex_guid: str) -> str | None:
        """Plex GUIDs (`plex://movie/...`) are findable via the All-libraries
        search endpoint. Returns the ratingKey of the matching item."""
        params = {"url": plex_guid}
        r = await self._client.get(
            f"{self.base_url}/library/metadata/matches",
            params=params,
            headers=self._auth_headers(),
        )
        if r.status_code == 200 and r.content:
            rk = _first_rating_key(r.text)
            if rk:
                return rk

        # Fall back: enumerate movie sections and search by guid.
        try:
            sections = await self._list_movie_sections()
        except Exception as e:
            log.warning("Plex section listing failed: %s", e)
            return None
        for section_key in sections:
            r = await self._client.get(
                f"{self.base_url}/library/sections/{section_key}/all",
                params={"guid": plex_guid},
                headers=self._auth_headers(),
            )
            if r.status_code == 200 and r.content:
                rk = _first_rating_key(r.text)
                if rk:
                    return rk
        return None

    async def _list_movie_sections(self) -> list[str]:
        r = await self._client.get(
            f"{self.base_url}/library/sections", headers=self._auth_headers()
        )
        if r.status_code != 200:
            return []
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            return []
        return [
            d.get("key", "")
            for d in root.findall(".//Directory")
            if d.get("type") == "movie" and d.get("key")
        ]

    async def _fetch_streams(self, rating_key: str) -> list[PlexSubtitleStream]:
        r = await self._client.get(
            f"{self.base_url}/library/metadata/{rating_key}",
            headers=self._auth_headers(),
        )
        if r.status_code != 200:
            log.warning(
                "Plex metadata HTTP %s for ratingKey=%s",
                r.status_code,
                rating_key,
            )
            return []
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError as e:
            log.warning("Plex metadata XML parse failed: %s", e)
            return []

        streams: list[PlexSubtitleStream] = []
        for stream_el in root.findall(".//Stream"):
            if stream_el.get("streamType") != "3":
                continue
            codec = (stream_el.get("codec") or "").lower()
            streams.append(
                PlexSubtitleStream(
                    stream_id=int(stream_el.get("id") or 0),
                    codec=codec,
                    language=(stream_el.get("language") or "").lower(),
                    title=(stream_el.get("title") or ""),
                    is_default=stream_el.get("default") == "1",
                )
            )

        # Order: text codecs first, English first, default first, then by id.
        def _rank(s: PlexSubtitleStream) -> tuple[int, int, int, int]:
            is_text = 0 if s.codec in _TEXT_CODECS else 1
            is_eng = 0 if s.language in ("eng", "en", "english") else 1
            is_default = 0 if s.is_default else 1
            return (is_text, is_eng, is_default, s.stream_id)

        streams.sort(key=_rank)
        return streams


def _first_rating_key(xml_text: str) -> str | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    for el in root.iter():
        rk = el.get("ratingKey")
        if rk:
            return rk
    return None
