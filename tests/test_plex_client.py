from __future__ import annotations

import httpx
import pytest

from poptrivia.plex_client import PlexClient, PlexSubtitleStream, _TEXT_CODECS


_SECTIONS_XML = """<?xml version="1.0"?>
<MediaContainer>
  <Directory key="1" type="movie" title="Movies"/>
  <Directory key="2" type="show" title="TV"/>
  <Directory key="3" type="movie" title="4K Movies"/>
</MediaContainer>"""


_METADATA_XML = """<?xml version="1.0"?>
<MediaContainer>
  <Video ratingKey="42" type="movie" title="The Shining">
    <Media>
      <Part>
        <Stream id="100" streamType="1" codec="hevc"/>
        <Stream id="101" streamType="2" codec="eac3" language="eng"/>
        <Stream id="102" streamType="3" codec="hdmv_pgs_subtitle" language="eng" default="1"/>
        <Stream id="103" streamType="3" codec="hdmv_pgs_subtitle" language="fra"/>
        <Stream id="104" streamType="3" codec="srt" language="eng" title="Downloaded by Plex"/>
        <Stream id="105" streamType="3" codec="srt" language="eng" title="commentary"/>
      </Part>
    </Media>
  </Video>
</MediaContainer>"""


_MATCHES_NOT_FOUND_XML = """<?xml version="1.0"?>
<MediaContainer size="0"/>"""

_SEARCH_FOUND_XML = """<?xml version="1.0"?>
<MediaContainer>
  <Video ratingKey="42" guid="plex://movie/abc"/>
</MediaContainer>"""


def _make_client(handler) -> PlexClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return PlexClient("http://plex:32400", "TOKEN", client=http)


async def test_find_subtitle_streams_ranks_text_first_then_english() -> None:
    """Text codecs come before image; English before other languages."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/library/metadata/matches":
            return httpx.Response(200, text=_MATCHES_NOT_FOUND_XML)
        if path == "/library/sections":
            return httpx.Response(200, text=_SECTIONS_XML)
        if path.startswith("/library/sections/") and path.endswith("/all"):
            return httpx.Response(200, text=_SEARCH_FOUND_XML)
        if path == "/library/metadata/42":
            return httpx.Response(200, text=_METADATA_XML)
        return httpx.Response(404)

    client = _make_client(handler)
    try:
        streams = await client.find_subtitle_streams("plex://movie/abc")
    finally:
        await client.aclose()

    # First two should be the SRT (text codec), English. The PGS streams
    # come after the text streams.
    assert len(streams) == 4
    assert all(isinstance(s, PlexSubtitleStream) for s in streams)
    assert streams[0].codec == "srt"
    assert streams[0].language == "eng"
    # PGS streams are pushed back even though one was default=1.
    assert streams[-1].codec == "hdmv_pgs_subtitle"


async def test_download_subtitle_returns_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/library/streams/104":
            assert request.headers["X-Plex-Token"] == "TOKEN"
            return httpx.Response(
                200,
                text="1\n00:00:01,000 --> 00:00:02,000\nHello\n",
            )
        return httpx.Response(404)

    client = _make_client(handler)
    try:
        text = await client.download_subtitle(104)
    finally:
        await client.aclose()

    assert "Hello" in text


async def test_no_movie_match_returns_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/library/metadata/matches":
            return httpx.Response(200, text=_MATCHES_NOT_FOUND_XML)
        if path == "/library/sections":
            return httpx.Response(200, text=_SECTIONS_XML)
        if path.startswith("/library/sections/") and path.endswith("/all"):
            return httpx.Response(200, text=_MATCHES_NOT_FOUND_XML)
        return httpx.Response(404)

    client = _make_client(handler)
    try:
        streams = await client.find_subtitle_streams("plex://movie/unknown")
    finally:
        await client.aclose()
    assert streams == []


def test_text_codecs_includes_expected_set() -> None:
    assert "srt" in _TEXT_CODECS
    assert "subrip" in _TEXT_CODECS
    assert "ass" in _TEXT_CODECS
    assert "hdmv_pgs_subtitle" not in _TEXT_CODECS
