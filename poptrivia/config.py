from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Plex / Tautulli
    plex_url: str = ""
    plex_token: str = ""
    tautulli_url: str
    tautulli_api_key: str
    monitored_users: Annotated[set[str], NoDecode] = Field(default_factory=set)

    # Ollama
    ollama_url: str
    ollama_model: str = "qwen3:32b"
    ollama_timeout: int = 600

    # Remote Whisper service (e.g. Speaches / faster-whisper-server on a
    # GPU host). Used as the fallback when neither embedded subs nor a
    # sidecar SRT are available. Leave blank to disable — prep then fails
    # cleanly for any movie that has only image-based subs.
    whisper_url: str = ""
    whisper_model: str = "Systran/faster-whisper-large-v3"
    whisper_timeout: int = 3600

    # Discord
    discord_webhook_url: str
    discord_system_webhook_url: str = ""

    # Source material
    tmdb_api_key: str = ""

    # Playback timing
    card_lead_time_seconds: int = 7
    session_poll_interval_seconds: int = 3
    missed_card_threshold_seconds: int = 30

    # Prep pipeline
    target_cards_per_movie: int = 50
    max_cards_per_movie: int = 70

    # Container paths
    config_dir: Path = Path("/config")
    media_dir: Path = Path("/media")
    tracks_dir: Path = Path("/config/tracks")
    cache_dir: Path = Path("/config/cache")

    # Curated tracking
    tracked_list_path: Path = Path("/config/tracked.txt")
    tracked_poll_interval_seconds: int = 1800  # 30 minutes

    # Logging
    log_level: str = "INFO"

    @field_validator("monitored_users", mode="before")
    @classmethod
    def _split_users(cls, v: object) -> object:
        if isinstance(v, str):
            return {part.strip() for part in v.split(",") if part.strip()}
        return v

    @field_validator("tautulli_url", "ollama_url", "whisper_url", mode="before")
    @classmethod
    def _ensure_protocol(cls, v: object) -> object:
        """Prepend http:// if a bare host:port was provided.

        Easy footgun that wastes minutes of pipeline time before any LLM
        call fails with 'missing protocol'. We normalize at load time.
        Empty strings pass through (whisper_url is optional).
        """
        if isinstance(v, str) and v and "://" not in v:
            return f"http://{v}"
        return v

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()

    @property
    def system_webhook_url(self) -> str:
        return self.discord_system_webhook_url or self.discord_webhook_url

    @property
    def db_path(self) -> Path:
        return self.config_dir / "poptrivia.db"

    def ensure_dirs(self) -> None:
        for d in (self.config_dir, self.tracks_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
