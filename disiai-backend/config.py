"""Central configuration for DISIAI, loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DISIAI"
    env: str = "local"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Text & implicit context extraction (Qwen 2.5 through Ollama).
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_api_key: str = "ollama"
    extractor_model: str = "qwen2.5:3b-instruct"
    extractor_use_llm: bool = True
    llm_timeout_seconds: float = 30.0

    # Local state.
    sqlite_path: str = "./disiai.db"

    # Voice input / output.
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    tts_voice: str = "id-ID-GadisNeural"
    tts_rate: str = "+0%"

    # Avatar.
    sadtalker_enabled: bool = False
    sadtalker_bin: str = ""
    sadtalker_ckpt: str = ""
    avatar_image_path: str = "./static/mock/avatar.png"
    mock_video_path: str = "./static/mock/avatar.mp4"

    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def db_path(self) -> Path:
        path = Path(self.sqlite_path)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def static_dir(self) -> Path:
        return BASE_DIR / "static"

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else BASE_DIR / path


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
