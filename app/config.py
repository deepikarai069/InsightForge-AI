"""Central configuration, read from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    env = Path(".env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split("#")[0].strip()
        os.environ.setdefault(k.strip(), v)


_load_dotenv()


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./data/insightforge.db"))
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "none").lower())
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-4-5"))
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "./data/chroma"))
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "hashing").lower())
    max_result_rows: int = field(default_factory=lambda: int(os.getenv("MAX_RESULT_ROWS", "1000")))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "100")))


def get_settings() -> Settings:
    return Settings()
