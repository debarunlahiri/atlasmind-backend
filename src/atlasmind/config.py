from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_STORAGE_ROOT = Path("/Volumes/Expansion/aiml/atlasmind")


class Settings(BaseSettings):
    """Configuration with all generated data rooted on the Expansion drive."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ATLASMIND_",
        extra="ignore",
    )

    environment: str = "development"
    storage_root: Path = DEFAULT_STORAGE_ROOT
    wikipedia_language: str = "en"
    wikipedia_user_agent: str = "AtlasMindCrawler/0.3 (educational project)"
    request_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    crawl_delay_seconds: float = Field(default=1.0, ge=0.5, le=60)
    postgres_dsn: str = "postgresql://debarunlahiri@localhost:5432/atlasmind"
    random_seed: int = 42
    compute_device: Literal["mps", "cpu", "auto"] = "mps"
    multimodal_model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    model_local_files_only: bool = False
    generator_max_context_characters: int = Field(default=4000, ge=250, le=50000)
    web_search_user_agent: str = "AtlasMindWebCrawler/0.7 (educational project)"
    web_search_timeout_seconds: float = Field(default=8.0, gt=0, le=30)
    web_search_max_workers: int = Field(default=5, ge=1, le=10)
    web_search_store_results: bool = True
    web_search_store_images: bool = False
    web_search_freshness: Literal["pd", "pw", "pm", "py"] = "pm"

    @property
    def corpus_dir(self) -> Path:
        return self.storage_root / "corpus"

    @property
    def model_dir(self) -> Path:
        return self.storage_root / "models"

    @property
    def index_dir(self) -> Path:
        return self.storage_root / "indexes"

    @property
    def cache_dir(self) -> Path:
        return self.storage_root / "cache"

    @property
    def huggingface_cache_dir(self) -> Path:
        return self.model_dir / "huggingface"


@lru_cache
def get_settings() -> Settings:
    return Settings()
