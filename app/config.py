"""Application configuration: the single place that reads environment variables."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed settings. Environment variable names are unchanged from earlier versions."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    # --- Models -----------------------------------------------------------------
    google_api_key: SecretStr | None = Field(None, validation_alias="GOOGLE_API_KEY")
    chat_model: str = Field("gemini-3.5-flash-lite", validation_alias="RAG_MODEL")
    embedding_model: str = Field("gemini-embedding-2-preview", validation_alias="RAG_EMBEDDING_MODEL")
    evaluation_model: str | None = Field(None, validation_alias="RAG_EVALUATION_MODEL")
    reranker_model: str = Field("BAAI/bge-reranker-base", validation_alias="RAG_RERANKER_MODEL")
    max_output_tokens: int = Field(512, gt=0)

    # --- LLM resilience -----------------------------------------------------------
    llm_max_attempts: int = Field(3, ge=1)
    llm_timeout_s: float = Field(30.0, gt=0)
    llm_retry_base_delay_s: float = Field(0.5, ge=0)
    llm_retry_max_delay_s: float = Field(4.0, ge=0)
    chat_timeout_s: float = Field(90.0, gt=0)

    # --- Retrieval ------------------------------------------------------------------
    vector_k: int = Field(8, gt=0)
    lexical_k: int = Field(8, gt=0)
    final_k: int = Field(5, gt=0)

    # --- Conversation memory ----------------------------------------------------------
    history_max_turns: int = Field(6, gt=0)
    history_max_chars: int = Field(2000, gt=0)
    rewrite_history_turns: int = Field(2, gt=0)
    rewrite_max_chars: int = Field(400, gt=0)

    # --- Storage & ingestion ---------------------------------------------------------
    database_path: Path = Field(PROJECT_ROOT / "data" / "app.sqlite3", validation_alias="RAG_DATABASE_PATH")
    upload_dir: Path = Field(PROJECT_ROOT / "data" / "uploads", validation_alias="RAG_UPLOAD_DIR")
    max_upload_bytes: int = Field(25 * 1024 * 1024, gt=0)
    max_sources_per_request: int = Field(20, gt=0)
    session_ttl_hours: float | None = Field(None, gt=0, validation_alias="RAG_SESSION_TTL_HOURS")

    # --- Evaluation --------------------------------------------------------------------
    evaluation_concurrency: int = Field(2, ge=1)
    shutdown_grace_s: float = Field(10.0, ge=0)

    # --- Observability -------------------------------------------------------------------
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")
    langfuse_public_key: SecretStr | None = Field(None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(None, validation_alias="LANGFUSE_SECRET_KEY")

    @property
    def judge_model(self) -> str:
        return self.evaluation_model or self.chat_model

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load `.env` into the process environment once (the Google, Langfuse and DeepEval
    SDKs read their credentials from there), then build the typed settings."""
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings()
