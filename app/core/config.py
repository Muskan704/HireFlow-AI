from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # ── LLM provider 
    # Set ACTIVE_LLM to "anthropic", "openai", or "groq"
    active_llm: str = Field(default="groq", env="ACTIVE_LLM")

    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-6", env="ANTHROPIC_MODEL")

    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")

    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", env="GROQ_MODEL")

    ollama_model: str = Field(default="gemma3:4b",env="OLLAMA_MODEL")
    ollama_host: str = Field(default="http://localhost:11434",env="OLLAMA_HOST")

    gemini_api_key: str = Field(default="",env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash",env="GEMINI_MODEL")

    # ── Embeddings 
    # "openai" uses text-embedding-3-small via OpenAI API (costs money)
    # "local"  uses all-MiniLM-L6-v2 via sentence-transformers (free, runs locally)
    embedding_provider: str = Field(default="local", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    # ── ChromaDB 
    chroma_persist_dir: str = Field(default="./data/chroma", env="CHROMA_PERSIST_DIR")

    # ── Stage 4 experiment mode 
    # "semantic" = Path A only (cosine similarity)
    # "llm"      = Path B only (LLM scores each section 0-10)
    # "both"     = run both, log both, compare
    similarity_mode: str = Field(default="both", env="SIMILARITY_MODE")

    # ── Persistence (M1) ──────────────────────────────────────────────────────
    database_url: str = Field(default="", env="DATABASE_URL")

    # ── App ────────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_resume_size_mb: int = Field(default=10, env="MAX_RESUME_SIZE_MB")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()