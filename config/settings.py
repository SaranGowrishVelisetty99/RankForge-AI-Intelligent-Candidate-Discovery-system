from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openrouter_api_key: str = ""
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_timeout_seconds: int = 60
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 2.0

    ranker_weight_mode: str = "weighted"

    log_level: str = "INFO"


settings = Settings()
