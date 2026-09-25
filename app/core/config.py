from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "AccessLearn AI"
    environment: str = "development"

    # Storage
    upload_dir: str = "./data/uploads"
    output_dir: str = "./data/outputs"
    max_upload_size_mb: int = 50

    # Database
    database_url: str = "sqlite:///./accesslearn.db"

    # AI — Google Gemini (primary)
    gemini_api_key: str = ""
    # Kept for backward compatibility; AIClient will fall back to this if
    # gemini_api_key is empty (legacy Anthropic setups).
    anthropic_api_key: str = ""
    # Default Gemini model. Override via AI_MODEL in .env if needed
    # (e.g. gemini-1.5-flash, gemini-2.0-flash, gemini-2.5-flash).
    ai_model: str = "gemini-3.8-flash"

    # OCR
    ocr_confidence_threshold: float = 0.75

    # Retention (Responsible AI requirement: limited storage duration)
    file_retention_days: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
