# app/config.py
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Identidade / App
    SERVICE_NAME: str = "svc-face-recon"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Supabase
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_ANON_KEY: str | None = None
    SUPABASE_STORAGE_BUCKET: str = "profiles"

    # Postgres (pgvector habilitado)
    DATABASE_URL: str

    # Face
    FACE_RECOGNITION_THRESHOLD: float = Field(
        0.35, description="Menor é mais parecido (cosine distance)."
    )
    MAX_FACES_PER_IMAGE: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
