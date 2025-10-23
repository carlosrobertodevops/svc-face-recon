from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    SERVICE_NAME: str = "svc-face-recon"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Supabase
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_ANON_KEY: str | None = None
    SUPABASE_STORAGE_BUCKET: str = "uploads"

    # Postgres
    DATABASE_URL: str

    MEMBERS_TABLE: str = "members"  # <- você sobrescreve via .env para 'membros'
    MEMBERS_ID_COLUMN: str = "membro_id"  # <- 'membro_id'
    MEMBERS_NAME_COLUMN: str = "nome_completo"  # <- 'nome_completo'
    MEMBERS_PHOTOS_COLUMN: str = "photo_path"  # <- 'fotos_path'

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
