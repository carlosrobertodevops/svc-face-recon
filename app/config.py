# app/config.py
from pydantic import Field

try:
    # Pydantic Settings v2
    from pydantic_settings import BaseSettings, SettingsConfigDict

    V2 = True
except Exception:
    # fallback p/ setups antigos
    from pydantic import BaseSettings  # type: ignore

    SettingsConfigDict = None
    V2 = False


class Settings(BaseSettings):
    """
    Lê .env do serviço e IGNORA variáveis extras (dd_*, gf_* etc.)
    Defaults já alinhados ao seu schema real (membros / membro_id / nome_completo / fotos_path).
    """

    # -------- Config Pydantic --------
    if V2:
        # Pydantic v2
        model_config = SettingsConfigDict(
            env_file=".env",
            env_ignore_empty=True,
            extra="ignore",  # <<< chave para não quebrar com dd_*/gf_*
        )
    else:
        # Pydantic v1 (fallback)
        class Config:
            env_file = ".env"
            extra = "ignore"  # <<< ignora variáveis desconhecidas

    # -------- App --------
    SERVICE_NAME: str = "svc-face-recon"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # -------- Supabase --------
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_ANON_KEY: str | None = None
    SUPABASE_STORAGE_BUCKET: str = "uploads"

    # -------- Postgres --------
    DATABASE_URL: str

    # -------- Face --------
    FACE_RECOGNITION_THRESHOLD: float = Field(0.35)
    MAX_FACES_PER_IMAGE: int = 5

    # -------- Fonte dos membros (defaults = seu schema real) --------
    MEMBERS_TABLE: str = "membros"
    MEMBERS_ID_COLUMN: str = "membro_id"
    MEMBERS_NAME_COLUMN: str = "nome_completo"  # <<< corrigido
    MEMBERS_PHOTOS_COLUMN: str = "fotos_path"


settings = Settings()
