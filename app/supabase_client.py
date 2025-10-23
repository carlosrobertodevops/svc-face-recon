# app/supabase_client.py
from supabase import create_client, Client
from .config import settings

_sb_client: Client | None = None


def get_supabase() -> Client:
    global _sb_client
    if _sb_client is None:
        _sb_client = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY
        )
    return _sb_client
