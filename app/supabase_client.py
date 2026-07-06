# DEPRECATED: Supabase removido (migração Postgres+MinIO). Módulo mantido vazio para evitar ImportError legado.


def get_supabase(*a, **k):
    raise RuntimeError("Supabase removido — use repository/storage")
