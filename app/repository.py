# app/repository.py
from typing import Optional, Union
import numpy as np
import psycopg
from .config import settings


def get_conn():
    return psycopg.connect(settings.DATABASE_URL)


def upsert_member_embedding(member_id: Union[int, str], embedding: np.ndarray) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into member_faces (member_id, embedding)
            values (%s, %s)
            on conflict (member_id)
            do update set embedding = excluded.embedding, updated_at = now();
            """,
            (int(member_id), embedding.tolist()),
        )


def fetch_all_embeddings() -> list[tuple[str, np.ndarray]]:
    out = []
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("select member_id, embedding from member_faces;")
        for mid, emb in cur.fetchall():
            out.append((str(int(mid)), np.array(emb, dtype=np.float32)))
    return out


def search_top1(embedding: np.ndarray) -> Optional[tuple[str, float]]:
    """
    Busca aproximada no pgvector usando distância de cosseno.
    Retorna (member_id, distance) ou None.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select member_id,
                   1 - (embedding <#> %s) as cosine_similarity
            from member_faces
            order by embedding <#> %s
            limit 1;
            """,
            (embedding.tolist(), embedding.tolist()),
        )
        row = cur.fetchone()
        if not row:
            return None
        found_id, sim = row
        dist = 1.0 - float(sim)
        return (str(found_id), dist)
