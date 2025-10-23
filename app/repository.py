from __future__ import annotations
import time
from typing import Iterable, List, Tuple

import numpy as np
import psycopg
from psycopg.rows import tuple_row
from psycopg.types import TypeInfo

from .config import settings


def get_conn(retries: int = 6, delay_sec: float = 2.0):
    """
    Conexão resiliente ao Postgres.
    Usa settings.DATABASE_URL (garanta que, em Docker, a URL aponte para 'db' e não 'localhost').
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            # row_factory padrão = tuple; ajustamos por cursor
            conn = psycopg.connect(settings.DATABASE_URL, autocommit=True)
            if attempt > 1:
                print(f"[INFO] Conectado ao Postgres após {attempt} tentativas.")
            return conn
        except Exception as e:
            last_exc = e
            print(
                f"[WARN] Tentativa {attempt}/{retries} de conectar ao Postgres falhou: {e}"
            )
            time.sleep(delay_sec)
    raise RuntimeError(f"Falha ao conectar ao Postgres: {last_exc}")


def _ensure_schema():
    """
    Garante que a tabela de embeddings exista.
    - member_id: TEXT (ou ajuste para o tipo do seu id)
    - embedding: BYTEA (armazenamos vetores float32 serializados)
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                member_id TEXT PRIMARY KEY,
                embedding BYTEA NOT NULL
            );
            """
        )


def upsert_member_embedding(member_id: str, embedding: np.ndarray):
    """
    Salva/atualiza o embedding (float32 normalizado) no banco.
    """
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32, copy=False)

    _ensure_schema()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO embeddings (member_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (member_id) DO UPDATE
              SET embedding = EXCLUDED.embedding;
            """,
            (member_id, memoryview(embedding.tobytes())),
        )


def fetch_all_embeddings() -> List[Tuple[str, np.ndarray]]:
    """
    Retorna todos os embeddings como (member_id, np.ndarray float32 normalizado).
    """
    _ensure_schema()
    out: List[Tuple[str, np.ndarray]] = []
    with get_conn() as conn, conn.cursor(row_factory=tuple_row) as cur:
        cur.execute("SELECT member_id, embedding FROM embeddings;")
        for member_id, blob in cur.fetchall():
            if not blob:
                continue
            vec = np.frombuffer(bytes(blob), dtype=np.float32)
            # re-normaliza por segurança (caso dados antigos)
            n = float(np.linalg.norm(vec))
            if n > 0.0:
                vec = (vec / n).astype(np.float32)
            else:
                vec = vec.astype(np.float32)
            out.append((str(member_id), vec))
    return out
