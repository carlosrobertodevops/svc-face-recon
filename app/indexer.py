# app/indexer.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, List

import hashlib
import json
import time
import numpy as np

from .supabase_client import get_supabase
from .config import settings
from .face_engine import face_engine
from .repository import upsert_member_embedding, fetch_all_embeddings
from .utils import (
    load_image_from_bytes,
    fetch_bytes_from_supabase_path,
    fetch_bytes_from_url,
    public_url_to_storage_path,
)

# ---------------------------------------------------------------------
# Redis (opcional)
# ---------------------------------------------------------------------
# - Usa redis.asyncio se REDIS_URL estiver presente
# - Mantém lock, progresso e cache de bytes de imagens

_redis_client = None  # singleton (async)


async def _get_redis():
    """Retorna um cliente Redis (async) OU None se não configurado."""
    global _redis_client
    redis_url = getattr(settings, "REDIS_URL", None)
    if not redis_url:
        return None
    if _redis_client is not None:
        return _redis_client
    try:
        from redis.asyncio import Redis

        _redis_client = Redis.from_url(redis_url, decode_responses=False)
        # ping só na primeira vez
        await _redis_client.ping()
        return _redis_client
    except Exception as e:
        print(f"[WARN] Redis indisponível ({e}). Seguindo sem Redis.")
        return None


async def _acquire_lock(rds, key: str, ttl_sec: int) -> bool:
    """
    Tenta adquirir lock (SETNX). Retorna True se conseguiu, False se já está travado.
    """
    try:
        # SET key value NX EX ttl
        ok = await rds.set(key, b"1", ex=ttl_sec, nx=True)
        return bool(ok)
    except Exception as e:
        print(f"[WARN] Falha ao adquirir lock '{key}': {e}")
        return True  # se der erro no redis, não bloqueia a execução


async def _release_lock(rds, key: str):
    try:
        await rds.delete(key)
    except Exception:
        pass


async def _set_status(rds, key: str, payload: dict, ttl_sec: int = 3600):
    """Grava JSON com status/progresso."""
    if not rds:
        return
    try:
        await rds.set(key, json.dumps(payload).encode("utf-8"), ex=ttl_sec)
    except Exception as e:
        print(f"[WARN] Falha ao gravar status no Redis: {e}")


async def _get_status(rds, key: str) -> Optional[dict]:
    """Lê o JSON de status (se existir)."""
    if not rds:
        return None
    try:
        raw = await rds.get(key)
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _bytes_cache_key(kind: str, value: str) -> str:
    # chave curta e estável para o cache de bytes
    h = hashlib.sha256(f"{kind}:{value}".encode("utf-8")).hexdigest()
    return f"img:{h}"


async def _get_bytes_cached(kind: str, value: str) -> Optional[bytes]:
    """Lê cache de bytes no Redis (se habilitado)."""
    rds = await _get_redis()
    if not rds:
        return None
    try:
        key = _bytes_cache_key(kind, value)
        return await rds.get(key)  # retorna bytes ou None
    except Exception:
        return None


async def _set_bytes_cached(kind: str, value: str, b: bytes, ttl_sec: int = 3600):
    """Grava cache de bytes no Redis (se habilitado)."""
    rds = await _get_redis()
    if not rds:
        return
    try:
        key = _bytes_cache_key(kind, value)
        await rds.set(key, b, ex=ttl_sec)
    except Exception:
        pass


# -----------------------------------------------------------
# Índice em memória
# -----------------------------------------------------------


class InMemoryIndex:
    def __init__(self):
        self.embeddings: Dict[str, np.ndarray] = {}

    def rebuild(self):
        # carrega tudo do banco
        self.embeddings = {mid: emb for (mid, emb) in fetch_all_embeddings()}

    def top1(self, query: np.ndarray) -> Optional[Tuple[str, float]]:
        if not self.embeddings:
            return None
        best_id, best_dist = None, 9e9
        for mid, emb in self.embeddings.items():
            dist = float(1.0 - float(np.dot(emb, query)))
            if dist < best_dist:
                best_id, best_dist = mid, dist
        if best_id is None:
            return None
        return (best_id, best_dist)


mem_index = InMemoryIndex()

# -----------------------------------------------------------
# Auxiliares
# -----------------------------------------------------------


def _normalize_sources_from_photos_field(photos_value) -> List[tuple[str, str]]:
    """
    Converte o campo de fotos (string única ou lista de strings) em fontes:
      - ("storage", "membros/abc.jpg")  ou
      - ("url", "https://...")
    Regras:
      * URL pública do Supabase -> ("storage", relpath)
      * começa com "membros/" -> storage
      * começa com "uploads/membros/" -> storage (removendo 'uploads/')
      * caso contrário -> url
    """
    if photos_value is None:
        return []
    values: List[str] = (
        photos_value if isinstance(photos_value, list) else [str(photos_value)]
    )

    out: List[tuple[str, str]] = []
    for s in values:
        s = (s or "").strip()
        if not s:
            continue
        parsed = public_url_to_storage_path(s)
        if parsed:
            bucket, rel = parsed
            if bucket == settings.SUPABASE_STORAGE_BUCKET:
                out.append(("storage", rel))
                continue
        if s.lower().startswith("membros/"):
            out.append(("storage", s))
        elif s.lower().startswith("uploads/membros/"):
            rel = s.split("uploads/", 1)[1]
            out.append(("storage", rel))
        else:
            out.append(("url", s))
    return out


def _avg_normalize(embs: List[np.ndarray]) -> Optional[np.ndarray]:
    if not embs:
        return None
    M = np.stack(embs, axis=0).astype(np.float32)
    m = M.mean(axis=0)
    n = np.linalg.norm(m)
    if n == 0:
        return None
    return (m / n).astype(np.float32)


async def _maybe_await(val):
    if hasattr(val, "__await__"):
        return await val
    return val


async def _get_bytes(kind: str, value: str) -> bytes:
    """
    Busca bytes de imagem com cache opcional em Redis.
    """
    # 1) tenta cache
    cached = await _get_bytes_cached(kind, value)
    if cached is not None:
        return cached

    # 2) baixa de fato
    if kind == "storage":
        b = await _maybe_await(fetch_bytes_from_supabase_path(value))
    else:
        b = await _maybe_await(fetch_bytes_from_url(value))

    # 3) salva no cache (best-effort)
    await _set_bytes_cached(kind, value, b)
    return b


# -----------------------------------------------------------
# Indexação
# -----------------------------------------------------------


async def build_index_from_members() -> dict:
    """
    Lê tabela configurada no Supabase, baixa fotos, gera embedding médio por membro,
    upserta no Postgres e recarrega o cache.

    Recursos novos:
      - Lock via Redis (key: index:lock)
      - Progresso em Redis (key: index:status)
      - Cache de bytes de imagens (chave img:SHA256)
    """
    sb = get_supabase()

    # chaves de controle no Redis
    rds = await _get_redis()
    lock_key = "index:lock"
    status_key = "index:status"
    lock_ttl = 60 * 30  # 30 min

    # Tenta adquirir lock (se falhar, retorna status atual)
    if rds:
        got_lock = await _acquire_lock(rds, lock_key, lock_ttl)
        if not got_lock:
            status = await _get_status(rds, status_key)
            return {
                "message": "Indexação já está em execução.",
                "busy": True,
                "status": status or {},
            }

    # Status inicial
    started_at = int(time.time())
    status = {
        "started_at": started_at,
        "total": 0,
        "processed": 0,
        "indexed": 0,
        "skipped_no_photo": 0,
        "skipped_no_face": 0,
        "errors": 0,
        "cache_reloaded": False,
        "state": "running",
    }
    await _set_status(rds, status_key, status)

    try:
        sel_cols = f"{settings.MEMBERS_ID_COLUMN}, {settings.MEMBERS_NAME_COLUMN}, {settings.MEMBERS_PHOTOS_COLUMN}"
        resp = sb.table(settings.MEMBERS_TABLE).select(sel_cols).execute()
        rows = resp.data or []

        total = len(rows)
        status["total"] = total
        await _set_status(rds, status_key, status)

        print(f"[INFO] Iniciando indexação de {total} membros...")

        # contadores
        indexed = 0
        skipped_no_photo = 0
        skipped_no_face = 0
        errors = 0
        processed = 0

        # atualiza status no Redis a cada N membros para reduzir I/O
        UPDATE_EVERY = 10

        for m in rows:
            mid = str(m[settings.MEMBERS_ID_COLUMN])
            name = m.get(settings.MEMBERS_NAME_COLUMN)
            photos_val = m.get(settings.MEMBERS_PHOTOS_COLUMN)
            sources = _normalize_sources_from_photos_field(photos_val)

            if not sources:
                skipped_no_photo += 1
                processed += 1
                if processed % UPDATE_EVERY == 0:
                    status.update(
                        {
                            "processed": processed,
                            "indexed": indexed,
                            "skipped_no_photo": skipped_no_photo,
                            "skipped_no_face": skipped_no_face,
                            "errors": errors,
                        }
                    )
                    await _set_status(rds, status_key, status)
                print(f"[WARN] Membro {mid} ({name}) sem fotos.")
                continue

            embeddings: List[np.ndarray] = []
            for kind, value in sources:
                try:
                    b = await _get_bytes(kind, value)  # sem asyncio.run
                    img = load_image_from_bytes(b)
                    faces = face_engine.extract_embeddings(img, max_faces=1)
                    if faces:
                        embeddings.append(faces[0]["embedding"])
                except Exception as e:
                    errors += 1
                    print(f"[WARN] member {mid} foto '{value}': {e}")

            emb_avg = _avg_normalize(embeddings)
            if emb_avg is None:
                skipped_no_face += 1
                processed += 1
                if processed % UPDATE_EVERY == 0:
                    status.update(
                        {
                            "processed": processed,
                            "indexed": indexed,
                            "skipped_no_photo": skipped_no_photo,
                            "skipped_no_face": skipped_no_face,
                            "errors": errors,
                        }
                    )
                    await _set_status(rds, status_key, status)
                print(f"[WARN] Membro {mid} ({name}) sem embedding válido.")
                continue

            try:
                upsert_member_embedding(mid, emb_avg)
                indexed += 1
            except Exception as e:
                errors += 1
                print(f"[WARN] member {mid} upsert: {e}")

            processed += 1
            if processed % UPDATE_EVERY == 0:
                status.update(
                    {
                        "processed": processed,
                        "indexed": indexed,
                        "skipped_no_photo": skipped_no_photo,
                        "skipped_no_face": skipped_no_face,
                        "errors": errors,
                    }
                )
                await _set_status(rds, status_key, status)

        # Recarrega o cache em memória
        try:
            mem_index.rebuild()
            cache_ok = True
        except Exception as e:
            cache_ok = False
            print(f"[WARN] Falha ao reconstruir cache: {e}")

        finished_at = int(time.time())
        status.update(
            {
                "processed": processed,
                "indexed": indexed,
                "skipped_no_photo": skipped_no_photo,
                "skipped_no_face": skipped_no_face,
                "errors": errors,
                "cache_reloaded": cache_ok,
                "state": "done",
                "finished_at": finished_at,
                "duration_sec": finished_at - started_at,
            }
        )
        await _set_status(rds, status_key, status)

        return {
            "indexed": indexed,
            "total": total,
            "skipped_no_photo": skipped_no_photo,
            "skipped_no_face": skipped_no_face,
            "errors": errors,
            "cache_reloaded": cache_ok,
            "progress": status,
        }

    finally:
        # libera lock
        if rds:
            await _release_lock(rds, lock_key)
