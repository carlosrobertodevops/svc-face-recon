# app/indexer.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, List
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


class InMemoryIndex:
    """
    Cache simples em memória (member_id -> embedding).
    """

    def __init__(self):
        self.embeddings: Dict[str, np.ndarray] = {}

    def rebuild(self):
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


def _normalize_sources_from_fotos_path(
    fotos_path: Optional[List[str]],
) -> List[tuple[str, str]]:
    """
    Converte a coluna fotos_path (varchar[]) em uma lista de fontes.
    Cada item vira uma tupla (kind, value):
      - ("storage", "membros/abc.jpg")
      - ("url",     "https://.../arquivo.jpg")
    Regras:
      * se string parecer URL pública do Supabase -> vira ("storage", relpath)
      * se começar com "membros/" -> ("storage", str)
      * senão -> assume URL genérica ("url", str)
    """
    out: List[tuple[str, str]] = []
    if not fotos_path:
        return out
    for s in fotos_path:
        s = (s or "").strip()
        if not s:
            continue
        parsed = public_url_to_storage_path(s)
        if parsed:
            bucket, rel = parsed
            # só indexamos o que está dentro do bucket esperado
            if bucket == settings.SUPABASE_STORAGE_BUCKET:
                out.append(("storage", rel))
                continue
        if s.lower().startswith("membros/"):
            out.append(("storage", s))
        elif s.lower().startswith("uploads/membros/"):
            # às vezes salvam com bucket no início
            rel = s.split("uploads/", 1)[1]  # membros/xxx.jpg
            out.append(("storage", rel))
        else:
            out.append(("url", s))
    return out


def _avg_normalize(embs: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Faz média e normaliza (L2). Retorna None se lista vazia.
    """
    if not embs:
        return None
    M = np.stack(embs, axis=0).astype(np.float32)
    m = M.mean(axis=0)
    n = np.linalg.norm(m)
    if n == 0:
        return None
    return (m / n).astype(np.float32)


def build_index_from_members() -> dict:
    """
    1) Lê public.membros (id, name, fotos_path::varchar[])
    2) Para cada membro, tenta baixar e extrair embeddings de MULTIPLAS fotos.
    3) Faz média dos embeddings e salva em member_faces (pgvector).
    4) Reconstrói cache em memória.
    """
    sb = get_supabase()
    # fotos_path é varchar[]; o SDK já entrega como list[str]
    resp = sb.table("membros").select("id, name, fotos_path").execute()
    rows = resp.data or []

    total = len(rows)
    indexed = 0

    for m in rows:
        mid = str(m["id"])
        fotos_path = m.get("fotos_path")  # pode vir None ou []
        sources = _normalize_sources_from_fotos_path(fotos_path)

        if not sources:
            # sem foto -> pula
            continue

        embeddings: List[np.ndarray] = []

        for kind, value in sources:
            try:
                if kind == "storage":
                    # value = "membros/xyz.jpg"
                    b = fetch_bytes_from_supabase_path(value)
                else:
                    # URL genérica (pública/assinada)
                    b = fetch_bytes_from_url(value)
                img = load_image_from_bytes(b)
                faces = face_engine.extract_embeddings(img, max_faces=1)
                if faces:
                    embeddings.append(faces[0]["embedding"])
            except Exception as e:
                print(f"[WARN] member {mid} foto '{value}': {e}")

        emb_avg = _avg_normalize(embeddings)
        if emb_avg is None:
            continue

        try:
            upsert_member_embedding(mid, emb_avg)
            indexed += 1
        except Exception as e:
            print(f"[WARN] member {mid} upsert: {e}")

    mem_index.rebuild()
    return {"indexed": indexed, "total": total}
