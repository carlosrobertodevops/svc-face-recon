# app/indexer.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

from .supabase_client import get_supabase
from .config import settings
from .face_engine import face_engine
from .repository import upsert_member_embedding, fetch_all_embeddings
from .utils import fetch_bytes_from_supabase_path, load_image_from_bytes


class InMemoryIndex:
    """
    Cache simples em memória (member_id -> embedding).
    Útil para reduzir round-trips em consultas repetidas.
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


def build_index_from_members() -> dict:
    """
    1) Lê members (id, name, photo_path)
    2) Baixa foto do storage
    3) Extrai melhor rosto
    4) Persiste embedding (pgvector) e reconstrói o cache
    """
    sb = get_supabase()
    resp = sb.table("members").select("id, name, photo_path").execute()
    members = resp.data or []

    indexed = 0
    for m in members:
        mid = str(m["id"])
        path = m.get("photo_path")
        if not path:
            continue
        try:
            img_b = fetch_bytes_from_supabase_path(path)
            img = load_image_from_bytes(img_b)
            faces = face_engine.extract_embeddings(img, max_faces=1)
            if not faces:
                continue
            emb = faces[0]["embedding"]
            upsert_member_embedding(mid, emb)
            indexed += 1
        except Exception as e:
            # log mínimo (em produção, usar logger estruturado)
            print(f"[WARN] member {mid}: {e}")

    mem_index.rebuild()
    return {"indexed": indexed, "total": len(members)}
