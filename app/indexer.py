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


def _normalize_sources_from_photos_field(photos_value) -> List[tuple[str, str]]:
    """
    Converte o campo de fotos (pode ser string única OU lista de strings) em fontes:
      - ("storage", "membros/abc.jpg")  ou
      - ("url", "https://...")
    Regras:
      * URL pública do Supabase -> ("storage", relpath)
      * começa com "membros/"      -> storage
      * começa com "uploads/membros/" -> vira storage removendo 'uploads/'
      * caso contrário -> url
    """
    values: List[str] = []
    if photos_value is None:
        return []
    if isinstance(photos_value, list):
        values = photos_value
    else:
        values = [str(photos_value)]

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
            rel = s.split("uploads/", 1)[1]  # "membros/xxx.jpg"
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


def build_index_from_members() -> dict:
    sb = get_supabase()

    # Monta seleção dinâmica com base nas envs
    sel_cols = f"{settings.MEMBERS_ID_COLUMN}, {settings.MEMBERS_NAME_COLUMN}, {settings.MEMBERS_PHOTOS_COLUMN}"
    resp = sb.table(settings.MEMBERS_TABLE).select(sel_cols).execute()
    rows = resp.data or []

    total = len(rows)
    indexed = 0

    for m in rows:
        mid = str(m[settings.MEMBERS_ID_COLUMN])
        photos_val = m.get(settings.MEMBERS_PHOTOS_COLUMN)
        sources = _normalize_sources_from_photos_field(photos_val)
        if not sources:
            continue

        embeddings: List[np.ndarray] = []
        for kind, value in sources:
            try:
                if kind == "storage":
                    b = fetch_bytes_from_supabase_path(value)
                else:
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
