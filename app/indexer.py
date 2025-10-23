from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import numpy as np
import asyncio
import traceback

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
    """Índice em memória para busca rápida de embeddings."""

    def __init__(self):
        self.embeddings: Dict[str, np.ndarray] = {}

    def rebuild(self):
        """Recarrega embeddings armazenados no banco (pgvector)."""
        self.embeddings = {mid: emb for (mid, emb) in fetch_all_embeddings()}
        print(
            f"[INFO] Índice reconstruído: {len(self.embeddings)} embeddings carregados."
        )

    def top1(self, query: np.ndarray) -> Optional[Tuple[str, float]]:
        """Busca o embedding mais próximo (menor distância)."""
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
    """
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
            rel = s.split("uploads/", 1)[1]
            out.append(("storage", rel))
        else:
            out.append(("url", s))
    return out


def _avg_normalize(embs: List[np.ndarray]) -> Optional[np.ndarray]:
    """Faz a média e normaliza embeddings de uma mesma pessoa."""
    if not embs:
        return None
    M = np.stack(embs, axis=0).astype(np.float32)
    m = M.mean(axis=0)
    n = np.linalg.norm(m)
    if n == 0:
        return None
    return (m / n).astype(np.float32)


def build_index_from_members() -> dict:
    """Lê a tabela de membros do Supabase, baixa suas fotos, extrai embeddings e grava no banco local."""
    sb = get_supabase()
    sel_cols = f"{settings.MEMBERS_ID_COLUMN}, {settings.MEMBERS_NAME_COLUMN}, {settings.MEMBERS_PHOTOS_COLUMN}"
    resp = sb.table(settings.MEMBERS_TABLE).select(sel_cols).execute()
    rows = resp.data or []

    total = len(rows)
    indexed = 0
    print(f"[INFO] Iniciando indexação de {total} membros...")

    for m in rows:
        mid = str(m[settings.MEMBERS_ID_COLUMN])
        name = m.get(settings.MEMBERS_NAME_COLUMN, "(sem nome)")
        photos_val = m.get(settings.MEMBERS_PHOTOS_COLUMN)
        sources = _normalize_sources_from_photos_field(photos_val)
        if not sources:
            print(f"[WARN] Membro {mid} ({name}) sem fotos.")
            continue

        embeddings: List[np.ndarray] = []
        for kind, value in sources:
            try:
                # Executa corrotinas async de forma síncrona
                if kind == "storage":
                    b = asyncio.run(fetch_bytes_from_supabase_path(value))
                else:
                    b = asyncio.run(fetch_bytes_from_url(value))

                if not b:
                    print(f"[WARN] Foto vazia/indisponível: {value}")
                    continue

                img = load_image_from_bytes(b)
                faces = face_engine.extract_embeddings(img, max_faces=1)
                if faces:
                    embeddings.append(faces[0]["embedding"])
                else:
                    print(f"[WARN] Nenhum rosto detectado em {value}")

            except Exception as e:
                print(f"[WARN] member {mid} foto '{value}': {e}")
                traceback.print_exc()

        emb_avg = _avg_normalize(embeddings)
        if emb_avg is None:
            print(f"[WARN] Membro {mid} ({name}) sem embedding válido.")
            continue

        try:
            upsert_member_embedding(mid, emb_avg)
            indexed += 1
        except Exception as e:
            print(f"[WARN] member {mid} upsert: {e}")
            traceback.print_exc()

    mem_index.rebuild()
    print(f"[INFO] Indexação concluída. {indexed}/{total} membros processados.")
    return {"indexed": indexed, "total": total}
