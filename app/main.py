# app/main.py
from __future__ import annotations

import asyncio
import base64
import re
from typing import Optional, Dict, List, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .indexer import mem_index, build_index_from_members
from .repository import get_conn, upsert_member_embedding
from .supabase_client import get_supabase
from .utils import (
    load_image_from_bytes,
    fetch_bytes_from_supabase_path,
    fetch_bytes_from_url,
    public_url_to_storage_path,
)
from .face_engine import face_engine

# >>> modelos centralizados em app/schema.py <<<
from .schema import (
    IndexResponse,
    IndexResult,
    EnrollRequest,
    EnrollResponse,
    IdentityRequest,
    IdentityResponse,
    IdentityCandidate,
    VerifyRequest,
    VerifyResponse,
    CompareRequest,
    CompareResponse,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _normalize_supabase_relpath(p: str) -> str:
    """
    Normaliza caminho do Storage para o formato relativo do bucket:
    - remove barras iniciais
    - aceita prefixo 'uploads/' e remove
    """
    p = (p or "").strip().lstrip("/")
    if not p:
        return p
    if p.lower().startswith("uploads/"):
        p = p.split("uploads/", 1)[1]
    return p


_DRIVE_FILE_ID = re.compile(r"/d/([a-zA-Z0-9_-]+)")
_DRIVE_ID_QUERY = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


def _to_google_drive_download(url: str) -> str:
    """
    Converte links 'view' do Google Drive em links de download direto.
      - https://drive.google.com/file/d/FILE_ID/view?usp=sharing
      - https://drive.google.com/open?id=FILE_ID
    -> https://drive.google.com/uc?export=download&id=FILE_ID
    """
    if "drive.google.com" not in url:
        return url
    file_id = None
    m = _DRIVE_FILE_ID.search(url)
    if m:
        file_id = m.group(1)
    else:
        m = _DRIVE_ID_QUERY.search(url)
        if m:
            file_id = m.group(1)
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


async def _maybe_await(v):
    if hasattr(v, "__await__"):
        return await v
    return v


async def _fetch_image_bytes(
    *,
    supabase_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_b64: Optional[str] = None,
) -> bytes:
    """
    Padroniza todas as entradas de imagem.
    Prioridade:
      1) supabase_path (aceita caminho relativo, URL pública do Supabase, com/sem 'uploads/')
      2) image_url (se for Supabase público -> usa Storage; se for Drive 'view' -> converte p/ download)
      3) image_b64
    """
    # 1) SUPABASE_PATH
    if supabase_path:
        val = supabase_path.strip()
        parsed = public_url_to_storage_path(val)
        if parsed and parsed[0] == settings.SUPABASE_STORAGE_BUCKET:
            _, rel = parsed
            rel = _normalize_supabase_relpath(rel)
            b = await _maybe_await(fetch_bytes_from_supabase_path(rel))
            if not b:
                raise FileNotFoundError(f"Objeto não encontrado no Storage: {rel}")
            return b
        # caminho relativo
        rel = _normalize_supabase_relpath(val)
        b = await _maybe_await(fetch_bytes_from_supabase_path(rel))
        if not b:
            raise FileNotFoundError(f"Objeto não encontrado no Storage: {rel}")
        return b

    # 2) IMAGE_URL
    if image_url:
        url = image_url.strip()
        parsed = public_url_to_storage_path(url)
        if parsed and parsed[0] == settings.SUPABASE_STORAGE_BUCKET:
            _, rel = parsed
            rel = _normalize_supabase_relpath(rel)
            b = await _maybe_await(fetch_bytes_from_supabase_path(rel))
            if not b:
                raise FileNotFoundError(f"Objeto não encontrado no Storage: {rel}")
            return b
        url = _to_google_drive_download(url)
        b = await _maybe_await(fetch_bytes_from_url(url))
        if not b:
            raise ValueError("Falha ao baixar URL da imagem")
        return b

    # 3) IMAGE_B64
    if image_b64:
        try:
            return base64.b64decode(image_b64, validate=True)
        except Exception as e:
            raise ValueError(f"Base64 inválido: {e}")

    raise ValueError("Envie supabase_path, image_url ou image_b64.")


async def _read_upload_bytes(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise ValueError("Arquivo vazio.")
    return data


async def _extract_single_embedding(img_bytes: bytes) -> Optional[List[float]]:
    img = load_image_from_bytes(img_bytes)
    faces = face_engine.extract_embeddings(img, max_faces=1)
    if not faces:
        return None
    emb = faces[0]["embedding"]
    if emb is None:
        return None
    return emb.astype("float32").tolist()


def _distance(emb_a: List[float], emb_b: List[float]) -> float:
    s = 0.0
    for x, y in zip(emb_a, emb_b):
        s += float(x) * float(y)
    return float(1.0 - s)


def _fetch_member_fields_from_supabase(member_id: str) -> Dict[str, Optional[str]]:
    """
    Busca nome e primeira foto (relpath) no Supabase para enriquecer as respostas.
    - Campo de fotos pode ser string ou lista de strings (conforme sua tabela).
    - Retorna {'name': str|None, 'photo_path': str|None}
    """
    out = {"name": None, "photo_path": None}
    try:
        sb = get_supabase()
        sel = f"{settings.MEMBERS_NAME_COLUMN}, {settings.MEMBERS_PHOTOS_COLUMN}"
        resp = (
            sb.table(settings.MEMBERS_TABLE)
            .select(sel)
            .eq(settings.MEMBERS_ID_COLUMN, member_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return out
        row = rows[0]
        out["name"] = row.get(settings.MEMBERS_NAME_COLUMN)

        photos_val = row.get(settings.MEMBERS_PHOTOS_COLUMN)
        if photos_val is None:
            return out

        # aceita string única ou lista
        values: List[str] = (
            photos_val if isinstance(photos_val, list) else [str(photos_val)]
        )
        for s in values:
            s = (s or "").strip()
            if not s:
                continue
            # se for URL pública do supabase, transforma em (bucket, rel)
            parsed = public_url_to_storage_path(s)
            if parsed and parsed[0] == settings.SUPABASE_STORAGE_BUCKET:
                _, rel = parsed
                out["photo_path"] = _normalize_supabase_relpath(rel)
                break
            # se for caminho relativo, prioriza 'membros/' ou 'uploads/membros/'
            if s.lower().startswith("uploads/"):
                s = _normalize_supabase_relpath(s)
            if s.lower().startswith("membros/"):
                out["photo_path"] = s
                break
    except Exception:
        pass
    return out


# -----------------------------------------------------------------------------
# FastAPI + CORS
# -----------------------------------------------------------------------------

app = FastAPI(
    title="svc-face-recon",
    description="Microserviço de reconhecimento facial (Supabase Storage + pgvector).",
    version="1.0.0",
    openapi_tags=[
        {"name": "ops", "description": "Operações de saúde/diagnóstico do serviço."},
        {
            "name": "face",
            "description": "Reconhecimento facial: indexação, identificação, verificação e comparação.",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# OPS
# -----------------------------------------------------------------------------


@app.get("/live", tags=["ops"])
async def live():
    return {"status": "live"}


@app.get("/health", tags=["ops"])
async def health():
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/ready", tags=["ops"])
async def ready():
    return {"ready": True, "cache_embeddings": len(mem_index.embeddings)}


@app.get("/ops/status", tags=["ops"])
async def ops_status():
    return {
        "service": "svc-face-recon",
        "threshold": settings.FACE_RECOGNITION_THRESHOLD,
        "cache_embeddings": len(mem_index.embeddings),
        "db_url_host_hint": settings.DATABASE_URL.split("@")[-1].split("/")[0]
        if "@" in settings.DATABASE_URL
        else settings.DATABASE_URL,
    }


# -----------------------------------------------------------------------------
# FACE — JSON
# -----------------------------------------------------------------------------


@app.post("/index", response_model=IndexResponse, tags=["face"])
async def index_all():
    """
    Reindexa a partir do Supabase: baixa fotos, gera embeddings, salva no Postgres e recarrega o cache.
    """
    try:
        result_dict = (
            await build_index_from_members()
        )  # precisa ser compatível com IndexResult
        # valida já no retorno
        return IndexResponse(ok=True, result=IndexResult(**result_dict))
    except Exception as e:
        return IndexResponse(ok=False, error=str(e))


@app.post("/enroll", response_model=EnrollResponse, tags=["face"])
async def enroll(req: EnrollRequest):
    """
    Cadastra/atualiza o embedding de um membro a partir de uma imagem única.
    """
    try:
        b = await _fetch_image_bytes(
            supabase_path=req.supabase_path,
            image_url=req.image_url,
            image_b64=req.image_b64,
        )
        emb = await _extract_single_embedding(b)
        if emb is None:
            raise HTTPException(status_code=400, detail="Nenhum rosto detectado.")
        upsert_member_embedding(req.member_id, face_engine.to_numpy(emb))
        mem_index.rebuild()
        return EnrollResponse(ok=True, member_id=req.member_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/identify", response_model=IdentityResponse, tags=["face"])
async def identity(req: IdentityRequest):
    """
    Identifica os top_k candidatos no cache em memória.
    """
    thr = float(settings.FACE_RECOGNITION_THRESHOLD)
    try:
        b = await _fetch_image_bytes(
            supabase_path=req.supabase_path,
            image_url=req.image_url,
            image_b64=req.image_b64,
        )
        emb = await _extract_single_embedding(b)
        if emb is None:
            return IdentityResponse(ok=False, threshold=thr, candidates=[])

        # varredura no cache
        cands: List[Tuple[str, float]] = []
        for mid, ref in mem_index.embeddings.items():
            d = _distance(emb, ref.tolist())
            cands.append((mid, d))
        cands.sort(key=lambda x: x[1])
        top = cands[: max(1, req.top_k)]

        out: List[IdentityCandidate] = []
        for mid, dist in top:
            meta = _fetch_member_fields_from_supabase(mid)
            out.append(
                IdentityCandidate(
                    member_id=mid,
                    distance=dist,
                    matched=dist <= thr,
                    name=meta.get("name"),
                    photo_path=meta.get("photo_path"),
                )
            )
        return IdentityResponse(ok=True, threshold=thr, candidates=out)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify", response_model=VerifyResponse, tags=["face"])
async def verify(req: VerifyRequest):
    """
    Verifica se a imagem pertence ao member_id informado.
    """
    thr = float(settings.FACE_RECOGNITION_THRESHOLD)
    try:
        ref = mem_index.embeddings.get(req.member_id)
        if ref is None:
            raise HTTPException(
                status_code=404, detail="member_id sem embedding no cache/banco."
            )

        b = await _fetch_image_bytes(
            supabase_path=req.supabase_path,
            image_url=req.image_url,
            image_b64=req.image_b64,
        )
        emb = await _extract_single_embedding(b)
        if emb is None:
            raise HTTPException(status_code=400, detail="Nenhum rosto detectado.")

        dist = _distance(emb, ref.tolist())
        meta = _fetch_member_fields_from_supabase(req.member_id)
        return VerifyResponse(
            ok=True,
            member_id=req.member_id,
            name=meta.get("name"),
            distance=dist,
            threshold=thr,
            matched=(dist <= thr),
            photo_path=meta.get("photo_path"),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=CompareResponse, tags=["face"])
async def compare(req: CompareRequest):
    """
    Compara duas imagens (A vs B) e retorna distância e is_same.
    """
    thr = float(settings.FACE_RECOGNITION_THRESHOLD)
    try:
        a_bytes = await _fetch_image_bytes(
            supabase_path=req.a_supabase_path or req.image_a_storage_path,
            image_url=req.a_image_url or req.image_a_url,
            image_b64=req.a_image_b64,
        )
        b_bytes = await _fetch_image_bytes(
            supabase_path=req.b_supabase_path or req.image_b_storage_path,
            image_url=req.b_image_url or req.image_b_url,
            image_b64=req.b_image_b64,
        )
        emb_a = await _extract_single_embedding(a_bytes)
        emb_b = await _extract_single_embedding(b_bytes)
        if emb_a is None or emb_b is None:
            raise HTTPException(
                status_code=400, detail="Não foi possível extrair rosto em A ou B."
            )
        dist = _distance(emb_a, emb_b)
        return CompareResponse(
            ok=True, distance=dist, threshold=thr, is_same=(dist <= thr)
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# FACE — MULTIPART/FILE (facilita testes pelo Swagger)
# -----------------------------------------------------------------------------


@app.post("/enroll/file", response_model=EnrollResponse, tags=["face"])
async def enroll_file(
    member_id: str = Form(...),
    image: UploadFile = File(...),
):
    try:
        b = await _read_upload_bytes(image)
        emb = await _extract_single_embedding(b)
        if emb is None:
            raise HTTPException(status_code=400, detail="Nenhum rosto detectado.")
        upsert_member_embedding(member_id, face_engine.to_numpy(emb))
        mem_index.rebuild()
        return EnrollResponse(ok=True, member_id=member_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/identify/file", response_model=IdentityResponse, tags=["face"])
async def identity_file(
    image: UploadFile = File(...),
    top_k: int = Form(1, ge=1, le=10),
):
    thr = float(settings.FACE_RECOGNITION_THRESHOLD)
    try:
        b = await _read_upload_bytes(image)
        emb = await _extract_single_embedding(b)
        if emb is None:
            return IdentityResponse(ok=False, threshold=thr, candidates=[])
        cands: List[Tuple[str, float]] = []
        for mid, ref in mem_index.embeddings.items():
            d = _distance(emb, ref.tolist())
            cands.append((mid, d))
        cands.sort(key=lambda x: x[1])
        top = cands[: max(1, top_k)]
        out: List[IdentityCandidate] = []
        for mid, dist in top:
            meta = _fetch_member_fields_from_supabase(mid)
            out.append(
                IdentityCandidate(
                    member_id=mid,
                    distance=dist,
                    matched=dist <= thr,
                    name=meta.get("name"),
                    photo_path=meta.get("photo_path"),
                )
            )
        return IdentityResponse(ok=True, threshold=thr, candidates=out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify/file", response_model=VerifyResponse, tags=["face"])
async def verify_file(
    member_id: str = Form(...),
    image: UploadFile = File(...),
):
    thr = float(settings.FACE_RECOGNITION_THRESHOLD)
    try:
        ref = mem_index.embeddings.get(member_id)
        if ref is None:
            raise HTTPException(
                status_code=404, detail="member_id sem embedding no cache/banco."
            )
        b = await _read_upload_bytes(image)
        emb = await _extract_single_embedding(b)
        if emb is None:
            raise HTTPException(status_code=400, detail="Nenhum rosto detectado.")
        dist = _distance(emb, ref.tolist())
        meta = _fetch_member_fields_from_supabase(member_id)
        return VerifyResponse(
            ok=True,
            member_id=member_id,
            name=meta.get("name"),
            distance=dist,
            threshold=thr,
            matched=(dist <= thr),
            photo_path=meta.get("photo_path"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare/files", response_model=CompareResponse, tags=["face"])
async def compare_files(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
):
    thr = float(settings.FACE_RECOGNITION_THRESHOLD)
    try:
        a_bytes = await _read_upload_bytes(image_a)
        b_bytes = await _read_upload_bytes(image_b)
        emb_a = await _extract_single_embedding(a_bytes)
        emb_b = await _extract_single_embedding(b_bytes)
        if emb_a is None or emb_b is None:
            raise HTTPException(
                status_code=400, detail="Não foi possível extrair rosto em A ou B."
            )
        dist = _distance(emb_a, emb_b)
        return CompareResponse(
            ok=True, distance=dist, threshold=thr, is_same=(dist <= thr)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup():
    # pequeno atraso ajuda em ambientes containerizados
    await asyncio.sleep(1.5)
    try:
        mem_index.rebuild()
        print("[INFO] Índice em memória carregado a partir do banco.")
    except Exception as e:
        print(f"[WARN] Falha ao reconstruir índice inicial: {e}")


# from __future__ import annotations

# import asyncio
# import base64
# import re
# from typing import Optional, Dict, List, Tuple

# from fastapi import FastAPI, HTTPException, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field

# from .config import settings
# from .indexer import mem_index, build_index_from_members
# from .repository import get_conn, upsert_member_embedding
# from .supabase_client import get_supabase
# from .utils import (
#     load_image_from_bytes,
#     fetch_bytes_from_supabase_path,
#     fetch_bytes_from_url,
#     public_url_to_storage_path,
# )
# from .face_engine import face_engine

# # -----------------------------------------------------------------------------
# # Helpers
# # -----------------------------------------------------------------------------


# def _normalize_supabase_relpath(p: str) -> str:
#     """
#     Normaliza um caminho do Storage para o formato relativo do bucket:
#     - remove barras iniciais
#     - aceita prefixo 'uploads/' e remove
#     """
#     p = (p or "").strip().lstrip("/")
#     if not p:
#         return p
#     if p.lower().startswith("uploads/"):
#         p = p.split("uploads/", 1)[1]
#     return p


# _DRIVE_FILE_ID = re.compile(r"/d/([a-zA-Z0-9_-]+)")
# _DRIVE_ID_QUERY = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


# def _to_google_drive_download(url: str) -> str:
#     """
#     Converte links 'view' do Google Drive em links de download direto.
#     - https://drive.google.com/file/d/FILE_ID/view?usp=sharing  ->  uc?export=download&id=FILE_ID
#     - https://drive.google.com/open?id=FILE_ID                 ->  uc?export=download&id=FILE_ID
#     """
#     if "drive.google.com" not in url:
#         return url
#     file_id = None
#     m = _DRIVE_FILE_ID.search(url)
#     if m:
#         file_id = m.group(1)
#     else:
#         m = _DRIVE_ID_QUERY.search(url)
#         if m:
#             file_id = m.group(1)
#     if file_id:
#         return f"https://drive.google.com/uc?export=download&id={file_id}"
#     return url


# async def _maybe_await(v):
#     if hasattr(v, "__await__"):
#         return await v
#     return v


# async def _fetch_image_bytes(
#     *,
#     supabase_path: Optional[str] = None,
#     image_url: Optional[str] = None,
#     image_b64: Optional[str] = None,
# ) -> bytes:
#     """
#     Padroniza todas as entradas de imagem.
#     Prioridade:
#       1) supabase_path (aceita caminho relativo, URL pública do Supabase, com/sem 'uploads/')
#       2) image_url (se for Supabase público -> usa Storage; se for Drive 'view' -> converte p/ download)
#       3) image_b64
#     """
#     # 1) SUPABASE_PATH
#     if supabase_path:
#         val = supabase_path.strip()
#         # Se vier uma URL pública do Supabase, transformamos em (bucket, relpath)
#         parsed = public_url_to_storage_path(val)
#         if parsed and parsed[0] == settings.SUPABASE_STORAGE_BUCKET:
#             _, rel = parsed
#             rel = _normalize_supabase_relpath(rel)
#             b = await _maybe_await(fetch_bytes_from_supabase_path(rel))
#             if not b:
#                 raise FileNotFoundError(f"Objeto não encontrado no Storage: {rel}")
#             return b
#         # Se vier um caminho relativo, normalizamos
#         rel = _normalize_supabase_relpath(val)
#         b = await _maybe_await(fetch_bytes_from_supabase_path(rel))
#         if not b:
#             raise FileNotFoundError(f"Objeto não encontrado no Storage: {rel}")
#         return b

#     # 2) IMAGE_URL
#     if image_url:
#         url = image_url.strip()
#         # Se for URL pública do Supabase, baixe via Storage (mais robusto)
#         parsed = public_url_to_storage_path(url)
#         if parsed and parsed[0] == settings.SUPABASE_STORAGE_BUCKET:
#             _, rel = parsed
#             rel = _normalize_supabase_relpath(rel)
#             b = await _maybe_await(fetch_bytes_from_supabase_path(rel))
#             if not b:
#                 raise FileNotFoundError(f"Objeto não encontrado no Storage: {rel}")
#             return b
#         # Se for Google Drive 'view', converte p/ download
#         url = _to_google_drive_download(url)
#         b = await _maybe_await(fetch_bytes_from_url(url))
#         if not b:
#             raise ValueError("Falha ao baixar URL da imagem")
#         return b

#     # 3) IMAGE_B64
#     if image_b64:
#         try:
#             return base64.b64decode(image_b64, validate=True)
#         except Exception as e:
#             raise ValueError(f"Base64 inválido: {e}")

#     raise ValueError("Envie supabase_path, image_url ou image_b64.")


# async def _read_upload_bytes(file: UploadFile) -> bytes:
#     data = await file.read()
#     if not data:
#         raise ValueError("Arquivo vazio.")
#     return data


# async def _extract_single_embedding(img_bytes: bytes) -> Optional[List[float]]:
#     img = load_image_from_bytes(img_bytes)
#     faces = face_engine.extract_embeddings(img, max_faces=1)
#     if not faces:
#         return None
#     emb = faces[0]["embedding"]
#     if emb is None:
#         return None
#     return emb.astype("float32").tolist()


# def _distance(emb_a: List[float], emb_b: List[float]) -> float:
#     s = 0.0
#     for x, y in zip(emb_a, emb_b):
#         s += float(x) * float(y)
#     return float(1.0 - s)


# def _fetch_member_name_from_supabase(member_id: str) -> Optional[str]:
#     try:
#         sb = get_supabase()
#         sel = f"{settings.MEMBERS_NAME_COLUMN}"
#         resp = (
#             sb.table(settings.MEMBERS_TABLE)
#             .select(sel)
#             .eq(settings.MEMBERS_ID_COLUMN, member_id)
#             .limit(1)
#             .execute()
#         )
#         rows = resp.data or []
#         if rows:
#             return rows[0].get(settings.MEMBERS_NAME_COLUMN)
#     except Exception:
#         pass
#     return None


# # -----------------------------------------------------------------------------
# # Schemas (JSON)
# # -----------------------------------------------------------------------------


# class IndexResponse(BaseModel):
#     ok: bool
#     result: Optional[Dict[str, int]] = None
#     error: Optional[str] = None


# class IdentityRequest(BaseModel):
#     supabase_path: Optional[str] = Field(None, description="Caminho/URL do Storage")
#     image_url: Optional[str] = Field(None, description="URL pública (HTTP/HTTPS)")
#     image_b64: Optional[str] = Field(None, description="Imagem em Base64")
#     top_k: int = Field(1, ge=1, le=10, description="Quantos candidatos retornar")


# class IdentityCandidate(BaseModel):
#     member_id: str
#     distance: float
#     matched: bool
#     name: Optional[str] = None


# class IdentityResponse(BaseModel):
#     ok: bool
#     threshold: float
#     candidates: List[IdentityCandidate] = []


# class EnrollRequest(BaseModel):
#     member_id: str
#     supabase_path: Optional[str] = None
#     image_url: Optional[str] = None
#     image_b64: Optional[str] = None


# class VerifyRequest(BaseModel):
#     member_id: str
#     supabase_path: Optional[str] = None
#     image_url: Optional[str] = None
#     image_b64: Optional[str] = None


# class CompareRequest(BaseModel):
#     a_supabase_path: Optional[str] = None
#     a_image_url: Optional[str] = None
#     a_image_b64: Optional[str] = None
#     b_supabase_path: Optional[str] = None
#     b_image_url: Optional[str] = None
#     b_image_b64: Optional[str] = None


# class CompareResponse(BaseModel):
#     ok: bool
#     distance: float
#     threshold: float
#     is_same: bool


# # -----------------------------------------------------------------------------
# # FastAPI + CORS
# # -----------------------------------------------------------------------------

# app = FastAPI(
#     title="svc-face-recon",
#     description="Microserviço de reconhecimento facial (Supabase Storage + pgvector).",
#     version="1.0.0",
#     openapi_tags=[
#         {"name": "ops", "description": "Operações de saúde/diagnóstico do serviço."},
#         {
#             "name": "face",
#             "description": "Reconhecimento facial: indexação, identificação, verificação e comparação.",
#         },
#     ],
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -----------------------------------------------------------------------------
# # OPS
# # -----------------------------------------------------------------------------


# @app.get("/live", tags=["ops"])
# async def live():
#     return {"status": "live"}


# @app.get("/health", tags=["ops"])
# async def health():
#     try:
#         with get_conn() as conn:
#             with conn.cursor() as cur:
#                 cur.execute("SELECT 1;")
#                 cur.fetchone()
#         return {"ok": True}
#     except Exception as e:
#         return {"ok": False, "error": str(e)}


# @app.get("/ready", tags=["ops"])
# async def ready():
#     return {"ready": True, "cache_embeddings": len(mem_index.embeddings)}


# @app.get("/ops/status", tags=["ops"])
# async def ops_status():
#     return {
#         "service": "svc-face-recon",
#         "threshold": settings.FACE_RECOGNITION_THRESHOLD,
#         "cache_embeddings": len(mem_index.embeddings),
#         "db_url_host_hint": settings.DATABASE_URL.split("@")[-1].split("/")[0]
#         if "@" in settings.DATABASE_URL
#         else settings.DATABASE_URL,
#     }


# # -----------------------------------------------------------------------------
# # FACE — JSON
# # -----------------------------------------------------------------------------


# @app.post("/index", response_model=IndexResponse, tags=["face"])
# async def index_all():
#     try:
#         result = await build_index_from_members()
#         return IndexResponse(ok=True, result=result)
#     except Exception as e:
#         return IndexResponse(ok=False, error=str(e))


# @app.post("/enroll", tags=["face"])
# async def enroll(req: EnrollRequest):
#     try:
#         b = await _fetch_image_bytes(
#             supabase_path=req.supabase_path,
#             image_url=req.image_url,
#             image_b64=req.image_b64,
#         )
#         emb = await _extract_single_embedding(b)
#         if emb is None:
#             raise HTTPException(status_code=400, detail="Nenhum rosto detectado.")
#         upsert_member_embedding(req.member_id, face_engine.to_numpy(emb))
#         mem_index.rebuild()
#         return {"ok": True, "member_id": req.member_id}
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/identify", response_model=IdentityResponse, tags=["face"])
# async def identity(req: IdentityRequest):
#     thr = float(settings.FACE_RECOGNITION_THRESHOLD)
#     try:
#         b = await _fetch_image_bytes(
#             supabase_path=req.supabase_path,
#             image_url=req.image_url,
#             image_b64=req.image_b64,
#         )
#         emb = await _extract_single_embedding(b)
#         if emb is None:
#             return IdentityResponse(ok=False, threshold=thr, candidates=[])
#         cands: List[Tuple[str, float]] = []
#         for mid, ref in mem_index.embeddings.items():
#             d = _distance(emb, ref.tolist())
#             cands.append((mid, d))
#         cands.sort(key=lambda x: x[1])
#         top = cands[: max(1, req.top_k)]
#         out: List[IdentityCandidate] = [
#             IdentityCandidate(
#                 member_id=mid,
#                 distance=dist,
#                 matched=dist <= thr,
#                 name=_fetch_member_name_from_supabase(mid),
#             )
#             for mid, dist in top
#         ]
#         return IdentityResponse(ok=True, threshold=thr, candidates=out)
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/verify", tags=["face"])
# async def verify(req: VerifyRequest):
#     thr = float(settings.FACE_RECOGNITION_THRESHOLD)
#     try:
#         ref = mem_index.embeddings.get(req.member_id)
#         if ref is None:
#             raise HTTPException(
#                 status_code=404, detail="member_id sem embedding no cache/banco."
#             )
#         b = await _fetch_image_bytes(
#             supabase_path=req.supabase_path,
#             image_url=req.image_url,
#             image_b64=req.image_b64,
#         )
#         emb = await _extract_single_embedding(b)
#         if emb is None:
#             raise HTTPException(status_code=400, detail="Nenhum rosto detectado.")
#         dist = _distance(emb, ref.tolist())
#         return {
#             "ok": True,
#             "member_id": req.member_id,
#             "name": _fetch_member_name_from_supabase(req.member_id),
#             "distance": dist,
#             "threshold": thr,
#             "matched": dist <= thr,
#         }
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/compare", response_model=CompareResponse, tags=["face"])
# async def compare(req: CompareRequest):
#     thr = float(settings.FACE_RECOGNITION_THRESHOLD)
#     try:
#         a_bytes = await _fetch_image_bytes(
#             supabase_path=req.a_supabase_path,
#             image_url=req.a_image_url,
#             image_b64=req.a_image_b64,
#         )
#         b_bytes = await _fetch_image_bytes(
#             supabase_path=req.b_supabase_path,
#             image_url=req.b_image_url,
#             image_b64=req.b_image_b64,
#         )
#         emb_a = await _extract_single_embedding(a_bytes)
#         emb_b = await _extract_single_embedding(b_bytes)
#         if emb_a is None or emb_b is None:
#             raise HTTPException(
#                 status_code=400, detail="Não foi possível extrair rosto em A ou B."
#             )
#         dist = _distance(emb_a, emb_b)
#         return CompareResponse(
#             ok=True, distance=dist, threshold=thr, is_same=(dist <= thr)
#         )
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # -----------------------------------------------------------------------------
# # FACE — MULTIPART/FILE (para escolher arquivo no Swagger)
# # -----------------------------------------------------------------------------


# @app.post("/enroll/file", tags=["face"])
# async def enroll_file(
#     member_id: str = Form(...),
#     image: UploadFile = File(...),
# ):
#     try:
#         b = await _read_upload_bytes(image)
#         emb = await _extract_single_embedding(b)
#         if emb is None:
#             raise HTTPException(status_code=400, detail="Nenhum rosto detectado.")
#         upsert_member_embedding(member_id, face_engine.to_numpy(emb))
#         mem_index.rebuild()
#         return {"ok": True, "member_id": member_id}
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/identify/file", response_model=IdentityResponse, tags=["face"])
# async def identity_file(
#     image: UploadFile = File(...),
#     top_k: int = Form(1, ge=1, le=10),
# ):
#     thr = float(settings.FACE_RECOGNITION_THRESHOLD)
#     try:
#         b = await _read_upload_bytes(image)
#         emb = await _extract_single_embedding(b)
#         if emb is None:
#             return IdentityResponse(ok=False, threshold=thr, candidates=[])
#         cands: List[Tuple[str, float]] = []
#         for mid, ref in mem_index.embeddings.items():
#             d = _distance(emb, ref.tolist())
#             cands.append((mid, d))
#         cands.sort(key=lambda x: x[1])
#         top = cands[: max(1, top_k)]
#         out: List[IdentityCandidate] = [
#             IdentityCandidate(
#                 member_id=mid,
#                 distance=dist,
#                 matched=dist <= thr,
#                 name=_fetch_member_name_from_supabase(mid),
#             )
#             for mid, dist in top
#         ]
#         return IdentityResponse(ok=True, threshold=thr, candidates=out)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/verify/file", tags=["face"])
# async def verify_file(
#     member_id: str = Form(...),
#     image: UploadFile = File(...),
# ):
#     thr = float(settings.FACE_RECOGNITION_THRESHOLD)
#     try:
#         ref = mem_index.embeddings.get(member_id)
#         if ref is None:
#             raise HTTPException(
#                 status_code=404, detail="member_id sem embedding no cache/banco."
#             )
#         b = await _read_upload_bytes(image)
#         emb = await _extract_single_embedding(b)
#         if emb is None:
#             raise HTTPException(status_code=400, detail="Nenhum rosto detectado.")
#         dist = _distance(emb, ref.tolist())
#         return {
#             "ok": True,
#             "member_id": member_id,
#             "name": _fetch_member_name_from_supabase(member_id),
#             "distance": dist,
#             "threshold": thr,
#             "matched": dist <= thr,
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/compare/files", response_model=CompareResponse, tags=["face"])
# async def compare_files(
#     image_a: UploadFile = File(...),
#     image_b: UploadFile = File(...),
# ):
#     thr = float(settings.FACE_RECOGNITION_THRESHOLD)
#     try:
#         a_bytes = await _read_upload_bytes(image_a)
#         b_bytes = await _read_upload_bytes(image_b)
#         emb_a = await _extract_single_embedding(a_bytes)
#         emb_b = await _extract_single_embedding(b_bytes)
#         if emb_a is None or emb_b is None:
#             raise HTTPException(
#                 status_code=400, detail="Não foi possível extrair rosto em A ou B."
#             )
#         dist = _distance(emb_a, emb_b)
#         return CompareResponse(
#             ok=True, distance=dist, threshold=thr, is_same=(dist <= thr)
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # -----------------------------------------------------------------------------
# # Startup
# # -----------------------------------------------------------------------------


# @app.on_event("startup")
# async def on_startup():
#     await asyncio.sleep(1.5)
#     try:
#         mem_index.rebuild()
#         print("[INFO] Índice em memória carregado a partir do banco.")
#     except Exception as e:
#         print(f"[WARN] Falha ao reconstruir índice inicial: {e}")
