# app/main.py
# =============================================================================
# Arquivo: main.py
# Versão: v1
# Objetivo: Reconhecimento facial
# Funções/métodos:
# - endpoints do microserviço de reconhecimento facial
# =============================================================================

# app/main.py
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from typing import Optional, Dict, List, Tuple, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

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

# >>> ADIÇÃO: importa o montador de docs custom
from .docs import mount_docs_routes

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize_supabase_relpath(p: str) -> str:
    p = (p or "").strip().lstrip("/")
    if not p:
        return p
    if p.lower().startswith("uploads/"):
        p = p.split("uploads/", 1)[1]
    return p

_DRIVE_FILE_ID = re.compile(r"/d/([a-zA-Z0-9_-]+)")
_DRIVE_ID_QUERY = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")

def _to_google_drive_download(url: str) -> str:
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

def _fetch_member_name_from_supabase(member_id: str) -> Optional[str]:
    try:
        sb = get_supabase()
        sel = f"{settings.MEMBERS_NAME_COLUMN}"
        resp = (
            sb.table(settings.MEMBERS_TABLE)
            .select(sel)
            .eq(settings.MEMBERS_ID_COLUMN, member_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if rows:
            return rows[0].get(settings.MEMBERS_NAME_COLUMN)
    except Exception:
        pass
    return None

def _public_url_from_storage_path(relpath: str) -> Optional[str]:
    try:
        sb = get_supabase()
        res = sb.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(relpath)
        if isinstance(res, dict):
            data = res.get("data") or {}
            url = data.get("publicUrl") or data.get("publicURL")
            if url:
                return url
        elif isinstance(res, str):
            return res
    except Exception:
        pass
    return None

# ---------- FIX: normalizador robusto para a coluna `fotos_path` ---------------

_URL_RE = re.compile(r"https?://[^\s'\"\]]+", re.IGNORECASE)
_PATH_RE = re.compile(
    r"(?:uploads/)?[^\s'\"\]]+\.(?:jpg|jpeg|png|webp)", re.IGNORECASE
)

def _coerce_photo_value_to_public_url(raw: Any) -> Optional[str]:
    """
    Converte o valor cru de `fotos_path` (ou equivalente) em uma URL pública.
    """
    if raw is None:
        return None

    if isinstance(raw, (list, tuple)):
        for it in raw:
            url = _coerce_photo_value_to_public_url(it)
            if url:
                return url
        return None

    if isinstance(raw, dict):
        for k in ("path", "url", "public_url", "publicUrl", "Key", "name"):
            if k in raw and raw[k]:
                return _coerce_photo_value_to_public_url(raw[k])
        return None

    s = str(raw).strip()
    if not s:
        return None

    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            j = json.loads(s)
            return _coerce_photo_value_to_public_url(j)
        except Exception:
            pass

    m = _URL_RE.search(s)
    if m:
        return m.group(0)

    m = _PATH_RE.search(s)
    if m:
        rel = _normalize_supabase_relpath(m.group(0))
        return _public_url_from_storage_path(rel)

    s_clean = s.strip(" \t\r\n'\"[]")
    if s_clean:
        rel = _normalize_supabase_relpath(s_clean)
        return _public_url_from_storage_path(rel)

    return None

def _member_photo_public_url(member_id: str) -> Optional[str]:
    col = getattr(settings, "MEMBERS_PHOTOS_COLUMN", None) or getattr(
        settings, "MEMBERS_PHOTO_COLUMN", None
    )
    if col:
        try:
            sb = get_supabase()
            resp = (
                sb.table(settings.MEMBERS_TABLE)
                .select(col)
                .eq(settings.MEMBERS_ID_COLUMN, member_id)
                .limit(1)
                .execute()
            )
            rows = resp.data or []
            if rows:
                raw = (rows[0] or {}).get(col)
                url = _coerce_photo_value_to_public_url(raw)
                if url:
                    return url
        except Exception:
            pass

    candidates = [
        f"{member_id}.jpg",
        f"{member_id}.jpeg",
        f"{member_id}.png",
        f"{member_id}.webp",
        f"uploads/{member_id}.jpg",
        f"uploads/{member_id}.jpeg",
        f"uploads/{member_id}.png",
        f"uploads/{member_id}.webp",
    ]
    for rel in candidates:
        rel = _normalize_supabase_relpath(rel)
        url = _public_url_from_storage_path(rel)
        if url:
            return url
    return None

def _make_preview_b64(img_bytes: bytes, max_side: int = 256) -> str:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _make_preview_data_url(img_bytes: bytes, max_side: int = 256) -> str:
    return "data:image/jpeg;base64," + _make_preview_b64(img_bytes, max_side=max_side)

# -----------------------------------------------------------------------------
# Schemas (JSON) + novos campos de imagem
# -----------------------------------------------------------------------------

class IndexProgress(BaseModel):
    started_at: int
    duration_sec: Optional[int] = None
    total: Optional[int] = None
    processed: Optional[int] = None
    indexed: Optional[int] = None
    skipped_no_photo: Optional[int] = None
    skipped_no_face: Optional[int] = None
    errors: Optional[int] = None
    cache_reloaded: Optional[bool] = None
    state: Optional[str] = None
    finished_at: Optional[int] = None
    percent: Optional[float] = None

class IndexResult(BaseModel):
    total: int
    indexed: int
    skipped_no_photo: int
    skipped_no_face: int
    errors: int
    cache_reloaded: bool
    progress: Optional[IndexProgress] = None

class IndexResponse(BaseModel):
    ok: bool
    result: Optional[IndexResult] = None
    error: Optional[str] = None

class IdentityRequest(BaseModel):
    supabase_path: Optional[str] = Field(None, description="Caminho/URL do Storage")
    image_url: Optional[str] = Field(None, description="URL pública (HTTP/HTTPS)")
    image_b64: Optional[str] = Field(None, description="Imagem em Base64")
    top_k: int = Field(1, ge=1, le=10, description="Quantos candidatos retornar")

class IdentityCandidate(BaseModel):
    member_id: str
    distance: float
    matched: bool
    name: Optional[str] = None
    photo_url: Optional[str] = None  # NOVO

class IdentityResponse(BaseModel):
    ok: bool
    threshold: float
    candidates: List[IdentityCandidate] = []

class EnrollRequest(BaseModel):
    member_id: str
    supabase_path: Optional[str] = None
    image_url: Optional[str] = None
    image_b64: Optional[str] = None

class VerifyRequest(BaseModel):
    member_id: str
    supabase_path: Optional[str] = None
    image_url: Optional[str] = None
    image_b64: Optional[str] = None

class CompareRequest(BaseModel):
    a_supabase_path: Optional[str] = None
    a_image_url: Optional[str] = None
    a_image_b64: Optional[str] = None
    b_supabase_path: Optional[str] = None
    b_image_url: Optional[str] = None
    b_image_b64: Optional[str] = None

class CompareResponse(BaseModel):
    ok: bool
    distance: float
    threshold: float
    is_same: bool
    a_preview_b64: Optional[str] = None  # NOVO
    b_preview_b64: Optional[str] = None  # NOVO

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
    # >>> ADIÇÃO: desativa Swagger nativo para usar nosso /docs custom
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
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
    try:
        result = await build_index_from_members()
        return IndexResponse(ok=True, result=result)
    except Exception as e:
        return IndexResponse(ok=False, error=str(e))

@app.post("/enroll", tags=["face"])
async def enroll(req: EnrollRequest):
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
        return {"ok": True, "member_id": req.member_id}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/identify", response_model=IdentityResponse, tags=["face"])
async def identity(req: IdentityRequest):
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
        cands: List[Tuple[str, float]] = []
        for mid, ref in mem_index.embeddings.items():
            d = _distance(emb, ref.tolist())
            cands.append((mid, d))
        cands.sort(key=lambda x: x[1])
        top = cands[: max(1, req.top_k)]
        out: List[IdentityCandidate] = [
            IdentityCandidate(
                member_id=mid,
                distance=dist,
                matched=dist <= thr,
                name=_fetch_member_name_from_supabase(mid),
                photo_url=_member_photo_public_url(mid),
            )
            for mid, dist in top
        ]
        return IdentityResponse(ok=True, threshold=thr, candidates=out)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail:str(e))

@app.post("/verify", tags=["face"])
async def verify(req: VerifyRequest):
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
        return {
            "ok": True,
            "member_id": req.member_id,
            "name": _fetch_member_name_from_supabase(req.member_id),
            "distance": dist,
            "threshold": thr,
            "matched": dist <= thr,
            "photo_url": _member_photo_public_url(req.member_id),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail:str(e))

@app.post("/compare", response_model=CompareResponse, tags=["face"])
async def compare(req: CompareRequest):
    thr = float(settings.FACE_RECOGNITION_THRESHOLD)
    try:
        a_bytes = await _fetch_image_bytes(
            supabase_path=req.a_supabase_path,
            image_url=req.a_image_url,
            image_b64=req.a_image_b64,
        )
        b_bytes = await _fetch_image_bytes(
            supabase_path=req.b_supabase_path,
            image_url=req.b_image_url,
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
            ok=True,
            distance=dist,
            threshold=thr,
            is_same=(dist <= thr),
            a_preview_b64=_make_preview_data_url(a_bytes),
            b_preview_b64=_make_preview_data_url(b_bytes),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail:str(e))

# -----------------------------------------------------------------------------
# FACE — MULTIPART/FILE (Swagger)
# -----------------------------------------------------------------------------

@app.post("/enroll/file", tags=["face"])
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
        return {"ok": True, "member_id": member_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail:str(e))

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
        out: List[IdentityCandidate] = [
            IdentityCandidate(
                member_id=mid,
                distance=dist,
                matched=dist <= thr,
                name=_fetch_member_name_from_supabase(mid),
                photo_url=_member_photo_public_url(mid),
            )
            for mid, dist in top
        ]
        return IdentityResponse(ok=True, threshold=thr, candidates=out)
    except Exception as e:
        raise HTTPException(status_code=500, detail:str(e))

@app.post("/verify/file", tags=["face"])
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
        return {
            "ok": True,
            "member_id": member_id,
            "name": _fetch_member_name_from_supabase(member_id),
            "distance": dist,
            "threshold": thr,
            "matched": dist <= thr,
            "photo_url": _member_photo_public_url(member_id),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail:str(e))

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
            ok=True,
            distance=dist,
            threshold=thr,
            is_same=(dist <= thr),
            a_preview_b64=_make_preview_data_url(a_bytes),
            b_preview_b64=_make_preview_data_url(b_bytes),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail:str(e))

# -----------------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    await asyncio.sleep(1.5)
    try:
        mem_index.rebuild()
        print("[INFO] Índice em memória carregado a partir do banco.")
    except Exception as e:
        print(f"[WARN] Falha ao reconstruir índice inicial: {e}")

# >>> ADIÇÃO: monta os docs custom (depois de incluir TODAS as rotas)
mount_docs_routes(app)


# # app/main.py
# from __future__ import annotations

# import asyncio
# import base64
# import io
# import json
# import re
# from typing import Optional, Dict, List, Tuple, Any

# from fastapi import FastAPI, HTTPException, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from PIL import Image

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
#     p = (p or "").strip().lstrip("/")
#     if not p:
#         return p
#     if p.lower().startswith("uploads/"):
#         p = p.split("uploads/", 1)[1]
#     return p

# _DRIVE_FILE_ID = re.compile(r"/d/([a-zA-Z0-9_-]+)")
# _DRIVE_ID_QUERY = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")

# def _to_google_drive_download(url: str) -> str:
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
#     # 1) SUPABASE_PATH
#     if supabase_path:
#         val = supabase_path.strip()
#         parsed = public_url_to_storage_path(val)
#         if parsed and parsed[0] == settings.SUPABASE_STORAGE_BUCKET:
#             _, rel = parsed
#             rel = _normalize_supabase_relpath(rel)
#             b = await _maybe_await(fetch_bytes_from_supabase_path(rel))
#             if not b:
#                 raise FileNotFoundError(f"Objeto não encontrado no Storage: {rel}")
#             return b
#         rel = _normalize_supabase_relpath(val)
#         b = await _maybe_await(fetch_bytes_from_supabase_path(rel))
#         if not b:
#             raise FileNotFoundError(f"Objeto não encontrado no Storage: {rel}")
#         return b

#     # 2) IMAGE_URL
#     if image_url:
#         url = image_url.strip()
#         parsed = public_url_to_storage_path(url)
#         if parsed and parsed[0] == settings.SUPABASE_STORAGE_BUCKET:
#             _, rel = parsed
#             rel = _normalize_supabase_relpath(rel)
#             b = await _maybe_await(fetch_bytes_from_supabase_path(rel))
#             if not b:
#                 raise FileNotFoundError(f"Objeto não encontrado no Storage: {rel}")
#             return b
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

# def _public_url_from_storage_path(relpath: str) -> Optional[str]:
#     try:
#         sb = get_supabase()
#         res = sb.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(relpath)
#         if isinstance(res, dict):
#             data = res.get("data") or {}
#             url = data.get("publicUrl") or data.get("publicURL")
#             if url:
#                 return url
#         elif isinstance(res, str):
#             return res
#     except Exception:
#         pass
#     return None

# # ---------- FIX: normalizador robusto para a coluna `fotos_path` ---------------

# _URL_RE = re.compile(r"https?://[^\s'\"\]]+", re.IGNORECASE)
# _PATH_RE = re.compile(
#     r"(?:uploads/)?[^\s'\"\]]+\.(?:jpg|jpeg|png|webp)", re.IGNORECASE
# )

# def _coerce_photo_value_to_public_url(raw: Any) -> Optional[str]:
#     """
#     Converte o valor cru de `fotos_path` (ou equivalente) em uma URL pública:
#     - aceita URL direta
#     - aceita caminho do Storage (com/sem 'uploads/')
#     - aceita string com colchetes/aspas (ex.: "['uploads/.../123.png']")
#     - tenta fazer parse de JSON (lista/dict) e pega o primeiro caminho/URL
#     - aceita lista/tupla/dict Python
#     """
#     if raw is None:
#         return None

#     # lista/tupla -> pega primeiro não-vazio
#     if isinstance(raw, (list, tuple)):
#         for it in raw:
#             url = _coerce_photo_value_to_public_url(it)
#             if url:
#                 return url
#         return None

#     # dict -> tenta chaves comuns
#     if isinstance(raw, dict):
#         for k in ("path", "url", "public_url", "publicUrl", "Key", "name"):
#             if k in raw and raw[k]:
#                 return _coerce_photo_value_to_public_url(raw[k])
#         return None

#     # string
#     s = str(raw).strip()
#     if not s:
#         return None

#     # tenta decodificar JSON (lista/dict) serializado em string
#     if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
#         try:
#             j = json.loads(s)
#             return _coerce_photo_value_to_public_url(j)
#         except Exception:
#             # segue o fluxo com regex/normalização
#             pass

#     # primeiro: se já tem uma URL http(s) dentro da string, usa a primeira
#     m = _URL_RE.search(s)
#     if m:
#         return m.group(0)

#     # senão, tenta extrair um caminho de arquivo válido
#     m = _PATH_RE.search(s)
#     if m:
#         rel = _normalize_supabase_relpath(m.group(0))
#         return _public_url_from_storage_path(rel)

#     # por fim, se não casou com regex, trata a string inteira como caminho
#     # removendo colchetes/aspas soltas
#     s_clean = s.strip(" \t\r\n'\"[]")
#     if s_clean:
#         rel = _normalize_supabase_relpath(s_clean)
#         return _public_url_from_storage_path(rel)

#     return None

# def _member_photo_public_url(member_id: str) -> Optional[str]:
#     """
#     Resolve a foto pública do membro.
#     Prioridade:
#       1) Coluna definida em .env (MEMBERS_PHOTOS_COLUMN ou MEMBERS_PHOTO_COLUMN)
#       2) Fallbacks por convenção {member_id}.jpg|jpeg|png|webp (com/sem uploads/)
#     """
#     col = getattr(settings, "MEMBERS_PHOTOS_COLUMN", None) or getattr(
#         settings, "MEMBERS_PHOTO_COLUMN", None
#     )

#     # 1) tabela/coluna
#     if col:
#         try:
#             sb = get_supabase()
#             resp = (
#                 sb.table(settings.MEMBERS_TABLE)
#                 .select(col)
#                 .eq(settings.MEMBERS_ID_COLUMN, member_id)
#                 .limit(1)
#                 .execute()
#             )
#             rows = resp.data or []
#             if rows:
#                 raw = (rows[0] or {}).get(col)
#                 url = _coerce_photo_value_to_public_url(raw)
#                 if url:
#                     return url
#         except Exception:
#             pass

#     # 2) fallbacks
#     candidates = [
#         f"{member_id}.jpg",
#         f"{member_id}.jpeg",
#         f"{member_id}.png",
#         f"{member_id}.webp",
#         f"uploads/{member_id}.jpg",
#         f"uploads/{member_id}.jpeg",
#         f"uploads/{member_id}.png",
#         f"uploads/{member_id}.webp",
#     ]
#     for rel in candidates:
#         rel = _normalize_supabase_relpath(rel)
#         url = _public_url_from_storage_path(rel)
#         if url:
#             return url
#     return None

# def _make_preview_b64(img_bytes: bytes, max_side: int = 256) -> str:
#     im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#     im.thumbnail((max_side, max_side))
#     buf = io.BytesIO()
#     im.save(buf, format="JPEG", quality=80)
#     return base64.b64encode(buf.getvalue()).decode("ascii")

# def _make_preview_data_url(img_bytes: bytes, max_side: int = 256) -> str:
#     return "data:image/jpeg;base64," + _make_preview_b64(img_bytes, max_side=max_side)

# # -----------------------------------------------------------------------------
# # Schemas (JSON) + novos campos de imagem
# # -----------------------------------------------------------------------------

# class IndexProgress(BaseModel):
#     started_at: int
#     duration_sec: Optional[int] = None
#     total: Optional[int] = None
#     processed: Optional[int] = None
#     indexed: Optional[int] = None
#     skipped_no_photo: Optional[int] = None
#     skipped_no_face: Optional[int] = None
#     errors: Optional[int] = None
#     cache_reloaded: Optional[bool] = None
#     state: Optional[str] = None
#     finished_at: Optional[int] = None
#     percent: Optional[float] = None

# class IndexResult(BaseModel):
#     total: int
#     indexed: int
#     skipped_no_photo: int
#     skipped_no_face: int
#     errors: int
#     cache_reloaded: bool
#     progress: Optional[IndexProgress] = None

# class IndexResponse(BaseModel):
#     ok: bool
#     result: Optional[IndexResult] = None
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
#     photo_url: Optional[str] = None  # NOVO

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
#     a_preview_b64: Optional[str] = None  # NOVO
#     b_preview_b64: Optional[str] = None  # NOVO

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
#                 photo_url=_member_photo_public_url(mid),  # usa normalizador
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
#             "photo_url": _member_photo_public_url(req.member_id),  # usa normalizador
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
#             ok=True,
#             distance=dist,
#             threshold=thr,
#             is_same=(dist <= thr),
#             a_preview_b64=_make_preview_data_url(a_bytes),
#             b_preview_b64=_make_preview_data_url(b_bytes),
#         )
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # -----------------------------------------------------------------------------
# # FACE — MULTIPART/FILE (Swagger)
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
#                 photo_url=_member_photo_public_url(mid),  # usa normalizador
#             )
#             for mid, dist in top
#         ]
#         return IdentityResponse(ok=True, threshold=thr, candidates=out)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/verify/file", tags=["face"])
# async def verify_file(  # (se no seu repo já está correto, mantenha)
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
#             "photo_url": _member_photo_public_url(member_id),  # usa normalizador
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
#             ok=True,
#             distance=dist,
#             threshold=thr,
#             is_same=(dist <= thr),
#             a_preview_b64=_make_preview_data_url(a_bytes),
#             b_preview_b64=_make_preview_data_url(b_bytes),
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
