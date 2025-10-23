from __future__ import annotations

import asyncio
import base64
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .indexer import mem_index, build_index_from_members
from .repository import get_conn, upsert_member_embedding
from .supabase_client import get_supabase
from .utils import (
    load_image_from_bytes,
    fetch_bytes_from_supabase_path,
    fetch_bytes_from_url,
)
from .face_engine import face_engine


# -----------------------------------------------------------------------------
# Utilidades internas
# -----------------------------------------------------------------------------


async def _maybe_await(v):
    """Aceita funções sync ou async (compatível com suas utils)."""
    if hasattr(v, "__await__"):
        return await v
    return v


async def _fetch_image_bytes(
    *,
    supabase_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_b64: Optional[str] = None,
) -> bytes:
    """Padroniza as entradas de imagem."""
    if supabase_path:
        b = await _maybe_await(fetch_bytes_from_supabase_path(supabase_path))
        if not b:
            raise ValueError("Falha ao obter bytes do Supabase Storage")
        return b
    if image_url:
        b = await _maybe_await(fetch_bytes_from_url(image_url))
        if not b:
            raise ValueError("Falha ao baixar URL da imagem")
        return b
    if image_b64:
        try:
            return base64.b64decode(image_b64, validate=True)
        except Exception as e:
            raise ValueError(f"Base64 inválido: {e}")
    raise ValueError("Forneça supabase_path, image_url ou image_b64")


async def _extract_single_embedding(img_bytes: bytes) -> Optional[List[float]]:
    """Extrai um único embedding (normalizado) como lista de float."""
    img = load_image_from_bytes(img_bytes)
    faces = face_engine.extract_embeddings(img, max_faces=1)
    if not faces:
        return None
    emb = faces[0]["embedding"]
    if emb is None:
        return None
    return emb.astype("float32").tolist()


def _distance(emb_a: List[float], emb_b: List[float]) -> float:
    """Distância (1 - dot) para vetores L2-normalizados."""
    # dot
    s = 0.0
    for x, y in zip(emb_a, emb_b):
        s += float(x) * float(y)
    return float(1.0 - s)


def _fetch_member_name_from_supabase(member_id: str) -> Optional[str]:
    """Busca nome no Supabase, se quiser enriquecer /identify e /verify."""
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


# -----------------------------------------------------------------------------
# Modelos (mantendo os esquemas que aparecem no seu Swagger)
# -----------------------------------------------------------------------------


class IndexResponse(BaseModel):
    ok: bool
    result: Optional[Dict[str, int]] = None
    error: Optional[str] = None


class IdentityRequest(BaseModel):
    supabase_path: Optional[str] = Field(
        None, description="Caminho no bucket configurado"
    )
    image_url: Optional[str] = Field(None, description="URL pública da imagem")
    image_b64: Optional[str] = Field(None, description="Imagem em Base64")
    top_k: int = Field(1, ge=1, le=10, description="Quantos candidatos retornar")


class IdentityCandidate(BaseModel):
    member_id: str
    distance: float
    matched: bool
    name: Optional[str] = None


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
    # imagem A
    a_supabase_path: Optional[str] = None
    a_image_url: Optional[str] = None
    a_image_b64: Optional[str] = None
    # imagem B
    b_supabase_path: Optional[str] = None
    b_image_url: Optional[str] = None
    b_image_b64: Optional[str] = None


class CompareResponse(BaseModel):
    ok: bool
    distance: float
    threshold: float
    is_same: bool


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
    """Health + DB ping."""
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
    return {
        "ready": True,
        "cache_embeddings": len(mem_index.embeddings),
    }


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
# FACE
# -----------------------------------------------------------------------------


@app.post("/index", response_model=IndexResponse, tags=["face"])
async def index_all():
    """
    Reindexa a partir do Supabase: baixa fotos, gera embeddings, salva no Postgres e recarrega o cache.
    """
    try:
        result = await build_index_from_members()
        return IndexResponse(ok=True, result=result)
    except Exception as e:
        return IndexResponse(ok=False, error=str(e))


@app.post("/enroll", tags=["face"])
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

        # salva no banco
        upsert_member_embedding(req.member_id, face_engine.to_numpy(emb))
        # recarrega cache
        mem_index.rebuild()

        return {"ok": True, "member_id": req.member_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/identify", response_model=IdentityResponse, tags=["face"])
async def identity(req: IdentityRequest):
    """
    Identifica a pessoa mais provável (ou top_k candidatos) comparando contra o cache em memória.
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
        # distância = 1 - dot
        cands: List[Tuple[str, float]] = []
        for mid, ref in mem_index.embeddings.items():
            d = _distance(emb, ref.tolist())
            cands.append((mid, d))

        cands.sort(key=lambda x: x[1])
        top = cands[: max(1, req.top_k)]

        out: List[IdentityCandidate] = []
        for mid, dist in top:
            out.append(
                IdentityCandidate(
                    member_id=mid,
                    distance=dist,
                    matched=dist <= thr,
                    name=_fetch_member_name_from_supabase(mid),
                )
            )

        return IdentityResponse(ok=True, threshold=thr, candidates=out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify", tags=["face"])
async def verify(req: VerifyRequest):
    """
    Verifica se a imagem pertence ao `member_id` informado.
    """
    thr = float(settings.FACE_RECOGNITION_THRESHOLD)
    try:
        # precisa existir no cache
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
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=CompareResponse, tags=["face"])
async def compare(req: CompareRequest):
    """
    Compara duas imagens (A vs B) e retorna distância e `is_same`.
    """
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
    # pequeno atraso para cenários locais
    await asyncio.sleep(1.5)
    try:
        mem_index.rebuild()
        print("[INFO] Índice em memória carregado a partir do banco.")
    except Exception as e:
        print(f"[WARN] Falha ao reconstruir índice inicial: {e}")
