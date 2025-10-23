# app/main.py
from __future__ import annotations
import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator

# Datadog APM (opcional)
if os.getenv("DD_TRACE_ENABLED", "false").lower() in ("1", "true", "yes"):
    from ddtrace import patch, tracer

    patch(fastapi=True, httpx=True, psycopg=True)
    tracer.set_tags(
        {
            "service.name": os.getenv("DD_SERVICE", "svc-face-recon"),
            "env": os.getenv("DD_ENV", "local"),
            "version": os.getenv("DD_VERSION", "v1"),
        }
    )

from .config import settings
from .schemas import IdentifyRequest, IdentifyResponse, VerifyRequest, IndexResponse
from .utils import resolve_image_source, load_image_from_bytes, image_to_data_url
from .face_engine import face_engine, FaceEngine
from .repository import search_top1, upsert_member_embedding
from .indexer import build_index_from_members, mem_index
from .docs import mount_docs_routes


def crop_from_bbox(img: Image.Image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    return img.crop((x1, y1, x2, y2))


TAGS_METADATA = [
    {"name": "ops", "description": "Operações de saúde/diagnóstico do serviço."},
    {
        "name": "face",
        "description": "Reconhecimento facial: indexação, identificação, verificação e comparação.",
    },
]


app = FastAPI(
    title=f"{settings.SERVICE_NAME}",
    version="v1.0.0",
    description="Microserviço de **reconhecimento facial** e processamento de imagem integrado ao Supabase (Storage + pgvector).",
    openapi_url="/openapi.json",
    docs_url=None,
)
app.openapi_tags = TAGS_METADATA


# ------------------- MIDDLEWARES -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------- OPS ENDPOINTS -------------------
@app.get(
    "/live",
    tags=["ops"],
    summary="Live",
    description="Verifica se o processo está vivo.",
)
async def live():
    return {"ok": True}


@app.get(
    "/health", tags=["ops"], summary="Health", description="Checagem simples de saúde."
)
async def health():
    return {"status": "ok", "service": settings.SERVICE_NAME}


@app.get(
    "/ready", tags=["ops"], summary="Ready", description="Pronto para receber tráfego?"
)
async def ready():
    return {"ready": True}


@app.get(
    "/ops/status",
    tags=["ops"],
    summary="Ops Status",
    description="Informações operacionais resumidas.",
)
async def ops_status():
    return {
        "service": settings.SERVICE_NAME,
        "threshold": settings.FACE_RECOGNITION_THRESHOLD,
        "max_faces": settings.MAX_FACES_PER_IMAGE,
    }


# ------------------- FACE ENDPOINTS -------------------
@app.post(
    "/index",
    response_model=IndexResponse,
    tags=["face"],
    summary="Reindexar membros a partir do Storage/DB",
    description=(
        "Lê a tabela configurada (ex.: `public.membros`) com a coluna de fotos (ex.: `fotos_path`), "
        "baixa as imagens do Storage (bucket configurado), gera embeddings (média por membro) e salva em `member_faces` (pgvector). "
        "Ao final, reconstrói o cache em memória para consultas rápidas."
    ),
)
async def index_all():
    try:
        result = build_index_from_members()
        return IndexResponse(**result)
    except Exception as e:
        import traceback

        traceback.print_exc()
        # 🔥 Retorna JSON detalhado com endpoint + mensagem de erro
        raise HTTPException(
            status_code=500,
            detail={
                "endpoint": "/index",
                "error": str(e),
                "hint": "Verifique variáveis de ambiente MEMBERS_* e a conexão com o banco/Storage.",
            },
        )


@app.post(
    "/enroll",
    tags=["face"],
    summary="Cadastrar/atualizar embedding de um membro",
    description="Gera embedding de uma imagem (file, image_url ou supabase_path) e salva em `member_faces`.",
)
async def enroll(
    member_id: str = Form(..., description="ID do membro (uuid/int/bigint)"),
    file: UploadFile | None = File(None, description="Arquivo de imagem (JPEG/PNG)"),
    image_url: Optional[str] = Form(None, description="URL pública/assinada"),
    supabase_path: Optional[str] = Form(
        None, description="Caminho no Storage (ex: membros/123.jpg)"
    ),
):
    file_bytes = await file.read() if file else None
    img_bytes = await resolve_image_source(image_url, supabase_path, file_bytes)
    img = load_image_from_bytes(img_bytes)
    faces = face_engine.extract_embeddings(img, max_faces=1)
    if not faces:
        raise HTTPException(status_code=400, detail="Nenhum rosto detectado na imagem.")
    emb = faces[0]["embedding"]
    try:
        upsert_member_embedding(int(member_id), emb)
    except Exception:
        upsert_member_embedding(member_id, emb)
    mem_index.rebuild()
    return {"member_id": member_id, "status": "enrolled"}


@app.post(
    "/identify",
    response_model=IdentifyResponse,
    tags=["face"],
    summary="Identificar a pessoa mais provável",
    description="Retorna `member_id` mais próximo, distância (cosine) e `matched` baseado no threshold.",
)
async def identify(
    req: IdentifyRequest = Body(
        example={"supabase_path": "membros/123.jpg"},
        description="Fonte da imagem: `supabase_path` ou `image_url`. Alternativamente, envie como `file` multipart.",
    ),
    file: UploadFile | None = File(
        None,
        description="Arquivo de imagem (alternativa a `supabase_path`/`image_url`).",
    ),
):
    file_bytes = await file.read() if file else None
    try:
        img_bytes = await resolve_image_source(
            req.image_url, req.supabase_path, file_bytes
        )
        img = load_image_from_bytes(img_bytes)
        faces = face_engine.extract_embeddings(
            img, max_faces=settings.MAX_FACES_PER_IMAGE
        )
        if not faces:
            return IdentifyResponse(member_id=None, distance=None, matched=False)

        face = max(faces, key=lambda f: f["det_score"])
        q = face["embedding"]

        top = mem_index.top1(q)
        if not top:
            # fallback DB
            res = search_top1(q)
            if not res:
                return IdentifyResponse(member_id=None, distance=None, matched=False)
            member_id, distance = res
        else:
            member_id, distance = top

        matched = distance <= settings.FACE_RECOGNITION_THRESHOLD
        return IdentifyResponse(member_id=member_id, distance=distance, matched=matched)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/verify",
    response_model=IdentifyResponse,
    tags=["face"],
    summary="Verificar se a imagem corresponde a um member_id",
    description="Compara a imagem recebida com o `member_id` informado.",
)
async def verify(
    req: VerifyRequest = Body(
        example={"member_id": "242", "supabase_path": "membros/123.jpg"},
        description="Informe `member_id` e a fonte da imagem.",
    ),
    file: UploadFile | None = File(None, description="Arquivo de imagem (opcional)"),
):
    file_bytes = await file.read() if file else None
    try:
        img_bytes = await resolve_image_source(
            req.image_url, req.supabase_path, file_bytes
        )
        img = load_image_from_bytes(img_bytes)
        faces = face_engine.extract_embeddings(
            img, max_faces=settings.MAX_FACES_PER_IMAGE
        )
        if not faces:
            return IdentifyResponse(member_id=None, distance=None, matched=False)

        face = max(faces, key=lambda f: f["det_score"])
        q = face["embedding"]

        top = mem_index.top1(q)
        if not top:
            res = search_top1(q)
            if not res:
                return IdentifyResponse(member_id=None, distance=None, matched=False)
            found_id, distance = res
        else:
            found_id, distance = top

        matched = (str(found_id) == str(req.member_id)) and (
            distance <= settings.FACE_RECOGNITION_THRESHOLD
        )
        return IdentifyResponse(
            member_id=str(found_id), distance=distance, matched=matched
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/compare",
    tags=["face"],
    summary="Comparar duas imagens (A vs B)",
    description="Retorna distância cosine, `matched`, recortes base64 e metadados das duas faces.",
)
async def compare(
    file_a: UploadFile | None = File(None, description="Imagem A"),
    file_b: UploadFile | None = File(None, description="Imagem B"),
    image_url_a: Optional[str] = Form(None, description="URL (A)"),
    image_url_b: Optional[str] = Form(None, description="URL (B)"),
    supabase_path_a: Optional[str] = Form(None, description="Storage path (A)"),
    supabase_path_b: Optional[str] = Form(None, description="Storage path (B)"),
):
    try:
        bA = await resolve_image_source(
            image_url_a, supabase_path_a, await file_a.read() if file_a else None
        )
        bB = await resolve_image_source(
            image_url_b, supabase_path_b, await file_b.read() if file_b else None
        )

        imgA, imgB = load_image_from_bytes(bA), load_image_from_bytes(bB)

        facesA = face_engine.extract_embeddings(imgA, max_faces=1)
        facesB = face_engine.extract_embeddings(imgB, max_faces=1)
        if not facesA or not facesB:
            return JSONResponse(
                {"ok": False, "reason": "Rosto não detectado em uma das imagens."},
                status_code=400,
            )

        fA, fB = facesA[0], facesB[0]
        dist = FaceEngine.cosine_distance(fA["embedding"], fB["embedding"])

        cropA = crop_from_bbox(imgA, fA["bbox"])
        cropB = crop_from_bbox(imgB, fB["bbox"])

        return {
            "ok": True,
            "distance": dist,
            "threshold": settings.FACE_RECOGNITION_THRESHOLD,
            "matched": dist <= settings.FACE_RECOGNITION_THRESHOLD,
            "faceA": {
                "det_score": fA["det_score"],
                "bbox": fA["bbox"],
                "kps": fA["kps"],
                "crop_base64": image_to_data_url(cropA),
            },
            "faceB": {
                "det_score": fB["det_score"],
                "bbox": fB["bbox"],
                "kps": fB["kps"],
                "crop_base64": image_to_data_url(cropB),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------- DOCS & METRICS -------------------
mount_docs_routes(app)
Instrumentator().instrument(app).expose(app, include_in_schema=False)
