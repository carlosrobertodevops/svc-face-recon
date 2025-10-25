# app/schema.py
from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


# ---------------- OPS ----------------


class OpsStatus(BaseModel):
    service: str = Field(default="svc-face-recon")
    live: bool = True
    ready: bool = True
    version: str = "1.0.0"
    cache_embeddings: int = 0
    threshold: float | None = None
    db_url_host_hint: str | None = None


# ------------- FACE / INDEX -------------


class IndexProgress(BaseModel):
    """Progresso do /index armazenado também no Redis (quando disponível)."""

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


# ------------- ENROLL / IDENTIFY / VERIFY / COMPARE -------------


class EnrollRequest(BaseModel):
    member_id: str
    supabase_path: Optional[str] = None
    image_url: Optional[str] = None
    image_b64: Optional[str] = None


class EnrollResponse(BaseModel):
    ok: bool
    member_id: Optional[str] = None
    error: Optional[str] = None


class IdentityRequest(BaseModel):
    supabase_path: Optional[str] = None
    image_url: Optional[str] = None
    image_b64: Optional[str] = None
    top_k: int = Field(1, ge=1, le=10, description="Quantos candidatos retornar")


class IdentityCandidate(BaseModel):
    member_id: str
    distance: float
    matched: bool
    name: Optional[str] = None
    photo_path: Optional[str] = None  # <- adicionado


class IdentityResponse(BaseModel):
    ok: bool
    threshold: float
    candidates: List[IdentityCandidate] = []


class VerifyRequest(BaseModel):
    member_id: str
    supabase_path: Optional[str] = None
    image_url: Optional[str] = None
    image_b64: Optional[str] = None


class VerifyResponse(BaseModel):
    ok: bool
    member_id: str
    name: Optional[str] = None
    distance: float
    threshold: float
    matched: bool
    photo_path: Optional[str] = None  # <- adicionado


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
