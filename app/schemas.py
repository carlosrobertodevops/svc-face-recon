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


# ------------- INDEX (/index) -------------
class IndexProgress(BaseModel):
    started_at: Optional[int] = Field(None, description="epoch seconds")
    duration_sec: Optional[int] = Field(None, description="duração total em segundos")
    percent: Optional[float] = Field(None, description="0–100 (se aplicável)")
    state: Optional[str] = Field(None, description="running|done|error")


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


# ------------- FACE (JSON) -------------
class EnrollRequest(BaseModel):
    member_id: str
    supabase_path: Optional[str] = None
    image_url: Optional[str] = None
    image_b64: Optional[str] = None
    storage_path: Optional[str] = None  # deprecated


class EnrollResponse(BaseModel):
    ok: bool
    member_id: Optional[str] = None
    error: Optional[str] = None


class IdentityRequest(BaseModel):
    supabase_path: Optional[str] = None
    image_url: Optional[str] = None
    image_b64: Optional[str] = None
    top_k: int = Field(1, ge=1, le=10)
    storage_path: Optional[str] = None  # deprecated
    max_candidates: Optional[int] = None  # deprecated


class IdentityCandidate(BaseModel):
    member_id: str
    distance: float
    matched: bool
    name: Optional[str] = None
    photo_path: Optional[str] = None


class IdentityResponse(BaseModel):
    ok: bool
    threshold: float
    candidates: List[IdentityCandidate] = []


class VerifyRequest(BaseModel):
    member_id: str
    supabase_path: Optional[str] = None
    image_url: Optional[str] = None
    image_b64: Optional[str] = None
    storage_path: Optional[str] = None  # deprecated


class VerifyResponse(BaseModel):
    ok: bool
    member_id: str
    name: Optional[str] = None
    distance: float
    threshold: float
    matched: bool
    photo_path: Optional[str] = None


class CompareRequest(BaseModel):
    a_supabase_path: Optional[str] = None
    a_image_url: Optional[str] = None
    a_image_b64: Optional[str] = None
    b_supabase_path: Optional[str] = None
    b_image_url: Optional[str] = None
    b_image_b64: Optional[str] = None
    image_a_storage_path: Optional[str] = None  # deprecated
    image_a_url: Optional[str] = None  # deprecated
    image_b_storage_path: Optional[str] = None  # deprecated
    image_b_url: Optional[str] = None  # deprecated


class CompareResponse(BaseModel):
    ok: bool
    distance: float
    threshold: float
    is_same: bool
