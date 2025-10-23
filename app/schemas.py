# # app/schemas.py
# from __future__ import annotations
# from typing import Optional
# from pydantic import BaseModel, Field


# class IdentifyRequest(BaseModel):
#     image_url: Optional[str] = None
#     supabase_path: Optional[str] = None


# class IdentifyResponse(BaseModel):
#     member_id: Optional[str]
#     distance: Optional[float]
#     matched: bool


# class VerifyRequest(IdentifyRequest):
#     member_id: str = Field(...)


# class IndexResponse(BaseModel):
#     indexed: int
#     total: int


from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


# ------------- OPS -------------


class OpsStatus(BaseModel):
    service: str = Field(default="svc-face-recon")
    live: bool = True
    ready: bool = True
    version: str = "1.0.0"


# ------------- FACE / INDEX -------------


class IndexProgress(BaseModel):
    """Detalhes do progresso de execução do /index."""

    started_at: int = Field(..., description="epoch seconds em que a indexação iniciou")
    duration_sec: int = Field(..., description="duração total em segundos")


class IndexResult(BaseModel):
    total: int
    indexed: int
    skipped_no_photo: int
    skipped_no_face: int
    errors: int
    cache_reloaded: bool
    # >>> Aqui estava como int; agora aceitamos o objeto:
    progress: Optional[IndexProgress] = None


class IndexResponse(BaseModel):
    ok: bool
    result: Optional[IndexResult] = None
    error: Optional[str] = None


# ------------- OUTROS ENDPOINTS (mantidos para compatibilidade) -------------


class EnrollRequest(BaseModel):
    member_id: str
    image_url: Optional[str] = None
    storage_path: Optional[str] = None


class EnrollResponse(BaseModel):
    ok: bool
    error: Optional[str] = None


class IdentityRequest(BaseModel):
    image_url: Optional[str] = None
    storage_path: Optional[str] = None
    max_candidates: int = 1


class Candidate(BaseModel):
    member_id: str
    distance: float


class IdentityResponse(BaseModel):
    ok: bool
    top: Optional[Candidate] = None
    candidates: List[Candidate] = []
    error: Optional[str] = None


class VerifyRequest(BaseModel):
    member_id: str
    image_url: Optional[str] = None
    storage_path: Optional[str] = None


class VerifyResponse(BaseModel):
    ok: bool
    match: bool
    distance: float
    error: Optional[str] = None


class CompareRequest(BaseModel):
    image_a_url: Optional[str] = None
    image_a_storage_path: Optional[str] = None
    image_b_url: Optional[str] = None
    image_b_storage_path: Optional[str] = None


class CompareResponse(BaseModel):
    ok: bool
    distance: float
    error: Optional[str] = None
