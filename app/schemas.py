# from __future__ import annotations
# from typing import Optional, List
# from pydantic import BaseModel, Field


# # ------------- OPS -------------


# class OpsStatus(BaseModel):
#     service: str = Field(default="svc-face-recon")
#     live: bool = True
#     ready: bool = True
#     version: str = "1.0.0"


# # ------------- FACE / INDEX -------------


# class IndexProgress(BaseModel):
#     """Detalhes do progresso de execução do /index."""

#     started_at: int = Field(..., description="epoch seconds em que a indexação iniciou")
#     duration_sec: int = Field(..., description="duração total em segundos")


# class IndexResult(BaseModel):
#     total: int
#     indexed: int
#     skipped_no_photo: int
#     skipped_no_face: int
#     errors: int
#     cache_reloaded: bool
#     # >>> Aqui estava como int; agora aceitamos o objeto:
#     progress: Optional[IndexProgress] = None


# class IndexResponse(BaseModel):
#     ok: bool
#     result: Optional[IndexResult] = None
#     error: Optional[str] = None


# # ------------- OUTROS ENDPOINTS (mantidos para compatibilidade) -------------


# class EnrollRequest(BaseModel):
#     member_id: str
#     image_url: Optional[str] = None
#     storage_path: Optional[str] = None


# class EnrollResponse(BaseModel):
#     ok: bool
#     error: Optional[str] = None


# class IdentityRequest(BaseModel):
#     image_url: Optional[str] = None
#     storage_path: Optional[str] = None
#     max_candidates: int = 1


# class Candidate(BaseModel):
#     member_id: str
#     distance: float


# class IdentityResponse(BaseModel):
#     ok: bool
#     top: Optional[Candidate] = None
#     candidates: List[Candidate] = []
#     error: Optional[str] = None


# class VerifyRequest(BaseModel):
#     member_id: str
#     image_url: Optional[str] = None
#     storage_path: Optional[str] = None


# class VerifyResponse(BaseModel):
#     ok: bool
#     match: bool
#     distance: float
#     error: Optional[str] = None


# class CompareRequest(BaseModel):
#     image_a_url: Optional[str] = None
#     image_a_storage_path: Optional[str] = None
#     image_b_url: Optional[str] = None
#     image_b_storage_path: Optional[str] = None


# class CompareResponse(BaseModel):
#     ok: bool
#     distance: float
#     error: Optional[str] = None
# app/schema.py
from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field


# =============================================================================
# OPS
# =============================================================================


class OpsStatus(BaseModel):
    service: str = Field(default="svc-face-recon")
    live: bool = True
    ready: bool = True
    version: str = "1.0.0"


# =============================================================================
# INDEX (/index)
# =============================================================================


class IndexProgress(BaseModel):
    """
    Detalhes do progresso da execução do /index.
    Deixe como opcional para não quebrar clientes quando o serviço não enviar.
    """

    started_at: Optional[int] = Field(
        None, description="Epoch (segundos) em que a indexação iniciou"
    )
    duration_sec: Optional[int] = Field(None, description="Duração total em segundos")
    percent: Optional[float] = Field(
        None, description="Progresso estimado 0–100 (quando aplicável)"
    )
    state: Optional[str] = Field(None, description="Ex.: running, done, error")


class IndexResult(BaseModel):
    total: int
    indexed: int
    skipped_no_photo: int
    skipped_no_face: int
    errors: int
    cache_reloaded: bool
    progress: Optional[IndexProgress] = Field(
        None, description="Informações de progresso (se disponíveis)"
    )


class IndexResponse(BaseModel):
    ok: bool
    result: Optional[IndexResult] = None
    error: Optional[str] = None


# =============================================================================
# FACE – Requisições genéricas (JSON)
# =============================================================================


class EnrollRequest(BaseModel):
    """
    Cadastra/atualiza o embedding de um membro a partir de UMA imagem.
    Pelo menos um dos campos de imagem deve ser enviado.
    """

    member_id: str
    supabase_path: Optional[str] = Field(
        None,
        description="Caminho relativo no bucket configurado (ex.: 'membros/123.jpg')",
    )
    image_url: Optional[str] = Field(
        None,
        description="URL pública (Supabase/HTTP/HTTPS; Drive 'view' também é aceito)",
    )
    image_b64: Optional[str] = Field(
        None,
        description="Imagem em Base64 ('data:image/...;base64,...' ou só o payload)",
    )

    # Compatibilidade antiga:
    storage_path: Optional[str] = Field(
        None,
        description="DEPRECATED: use 'supabase_path'",
    )


class EnrollResponse(BaseModel):
    ok: bool
    member_id: Optional[str] = None
    error: Optional[str] = None


class IdentityRequest(BaseModel):
    """
    Identificação top-K contra o cache em memória.
    """

    supabase_path: Optional[str] = None
    image_url: Optional[str] = None
    image_b64: Optional[str] = None
    top_k: int = Field(
        1, ge=1, le=10, description="Quantidade de candidatos retornados"
    )

    # Compatibilidade antiga:
    storage_path: Optional[str] = Field(
        None, description="DEPRECATED: use 'supabase_path'"
    )
    max_candidates: Optional[int] = Field(None, description="DEPRECATED: use 'top_k'")


class IdentityCandidate(BaseModel):
    member_id: str
    distance: float = Field(..., description="Métrica 1 - dot (menor é melhor)")
    matched: bool = Field(..., description="Se passou do threshold configurado")
    name: Optional[str] = Field(
        None, description="Nome do membro (enriquecido do Supabase)"
    )
    photo_path: Optional[str] = Field(
        None,
        description="Caminho no Storage da foto usada para o embedding (quando disponível)",
    )


class IdentityResponse(BaseModel):
    ok: bool
    threshold: float
    candidates: List[IdentityCandidate] = []


class VerifyRequest(BaseModel):
    """
    Verifica se a imagem pertence ao 'member_id' informado.
    """

    member_id: str
    supabase_path: Optional[str] = None
    image_url: Optional[str] = None
    image_b64: Optional[str] = None

    # Compat:
    storage_path: Optional[str] = Field(
        None, description="DEPRECATED: use 'supabase_path'"
    )


class VerifyResponse(BaseModel):
    ok: bool
    member_id: str
    name: Optional[str] = None
    distance: float
    threshold: float
    matched: bool
    photo_path: Optional[str] = Field(
        None, description="Caminho no Storage da foto-base do membro (se disponível)"
    )


class CompareRequest(BaseModel):
    """
    Compara duas imagens (A x B).
    """

    a_supabase_path: Optional[str] = None
    a_image_url: Optional[str] = None
    a_image_b64: Optional[str] = None
    b_supabase_path: Optional[str] = None
    b_image_url: Optional[str] = None
    b_image_b64: Optional[str] = None

    # Compat:
    image_a_storage_path: Optional[str] = Field(
        None, description="DEPRECATED: use 'a_supabase_path'"
    )
    image_a_url: Optional[str] = Field(
        None, description="DEPRECATED: use 'a_image_url'"
    )
    image_b_storage_path: Optional[str] = Field(
        None, description="DEPRECATED: use 'b_supabase_path'"
    )
    image_b_url: Optional[str] = Field(
        None, description="DEPRECATED: use 'b_image_url'"
    )


class CompareResponse(BaseModel):
    ok: bool
    distance: float
    threshold: float
    is_same: bool
