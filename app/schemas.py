# app/schemas.py
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class IdentifyRequest(BaseModel):
    image_url: Optional[str] = None
    supabase_path: Optional[str] = None


class IdentifyResponse(BaseModel):
    member_id: Optional[str]
    distance: Optional[float]
    matched: bool


class VerifyRequest(IdentifyRequest):
    member_id: str = Field(...)


class IndexResponse(BaseModel):
    indexed: int
    total: int
