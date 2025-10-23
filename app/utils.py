# app/utils.py
from __future__ import annotations
import io
import base64
from typing import Optional
from PIL import Image
import httpx

from .config import settings
from .supabase_client import get_supabase


def load_image_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


async def fetch_bytes_from_url(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


def fetch_bytes_from_supabase_path(path: str) -> bytes:
    """
    Baixa arquivo do Supabase Storage via SDK (com service role key).
    """
    sb = get_supabase()
    res = sb.storage.from_(settings.SUPABASE_STORAGE_BUCKET).download(path)
    # SDK v2 retorna bytes
    if isinstance(res, bytes):
        return res
    return bytes(res)


async def resolve_image_source(
    image_url: Optional[str],
    supabase_path: Optional[str],
    file_bytes: Optional[bytes],
) -> bytes:
    if file_bytes:
        return file_bytes
    if image_url:
        return await fetch_bytes_from_url(image_url)
    if supabase_path:
        return fetch_bytes_from_supabase_path(supabase_path)
    raise ValueError(
        "Nenhuma fonte de imagem fornecida (file|image_url|supabase_path)."
    )


def image_to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
