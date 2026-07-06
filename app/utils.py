# app/utils.py
from __future__ import annotations
import io
import base64
import re
from typing import Optional, Iterable
from PIL import Image
import httpx

from .config import settings
from .storage import fetch_bytes as s3_fetch_bytes

PUBLIC_OBJ_RE = re.compile(r"/storage/v1/object/public/([^/]+)/(.+)$", re.IGNORECASE)


def load_image_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


async def fetch_bytes_from_url(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


def fetch_bytes_from_supabase_path(path: str) -> bytes:
    """
    Baixa arquivo do MinIO pela key.
    Ex.: path = 'membros/123.jpg' (key MinIO)
    Nome mantido por compatibilidade com chamadores existentes.
    """
    return s3_fetch_bytes(path)


def public_url_to_storage_path(url: str) -> Optional[tuple[str, str]]:
    """
    Se a string for uma URL pública do Supabase Storage, retorna (bucket, path_relativo).
    Caso contrário, retorna None.
    Ex.: https://<proj>.supabase.co/storage/v1/object/public/uploads/membros/abc.jpg
         -> ("uploads", "membros/abc.jpg")
    """
    m = PUBLIC_OBJ_RE.search(url)
    if not m:
        return None
    bucket, rel = m.group(1), m.group(2)
    return bucket, rel


async def resolve_image_source(
    image_url: Optional[str],
    supabase_path: Optional[str],
    file_bytes: Optional[bytes],
) -> bytes:
    """
    Resolve uma fonte de imagem:
      - file_bytes (multipart)
      - image_url: URL pública/assinada (http/https)
      - supabase_path: caminho relativo / key (ex.: 'membros/xyz.jpg')
      - se image_url for URL pública do Supabase, extrai a key e baixa do MinIO
    """
    if file_bytes:
        return file_bytes

    if image_url:
        # Se for URL pública do Supabase, extrai a key e baixa do MinIO
        parsed = public_url_to_storage_path(image_url)
        if parsed:
            _bucket, rel = parsed
            return s3_fetch_bytes(rel)
        # caso contrário, baixa via HTTP normal
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
