"""Camada de acesso a imagens no MinIO/S3 (substitui o Supabase Storage).

As chaves em ``fotos_path`` chegam no formato ``<categoria>/<arquivo>`` (ex.:
``membros/foto.jpg``), por vezes com barra inicial ou prefixo do bucket
(``uploads/``). Este modulo normaliza a chave e expoe download server-side
(endpoint interno) e URLs presigned acessiveis pelo navegador (endpoint publico).
"""

from __future__ import annotations

import logging
from typing import Optional

import boto3
import botocore.config

from .config import settings

logger = logging.getLogger("storage")

_internal_client_singleton: Optional[object] = None
_public_client_singleton: Optional[object] = None


def _s3_config() -> botocore.config.Config:
    # path addressing e' obrigatorio para MinIO (nao usa virtual-hosted-style).
    return botocore.config.Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
    )


def _client():
    """Client boto3 para o endpoint interno (download server-side)."""
    global _internal_client_singleton
    if _internal_client_singleton is None:
        _internal_client_singleton = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION,
            config=_s3_config(),
        )
    return _internal_client_singleton


def _public_client():
    """Client boto3 para o endpoint publico (presign alcancavel pelo navegador)."""
    global _public_client_singleton
    if _public_client_singleton is None:
        _public_client_singleton = boto3.client(
            "s3",
            endpoint_url=settings.S3_PUBLIC_ENDPOINT,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION,
            config=_s3_config(),
        )
    return _public_client_singleton


def normalize_key(key: str) -> str:
    """Remove barra inicial e o prefixo do bucket ('uploads/') se presente.

    Retorna a key relativa ao bucket.
    """
    k = key.lstrip("/")
    pref = settings.S3_BUCKET + "/"
    if k.startswith(pref):
        k = k[len(pref):]
    return k


def fetch_bytes(key: str) -> bytes:
    """Baixa o objeto do MinIO (bucket ``S3_BUCKET``) e retorna os bytes.

    Usa o client interno (``S3_ENDPOINT``). Levanta excecao em erro (o chamador trata).
    """
    normalized = normalize_key(key)
    try:
        obj = _client().get_object(Bucket=settings.S3_BUCKET, Key=normalized)
        return obj["Body"].read()
    except Exception:
        logger.exception("Falha ao baixar objeto key=%s", normalized)
        raise


def presigned_url(key: str, expires: int = 3600) -> str:
    """Gera URL presigned GET usando o endpoint PUBLICO (``S3_PUBLIC_ENDPOINT``).

    A URL retornada e' acessivel pelo navegador.
    """
    normalized = normalize_key(key)
    try:
        return _public_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.S3_BUCKET, "Key": normalized},
            ExpiresIn=expires,
        )
    except Exception:
        logger.exception("Falha ao gerar presigned url key=%s", normalized)
        raise
