import base64
import hashlib
import numpy as np
import redis
import json
from typing import Optional

from .config import settings

_redis = redis.Redis(
    host="redis",
    port=6379,
    decode_responses=False,  # armazena bytes puros
    socket_timeout=2,
    health_check_interval=30,
)


def _make_key(prefix: str, identifier: str) -> str:
    return f"{prefix}:{identifier}"


def get_embedding_for_photo(path_or_url: str) -> Optional[np.ndarray]:
    key = _make_key("photo_emb", hashlib.sha1(path_or_url.encode()).hexdigest())
    val = _redis.get(key)
    if val is None:
        return None
    try:
        arr = np.frombuffer(base64.b64decode(val), dtype=np.float32)
        return arr
    except Exception:
        return None


def set_embedding_for_photo(path_or_url: str, emb: np.ndarray):
    key = _make_key("photo_emb", hashlib.sha1(path_or_url.encode()).hexdigest())
    b64 = base64.b64encode(emb.astype(np.float32).tobytes())
    _redis.setex(key, 60 * 60 * 24 * 7, b64)  # 7 dias


def get_member_embedding(member_id: str) -> Optional[np.ndarray]:
    key = _make_key("member_emb", str(member_id))
    val = _redis.get(key)
    if val is None:
        return None
    try:
        arr = np.frombuffer(base64.b64decode(val), dtype=np.float32)
        return arr
    except Exception:
        return None


def set_member_embedding(member_id: str, emb: np.ndarray):
    key = _make_key("member_emb", str(member_id))
    b64 = base64.b64encode(emb.astype(np.float32).tobytes())
    _redis.setex(key, 60 * 60 * 24 * 7, b64)
