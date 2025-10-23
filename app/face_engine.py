# app/face_engine.py
from __future__ import annotations
import os

# --- Força cache do InsightFace no volume /models e desativa CUDA no ORT ---
os.environ.setdefault("INSIGHTFACE_HOME", "/models")
os.environ.setdefault("HF_HOME", "/models")
os.environ.setdefault("ORT_DISABLE_CUDA", "1")  # evita warnings de CUDA ausente

import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis


class FaceEngine:
    """
    Wrapper do InsightFace para detecção + embeddings.
    Usa o modelo 'buffalo_l' (512-dim) e CPU por padrão.
    """

    def __init__(self, det_size=(640, 640), cpu_only: bool = True):
        self.app = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection", "recognition"]
        )
        ctx_id = -1 if cpu_only else 0  # -1 = CPU
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def extract_embeddings(self, image: Image.Image, max_faces: int = 5):
        """
        Retorna lista de dicts:
        [{bbox, kps, det_score, embedding(np.float32[512])}]
        """
        np_img = np.array(image)
        faces = self.app.get(np_img)
        faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)[
            :max_faces
        ]
        out = []
        for f in faces:
            out.append(
                {
                    "bbox": f.bbox.tolist(),
                    "kps": f.kps.tolist(),
                    "det_score": float(f.det_score),
                    "embedding": f.normed_embedding.astype(np.float32),
                }
            )
        return out

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        # embeddings buffalo_l são normalizados; distância de cosseno = 1 - dot
        return float(1.0 - float(np.dot(a, b)))


face_engine = FaceEngine()
