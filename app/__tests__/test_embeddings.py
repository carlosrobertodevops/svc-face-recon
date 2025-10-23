# app/__tests__/test_embeddings.py
import numpy as np
from app.face_engine import FaceEngine


def test_cosine_distance_sanity():
    a = np.ones(512, dtype=np.float32)
    a = a / np.linalg.norm(a)
    b = a.copy()
    c = -a

    # Distância deve ser 0 entre vetores idênticos
    assert abs(FaceEngine.cosine_distance(a, b) - 0.0) < 1e-6

    # Distância deve ser ~2 entre opostos (pois 1 - (-1) = 2)
    assert abs(FaceEngine.cosine_distance(a, c) - 2.0) < 1e-4
