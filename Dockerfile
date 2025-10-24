# syntax=docker/dockerfile:1.6
# svc-face-recon/Dockerfile (multi-arch)

ARG TARGETPLATFORM
FROM --platform=${TARGETPLATFORM} python:3.11-slim

ENV SERVICE_NAME=svc-face-recon
# cache de modelos no volume (também usado pelo insightface/onnxruntime)
ENV INSIGHTFACE_HOME=/models
ENV HF_HOME=/models
# força CPU (sem CUDA) em qualquer host
ENV ORT_DISABLE_CUDA=1

WORKDIR /app

# Dependências nativas (funciona em amd64/arm64)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libstdc++6 libgomp1 build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Garante diretório de cache e symlink para ~/.insightface
RUN mkdir -p /models && \
    if [ ! -e /root/.insightface ]; then ln -s /models /root/.insightface; fi

# Requisitos Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-download dos modelos do insightface (sem CUDA)
RUN python - <<'PY'
import os
os.environ["INSIGHTFACE_HOME"]="/models"
os.environ["ORT_DISABLE_CUDA"]="1"
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection","recognition"])
app.prepare(ctx_id=-1, det_size=(640,640))
print("Modelos baixados para:", os.environ["INSIGHTFACE_HOME"])
PY

# Código da app
COPY app ./app
# opcional (apenas dev): .env.example para facilitar rodar local
COPY .env.example ./.env

EXPOSE 8000
ENV UVICORN_WORKERS=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
