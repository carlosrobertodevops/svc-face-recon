FROM python:3.11-slim

# Dependências nativas para onnxruntime/insightface e healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libstdc++6 libgomp1 build-essential curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache de modelos em volume
ENV INSIGHTFACE_HOME=/models
ENV HF_HOME=/models

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY .env.example ./.env

EXPOSE 8000
ENV UVICORN_WORKERS=1  # aumente se precisar; 1 já é ok para CPU-bound leve

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# # svc-face-recon/Dockerfile
# FROM python:3.11-slim

# ENV SERVICE_NAME=svc-face-recon
# # cache de modelos no volume
# ENV INSIGHTFACE_HOME=/models
# ENV HF_HOME=/models

# # Dependências nativas (Debian 13/Trixie → libgl1)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#   libgl1 libglib2.0-0 libstdc++6 libgomp1 build-essential curl \
#   && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Garante diretório de cache e um symlink fallback para ~/.insightface
# RUN mkdir -p /models && \
#   if [ ! -e /root/.insightface ]; then ln -s /models /root/.insightface; fi

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # depois de instalar requirements
# RUN python - <<'PY'
# import os
# os.environ["INSIGHTFACE_HOME"]="/models"
# os.environ["ORT_DISABLE_CUDA"]="1"
# from insightface.app import FaceAnalysis
# app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection","recognition"])
# app.prepare(ctx_id=-1, det_size=(640,640))
# print("Modelos baixados para:", os.environ["INSIGHTFACE_HOME"])
# PY

# COPY app ./app
# COPY .env.example ./.env

# EXPOSE 8000
# ENV UVICORN_WORKERS=1

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
