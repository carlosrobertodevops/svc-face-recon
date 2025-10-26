Perfeito, Carlos. 👌
Vamos oficializar o nome **`svc-face-recon`** e estruturar o projeto completo nesse padrão — pronto para rodar como um **microserviço Dockerizado**, integrado ao **Supabase** e ao **FlutterFlow 5**.

---

## 🧩 Nome oficial do projeto

> **svc-face-recon**
> *Face Recognition & Image Processing Microservice*

Função: detectar rostos, gerar embeddings, comparar faces e integrar com Supabase (Storage + Database) e o front-end FlutterFlow.

---

## 📁 Estrutura final de diretórios

```
svc-face-recon/
├─ app/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ docs.py                 # Swagger UI custom (/docs), /openapi.json e /openapi.yaml
│  ├─ face_engine.py
│  ├─ indexer.py
│  ├─ main.py                 # Endpoints: /live /health /ready /ops/status /index /enroll /identify /verify /compare
│  ├─ repository.py
│  ├─ schemas.py
│  ├─ supabase_client.py
│  ├─ utils.py
│  └─ __tests__/
│     ├─ test_api.py
│     └─ test_embeddings.py
│
├─ observability/
│  ├─ prometheus/
│  │  └─ prometheus.yml       # Scrape: svc-face-recon:8000/metrics, cadvisor:8080, node-exporter:9100
│  └─ grafana/
│     └─ provisioning/
│        ├─ datasources/
│        │  └─ datasource.yml # Datasource Prometheus
│        └─ dashboards/
│           ├─ dashboards.yml
│           └─ fastapi-overview.json
│
├─ Dockerfile
├─ docker-compose.local.yaml   # svc + Prometheus + Grafana + cAdvisor + node-exporter + Datadog (opcional)
├─ docker-compose.coolify.yaml # svc para produção na Coolify (secrets pela UI)
├─ requirements.txt            # FastAPI, InsightFace/ONNX, Supabase, psycopg, Prometheus instrumentator, ddtrace
├─ .env.example
├─ .env.local                  # (dev) SUPABASE_STORAGE_BUCKET=uploads
└─ .env.coolify                # (prod – opcional; preferir secrets na Coolify)

```

---

## 🧠 Serviços previstos (todos já integrados)

| Serviço                                 | Função                                             | Rota principal                                                     |
| --------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------ |
| **svc-face-recon**                      | Core FastAPI com InsightFace                       | `/health`, `/index`, `/identify`, `/verify`, `/compare`, `/enroll` |
| **Supabase**                            | Banco + Storage + Auth                             | Repositório de membros e imagens                                   |
| **FlutterFlow**                         | Frontend (upload, comparação, visualização)        | Consome rotas HTTP via API Calls                                   |
| *(futuro opcional)* **FAISS Worker**    | Busca vetorial paralela (GPU/CPU)                  | `/search/faiss`                                                    |
| *(futuro opcional)* **svc-face-worker** | Processamento assíncrono via fila (Celery + Redis) | background embeddings                                              |

---

## 🧱 Convenções e variáveis padrão

| Variável                     | Valor padrão     | Descrição                          |
| ---------------------------- | ---------------- | ---------------------------------- |
| `SERVICE_NAME`               | `svc-face-recon` | Nome do microserviço               |
| `PORT`                       | `8000`           | Porta padrão                       |
| `SUPABASE_URL`               | —                | URL Supabase                       |
| `SUPABASE_SERVICE_ROLE_KEY`  | —                | Chave service role                 |
| `SUPABASE_STORAGE_BUCKET`    | `profiles`       | Bucket padrão                      |
| `DATABASE_URL`               | —                | Conexão Postgres (pgvector)        |
| `FACE_RECOGNITION_THRESHOLD` | `0.35`           | Limiar de similaridade             |
| `MAX_FACES_PER_IMAGE`        | `5`              | Número máximo de rostos detectados |

> 👉 `.env.example` já contém todos os placeholders configurados com esse nome de serviço.

---

## 🐳 Dockerfile (com nome oficial aplicado)

```dockerfile
# svc-face-recon/Dockerfile
FROM python:3.11-slim

ENV SERVICE_NAME=svc-face-recon
WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY .env.example ./.env

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🐙 docker-compose.yaml

```yaml
version: "3.9"

services:
  svc-face-recon:
    build: .
    image: svc-face-recon:latest
    container_name: svc-face-recon
    env_file:
      - .env
    ports:
      - "8000:8000"
    volumes:
      - models-cache:/models
    restart: unless-stopped
    environment:
      - SERVICE_NAME=svc-face-recon

volumes:
  models-cache:
```

---

## 🔗 Integração com FlutterFlow 5

O FlutterFlow vai se conectar ao endpoint:

```
https://api.mondaha.com/face/   ← proxy público (exemplo)
     ├── POST /identify
     ├── POST /verify
     ├── POST /enroll
     ├── POST /compare
     └── POST /index
```

👉 Todos os endpoints seguem formato **REST/JSON** e aceitam:

* `supabase_path` (recomendado)
* `image_url`
* `file` (upload direto)

---

## 🧩 Sugestão de naming para APIs (no FlutterFlow)

| Endpoint    | API Call Name  | Tipo           |
| ----------- | -------------- | -------------- |
| `/identify` | `IdentifyFace` | POST JSON      |
| `/verify`   | `VerifyFace`   | POST JSON      |
| `/enroll`   | `EnrollFace`   | POST multipart |
| `/compare`  | `CompareFaces` | POST multipart |
| `/index`    | `RebuildIndex` | POST JSON      |

---

## 🧠 Próximos passos sugeridos

1. [ ] Criar repositório privado no GitHub: `mondaha/svc-face-recon`
2. [ ] Commitar a base que já construímos (eu posso gerar o README inicial completo pra você).
3. [ ] Adicionar **Supabase service key** ao `.env`.
4. [ ] Fazer `docker compose up -d --build` local.
5. [ ] Testar endpoints `/health` e `/identify`.
6. [ ] Configurar **API Calls** no FlutterFlow.

---

svc-face-recon — microserviço de reconhecimento facial (FastAPI + InsightFace + pgvector)


Exemplos de uso (curl)

1) Enroll (via URL)
```bash
curl -X POST http://localhost:8000/enroll \
  -H "Content-Type: application/json" \
  -d '{
    "member_id": "123",
    "image_url": "https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing"
  }'
```
2) Enroll (via Supabase Storage)
```bash
curl -X POST http://localhost:8000/enroll \
  -H "Content-Type: application/json" \
  -d '{
    "member_id": "123",
    "supabase_path": "membros/123.jpg"
  }'
```
3) Identify (top-3)
```bash
curl -X POST http://localhost:8000/identify \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://meu-site/img.jpg",
    "top_k": 3
  }'
```
```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "member_id": "123",
    "image_url": "https://meu-site/img.jpg"
  }'
```
5) Compare
```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "a_image_url": "https://meu-site/a.jpg",
    "b_image_url": "https://meu-site/b.jpg"
  }'
```
6) Enroll (arquivo local)
```bash
curl -X POST http://localhost:8000/enroll/file \
  -F "member_id=123" \
  -F "image=@/caminho/para/foto.jpg"

```
⸻

Créditos
	•	InsightFace pela qualidade dos modelos.
	•	pgvector pela simplicidade no armazenamento/vetorial.
	•	Supabase pelo Storage + PostgREST.
	•	FastAPI pela DX excelente.
