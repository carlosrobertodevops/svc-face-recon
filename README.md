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
│  ├─ main.py                   # FastAPI (endpoints)
│  ├─ config.py                 # Variáveis de ambiente
│  ├─ face_engine.py            # InsightFace + ONNX Runtime
│  ├─ supabase_client.py        # Integração Supabase
│  ├─ repository.py             # Persistência com pgvector
│  ├─ indexer.py                # Indexação, cache em memória
│  ├─ schemas.py                # Modelos Pydantic
│  ├─ utils.py                  # Download/conversão imagens
│  ├─ __tests__/                # (opcional) testes unitários
│  │  ├─ test_api.py
│  │  └─ test_embeddings.py
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.local.yaml
├─ docker-compose.coolify.yaml
├─ .env.example
├─ .env.local
├─ .env.coolify
└─ README.md
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

Posso agora gerar o **README.md completo** para o repositório `svc-face-recon`, com:

* introdução técnica
* instruções de build
* documentação de endpoints (OpenAPI + exemplos cURL e FlutterFlow)
* guia de integração FlutterFlow (passo a passo com capturas e actions)

Quer que eu gere esse `README.md` pronto agora?
