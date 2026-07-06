# CLAUDE.md — svc-face-recon

Instruções operacionais para agentes trabalhando neste repositório.

> **Branch canônica:** `svc-face-recon-fs` — todos os fatos abaixo refletem esta branch.
> **README legado:** o `README.md` atual está em tom rascunho e menciona Supabase/pgvector/FlutterFlow. **NÃO confie no README.** Use este CLAUDE.md como fonte de verdade.

---

## 1. Propósito

Microserviço de **reconhecimento facial** consumido pelo monorepo **mondaha**.
Gera e indexa embeddings faciais das fotos dos membros e responde consultas de identificação/verificação/comparação de faces via HTTP.

- **Enroll**: registra o embedding de um membro.
- **Index**: varre todas as fotos de todos os membros e (re)constrói o índice em memória + persiste embeddings.
- **Identify**: dada uma imagem, retorna os membros mais parecidos.
- **Verify**: confirma se uma imagem corresponde a um membro específico.
- **Compare**: compara duas imagens diretamente.

---

## 2. Stack

| Camada | Tecnologia |
|---|---|
| Runtime | Python 3.11 |
| Web | FastAPI + uvicorn |
| Motor facial | InsightFace `buffalo_l` (ONNX, **CPU**) |
| Banco | PostgreSQL — embeddings `BYTEA` + tabela `membros` (leitura) |
| Storage | MinIO / S3 (fotos, via `boto3`) |
| Cache/Lock | Redis (**opcional** — lock/cache do `/index`) |
| Supabase | **DEPRECATED** — não usar |

---

## 3. Layout `app/`

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Todos os endpoints (FastAPI app) |
| `config.py` | `Settings` (pydantic, `extra=ignore`) — env vars |
| `repository.py` | Acesso Postgres; `_ensure_schema()` cria tabela `embeddings` |
| `storage.py` | MinIO/S3 (download foto, presigned URL) |
| `indexer.py` | Índice em memória; `build_index_from_members`; lock Redis |
| `face_engine.py` | InsightFace (detecção + embedding) |
| `utils.py` | Helpers |
| `docs.py` | Swagger custom (`/docs`, `/openapi.json`, `/openapi.yaml`) |

---

## 4. Endpoints

### Operacionais

| Método | Rota | Retorno |
|---|---|---|
| GET | `/live` | `{"status":"live"}` |
| GET | `/health` | `SELECT 1` no Postgres → `{"ok":true}` |
| GET | `/ready` | `{"ready":true,"cache_embeddings":n}` |
| GET | `/ops/status` | Status operacional detalhado |
| GET | `/metrics` | Métricas Prometheus |

### Face — JSON

| Método | Rota | Body | Retorno |
|---|---|---|---|
| POST | `/index` | *(sem body)* | `IndexResponse` — gera embeddings de todas as fotos dos membros |
| POST | `/enroll` | `{member_id, supabase_path\|image_url\|image_b64}` | resultado do enroll |
| POST | `/identify` | `{supabase_path\|image_url\|image_b64, top_k:1..10}` | `IdentityResponse` |
| POST | `/verify` | `{member_id, supabase_path\|image_url\|image_b64}` | match do membro |
| POST | `/compare` | `{a_*, b_*}` | distância entre duas imagens |

### Face — multipart (`multipart/form-data`)

| Método | Rota | Campos | Retorno |
|---|---|---|---|
| POST | `/enroll/file` | `member_id` (Form), `image` (File) | resultado do enroll |
| POST | `/identify/file` | `image` (File), `top_k` (Form) | `IdentityResponse` |
| POST | `/verify/file` | `member_id` (Form), `image` (File) | match do membro |
| POST | `/compare/files` | dois arquivos | distância |

**`IdentityResponse`:**
```json
{
  "ok": true,
  "threshold": 0.35,
  "candidates": [
    { "member_id": "...", "distance": 0.21, "matched": true, "name": "...", "photo_url": "<presigned MinIO>" }
  ]
}
```

> `supabase_path` no body é apenas o **nome do campo** (legado). Aponta para uma **key do MinIO/S3**, não para Supabase.

---

## 5. Semântica de match

- `distance = 1 - dot(embedding_norm)` — embeddings são normalizados.
- **Menor distância = mais parecido.**
- `matched = distance <= FACE_RECOGNITION_THRESHOLD` (default `0.35`).
- `photo_url` = presigned GET do MinIO (expira; não expõe credenciais).
- `CORS` = `*`.

---

## 6. Variáveis de ambiente

`config.py` → `Settings` (pydantic, `extra=ignore`).

| Variável | Default | Obrigatória | Descrição |
|---|---|:---:|---|
| `DATABASE_URL` | — | **Sim** | Ex.: `postgresql://mondaha:mondaha@postgres:5432/mondaha` |
| `S3_ENDPOINT` | `http://minio:9000` | | Endpoint interno S3 |
| `S3_PUBLIC_ENDPOINT` | `http://localhost:9000` | | Endpoint p/ presigned URL (browser) |
| `S3_REGION` | `us-east-1` | | |
| `S3_BUCKET` | `uploads` | | Bucket das fotos |
| `S3_ACCESS_KEY` | `minioadmin` | | |
| `S3_SECRET_KEY` | `minioadmin` | | |
| `REDIS_URL` | — (opcional) | | Ex.: `redis://redis:6379/0` — lock/cache `/index` |
| `FACE_RECOGNITION_THRESHOLD` | `0.35` | | Corte de match |
| `MAX_FACES_PER_IMAGE` | `5` | | |
| `MEMBERS_TABLE` | `membros` | | Tabela lida (não criada) |
| `MEMBERS_ID_COLUMN` | `membro_id` | | |
| `MEMBERS_NAME_COLUMN` | `nome_completo` | | |
| `MEMBERS_PHOTOS_COLUMN` | `fotos_path` | | jsonb/array de keys MinIO |
| `HOST` | `0.0.0.0` | | |
| `PORT` | `8000` | | **Ignorado no CMD do Dockerfile** (ver gotchas) |

---

## 7. Schema do banco

- **`embeddings(member_id TEXT PK, embedding BYTEA)`** — auto-criada por `repository._ensure_schema()`.
- **`membros(membro_id, nome_completo, fotos_path)`** — **LIDA, mas deve pré-existir.** O serviço **não** cria essa tabela. No cenário mondaha, ela é provida pelo Postgres do monorepo.

---

## 8. Qual compose usar

| Cenário | Compose | Observação |
|---|---|---|
| **Integração mondaha** | `docker-compose.mondaha.yml` | Só `svc-face-recon`; build `Dockerfile` (porta 8000); rede externa `mondaha_default`; aponta postgres/minio/redis do mondaha. **OU** o serviço já embutido no `docker-compose.yml` do monorepo mondaha. |
| **Standalone / dev isolado** | `docker-compose.yaml` | Sobe pgvector próprio; porta **8001**. |
| **Standalone + monitoring** | `docker-compose.local.yaml` | Bridge de monitoramento. |
| **Deploy Coolify** | `docker-compose.coolify.yaml` | |

> **Regra de consumo:** dentro da rede mondaha, o monorepo chama o serviço por **DNS interno** `http://svc-face-recon:8000` (BFF server-only, **nunca exposto ao browser**). Rotas efetivamente usadas: `POST /identify/file`, `GET /ready`, `POST /index`.

### Configuração do lado mondaha

O adapter/BFF do mondaha lê estas envs (definidas no serviço `app` do `docker-compose.yml` raiz do monorepo):

| Env (mondaha) | Valor | Observação |
|---|---|---|
| `FACE_SERVICE_URL` | `http://svc-face-recon:8000` | DNS interno docker. **Era** `https://svc-face-recon.mondaha.com` (público) — migrado para consumo interno. |
| `FACE_SERVICE_THRESHOLD` | ex.: `0.35` | Threshold usado pelo cliente. |
| `FACE_SERVICE_TIMEOUT_MS` | ex.: `15000` | Timeout do fetch. **Agora configurável** (antes hardcoded 15s no adapter). |

- **Healthcheck no compose do mondaha:** o serviço `svc-face-recon` ganhou `healthcheck` (`curl -f http://localhost:8000/health`) e o serviço `app` passou a `depends_on: { svc-face-recon: { condition: service_healthy } }`. O `/health` faz `SELECT 1` no Postgres antes de reportar saudável.
- **Build no compose do mondaha:** usa `build` apontando para `../svc-face-recon/Dockerfile` (porta **8000**).

Detalhes completos de Docker em **[DOCKER.md](./DOCKER.md)**.

---

## 9. Gotchas

- **Porta hardcoded no Dockerfile principal:** o `CMD` é `uvicorn app.main:app --host 0.0.0.0 --port 8000`. A env `PORT` é **ignorada** — o container sempre escuta **8000**. `Dockerfile.prod` usa **8001**.
- **Tabela `membros` deve existir** antes de `POST /index` / `/identify` retornarem nomes; o serviço só cria `embeddings`.
- **Supabase está DEPRECATED** — ignore qualquer código/doc que aponte para Supabase.
- **README legado** — não usar como referência.
- **`buffalo_l` é pré-baixado** para `/models` no build da imagem (CPU-only). O primeiro build é mais lento por causa disso.
