# DOCKER.md — svc-face-recon

Guia Docker do microserviço de reconhecimento facial.

> **Branch:** `svc-face-recon-fs`. Consumido pelo monorepo **mondaha** via DNS interno `http://svc-face-recon:8000`.
> **README legado** (menciona Supabase/pgvector) — não confie nele; use este guia + `CLAUDE.md`.

---

## 1. Composes

| Arquivo | Propósito | Porta host | Banco | Rede |
|---|---|:---:|---|---|
| `docker-compose.mondaha.yml` | **Integração mondaha** — só `svc-face-recon`, aponta para postgres/minio/redis do monorepo | 8000 | Postgres do mondaha (externo) | **externa** `mondaha_default` |
| `docker-compose.yaml` | Standalone / dev isolado — sobe DB próprio | 8001 | pgvector próprio (embutido) | bridge própria |
| `docker-compose.local.yaml` | Standalone + monitoring | 8001 | pgvector próprio | bridge + monitoring |
| `docker-compose.coolify.yaml` | Deploy Coolify | conforme plataforma | conforme plataforma | conforme plataforma |

> No mondaha o serviço também pode já vir **embutido no `docker-compose.yml` do monorepo**. Nesse caso não é preciso subir `docker-compose.mondaha.yml` separado.

---

## 2. Dockerfiles

| Arquivo | Base | Porta | Uso |
|---|---|:---:|---|
| `Dockerfile` | `python:3.11-slim` | **8000** (hardcoded no `CMD`) | Imagem principal / integração mondaha |
| `Dockerfile.local` | `python:3.11-slim` | 8000 | Dev local |
| `Dockerfile.prod` | `python:3.11-slim` | **8001** | Produção |

Características do `Dockerfile` principal:
- `EXPOSE 8000`; `curl` instalado (para healthcheck).
- `CMD` fixo: `uvicorn app.main:app --host 0.0.0.0 --port 8000` → **env `PORT` é ignorada**.
- Pré-baixa o modelo InsightFace `buffalo_l` para `/models` durante o build (CPU-only).

---

## 3. Build & run por cenário

### Integração mondaha
```bash
# a rede mondaha_default precisa já existir (subida pelo monorepo mondaha)
docker compose -f docker-compose.mondaha.yml up -d --build

# verificar prontidão
curl http://localhost:8000/ready
```

### Standalone (DB próprio, porta 8001)
```bash
docker compose -f docker-compose.yaml up -d --build
curl http://localhost:8001/ready
```

### Standalone + monitoring
```bash
docker compose -f docker-compose.local.yaml up -d --build
```

### Build manual da imagem
```bash
docker build -t svc-face-recon:latest .
docker run --rm -p 8000:8000 \
  -e DATABASE_URL="postgresql://mondaha:mondaha@host.docker.internal:5432/mondaha" \
  -e S3_ENDPOINT="http://host.docker.internal:9000" \
  svc-face-recon:latest
```

---

## 4. Integração com mondaha

- **Rede:** `docker-compose.mondaha.yml` usa a rede **externa** `mondaha_default` (criada pelo monorepo). Sem ela subida, o `up` falha.
- **DNS interno:** o BFF server-only do mondaha chama `http://svc-face-recon:8000`.
- **Dependências** (resolvidas por DNS dentro de `mondaha_default`):

| Serviço | Host:porta interno |
|---|---|
| Postgres | `postgres:5432` |
| MinIO / S3 | `minio:9000` |
| Redis | `redis:6379` |

- **Healthcheck:** `curl -f http://localhost:8000/health` (retorna `{"ok":true}` após `SELECT 1`).
- **Rotas consumidas pelo mondaha:** `POST /identify/file`, `GET /ready`, `POST /index`.

---

## 5. Fluxo operacional

```
1. Popular a tabela `membros` (membro_id, nome_completo, fotos_path)
   → provida pelo Postgres do mondaha; svc NÃO cria essa tabela.

2. Subir/garantir fotos no MinIO (bucket `uploads`, keys em fotos_path).

3. POST /index         → varre fotos, gera embeddings, persiste em `embeddings`,
                         reconstrói índice em memória (lock via Redis se configurado).

4. GET /ready          → confirma cache_embeddings > 0.

5. POST /identify/file → envia uma imagem, recebe candidatos ordenados por distância.
```

Exemplos `curl`:
```bash
# reconstruir índice
curl -X POST http://svc-face-recon:8000/index

# identificar por arquivo (multipart)
curl -X POST http://svc-face-recon:8000/identify/file \
  -F "image=@/caminho/foto.jpg" \
  -F "top_k=5"

# identificar por JSON (key MinIO)
curl -X POST http://svc-face-recon:8000/identify \
  -H "Content-Type: application/json" \
  -d '{"supabase_path":"membros/abc.jpg","top_k":3}'

# enroll de um membro
curl -X POST http://svc-face-recon:8000/enroll/file \
  -F "member_id=123" \
  -F "image=@/caminho/foto.jpg"
```

---

## 6. Variáveis de ambiente

| Variável | Default | Obrigatória | Descrição |
|---|---|:---:|---|
| `DATABASE_URL` | — | **Sim** | `postgresql://mondaha:mondaha@postgres:5432/mondaha` |
| `S3_ENDPOINT` | `http://minio:9000` | | Endpoint interno S3 |
| `S3_PUBLIC_ENDPOINT` | `http://localhost:9000` | | Endpoint p/ presigned URL (browser) |
| `S3_REGION` | `us-east-1` | | |
| `S3_BUCKET` | `uploads` | | Bucket das fotos |
| `S3_ACCESS_KEY` | `minioadmin` | | |
| `S3_SECRET_KEY` | `minioadmin` | | |
| `REDIS_URL` | — (opcional) | | `redis://redis:6379/0` — lock/cache `/index` |
| `FACE_RECOGNITION_THRESHOLD` | `0.35` | | Corte de match (`distance <= threshold`) |
| `MAX_FACES_PER_IMAGE` | `5` | | |
| `MEMBERS_TABLE` | `membros` | | Tabela lida (não criada) |
| `MEMBERS_ID_COLUMN` | `membro_id` | | |
| `MEMBERS_NAME_COLUMN` | `nome_completo` | | |
| `MEMBERS_PHOTOS_COLUMN` | `fotos_path` | | Array de keys MinIO |
| `HOST` | `0.0.0.0` | | |
| `PORT` | `8000` | | **Ignorado no CMD do `Dockerfile`** |

> No `docker-compose.mondaha.yml` as envs vêm de `env_file: .env`.

---

## 7. Troubleshooting

| Sintoma | Causa provável | Ação |
|---|---|---|
| Serviço responde na 8000, esperava 8001 (ou vice-versa) | `Dockerfile` fixa 8000; `Dockerfile.prod` usa 8001; `PORT` é ignorada | Ajustar o mapeamento `ports:` no compose ou usar o Dockerfile correto |
| `/identify` retorna candidatos sem `name` / vazio | Tabela `membros` ausente ou vazia | Popular `membros` no Postgres do mondaha (o svc não a cria) |
| `up` falha por rede | Rede externa `mondaha_default` não existe | Subir o monorepo mondaha primeiro (`docker compose up` na raiz) |
| `/health` retorna erro / 500 | `DATABASE_URL` inválida ou Postgres indisponível | Conferir `DATABASE_URL` e conectividade `postgres:5432` |
| `photo_url` não abre no browser | `S3_PUBLIC_ENDPOINT` incorreto | Ajustar para o endpoint MinIO acessível pelo cliente |
| Fotos não encontradas / erro S3 | Bucket errado ou keys inexistentes | Conferir `S3_BUCKET=uploads` e keys em `fotos_path` |
| Build lento / falha ao baixar modelo | Download do `buffalo_l` no build (CPU) | Aguardar; garantir acesso de rede no build; cache de camadas ajuda |
| `cache_embeddings: 0` em `/ready` | Índice ainda não construído | Rodar `POST /index` |
