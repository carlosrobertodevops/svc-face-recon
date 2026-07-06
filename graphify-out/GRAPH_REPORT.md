# Graph Report - svc-face-recon  (2026-07-06)

## Corpus Check
- 16 files · ~12,069 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 172 nodes · 336 edges · 10 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `be298448`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]

## God Nodes (most connected - your core abstractions)
1. `build_index_from_members()` - 16 edges
2. `_fetch_image_bytes()` - 11 edges
3. `_extract_single_embedding()` - 10 edges
4. `get_conn()` - 10 edges
5. `upsert_member_embedding()` - 10 edges
6. `identity()` - 9 edges
7. `identity_file()` - 9 edges
8. `_get_bytes()` - 8 edges
9. `_member_photo_public_url()` - 8 edges
10. `fetch_bytes_from_supabase_path()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `build_index_from_members()` --calls--> `fetch_all_members()`  [EXTRACTED]
  app/indexer.py → app/repository.py
- `build_index_from_members()` --calls--> `upsert_member_embedding()`  [EXTRACTED]
  app/indexer.py → app/repository.py
- `index_all()` --calls--> `build_index_from_members()`  [EXTRACTED]
  app/main.py → app/indexer.py
- `_fetch_image_bytes()` --calls--> `fetch_bytes_from_supabase_path()`  [EXTRACTED]
  app/main.py → app/utils.py
- `_fetch_image_bytes()` --calls--> `fetch_bytes_from_url()`  [EXTRACTED]
  app/main.py → app/utils.py

## Import Cycles
- 1-file cycle: `app/docs.py -> app/docs.py`

## Communities (10 total, 0 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.11
Nodes (33): Any, _coerce_photo_value_to_public_url(), compare(), compare_files(), CompareRequest, CompareResponse, _distance(), enroll() (+25 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (36): _acquire_lock(), _avg_normalize(), build_index_from_members(), _bytes_cache_key(), _get_bytes(), _get_bytes_cached(), _get_redis(), _get_status() (+28 more)

### Community 2 - "Community 2"
Cohesion: 0.18
Nodes (18): index_all(), IndexResponse, IndexResult, CompareRequest, CompareResponse, EnrollRequest, EnrollResponse, IdentityCandidate (+10 more)

### Community 3 - "Community 3"
Cohesion: 0.18
Nodes (16): health(), _ensure_schema(), fetch_all_embeddings(), fetch_all_members(), fetch_member_name(), fetch_member_photos(), get_conn(), ndarray (+8 more)

### Community 4 - "Community 4"
Cohesion: 0.27
Nodes (10): get_embedding_for_photo(), get_member_embedding(), _make_key(), ndarray, set_embedding_for_photo(), set_member_embedding(), Config, Lê .env do serviço e IGNORA variáveis extras (dd_*, gf_* etc.)     Defaults já a (+2 more)

### Community 5 - "Community 5"
Cohesion: 0.23
Nodes (11): _client(), normalize_key(), presigned_url(), _public_client(), Camada de acesso a imagens no MinIO/S3 (substitui o Supabase Storage).  As chave, Client boto3 para o endpoint interno (download server-side)., Client boto3 para o endpoint publico (presign alcancavel pelo navegador)., Remove barra inicial e o prefixo do bucket ('uploads/') se presente.      Retorn (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.20
Nodes (5): FaceEngine, Image, ndarray, Wrapper do InsightFace para detecção + embeddings.     Usa o modelo 'buffalo_l', Retorna lista de dicts:         [{bbox, kps, det_score, embedding(np.float32[512

### Community 7 - "Community 7"
Cohesion: 0.20
Nodes (9): 🧱 Convenções e variáveis padrão, 🐙 docker-compose.yaml, 🐳 Dockerfile (com nome oficial aplicado), 📁 Estrutura final de diretórios, 🔗 Integração com FlutterFlow 5, 🧩 Nome oficial do projeto, 🧠 Próximos passos sugeridos, 🧠 Serviços previstos (todos já integrados) (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.48
Nodes (6): _collect_endpoints(), mount_docs_routes(), Retorna pares (method, path) únicos e ordenados, excluindo rotas internas de doc, swagger_ui_html(), FastAPI, HTMLResponse

## Knowledge Gaps
- **15 isolated node(s):** `Config`, `HTMLResponse`, `Image`, `ndarray`, `Any` (+10 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_index_from_members()` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `get_conn()` connect `Community 3` to `Community 0`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **What connects `Config`, `Lê .env do serviço e IGNORA variáveis extras (dd_*, gf_* etc.)     Defaults já a`, `HTMLResponse` to the rest of the system?**
  _47 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.11463414634146342 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.08658536585365853 - nodes in this community are weakly interconnected._