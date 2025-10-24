-- Extensão pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabela para embeddings
CREATE TABLE IF NOT EXISTS member_faces (
  member_id text PRIMARY KEY,
  embedding vector(512) NOT NULL
);

-- Índice HNSW para busca aproximada
CREATE INDEX IF NOT EXISTS member_faces_hnsw
ON member_faces USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);
