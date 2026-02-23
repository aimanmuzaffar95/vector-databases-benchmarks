# Vector DB Embedding Ingest + Recall Benchmarks

This repo ingests `train.csv`, creates embeddings with `intfloat/e5-large-v2` (1024 dims), stores vectors in:
- PostgreSQL + pgvector
- Qdrant
- Weaviate

It also benchmarks Recall@K and latency for each backend.

## Repository Layout

- `pgvector/insert-data-pgvector.py`
- `pgvector/benchmark_pgvector.py`
- `qdrant/insert-data-qdrant.py`
- `qdrant/benchmark_qdrant.py`
- `weaviate/insert-data-weaviate.py`
- `weaviate/benchmark_weaviate.py`
- `docker-compose-pgvector.yml`
- `docker-compose-qdrant.yml`
- `docker-compose-weviate.yml`
- `train.csv`

## Requirements

- Python 3.9+
- Docker + Docker Compose

Install Python deps:

```bash
pip install pandas sqlalchemy psycopg2-binary sentence-transformers numpy qdrant-client weaviate-client
```

## Start Databases

pgvector:

```bash
docker compose -f docker-compose-pgvector.yml up -d
```

Qdrant:

```bash
docker compose -f docker-compose-qdrant.yml up -d
```

Weaviate:

```bash
docker compose -f docker-compose-weviate.yml up -d
```

## Ingest Data

pgvector:

```bash
python3 pgvector/insert-data-pgvector.py
```

qdrant:

```bash
python3 qdrant/insert-data-qdrant.py
```

weaviate (`--distance` added):

```bash
python3 weaviate/insert-data-weaviate.py --distance cosine
python3 weaviate/insert-data-weaviate.py --distance dot
python3 weaviate/insert-data-weaviate.py --distance l2
```

Valid Weaviate distance values:
- `cosine`
- `dot`
- `l2`

## Run Benchmarks

pgvector:

```bash
python3 pgvector/benchmark_pgvector.py
python3 pgvector/benchmark_pgvector.py --metric cosine --k-values 1,5,10 --num-queries 500
python3 pgvector/benchmark_pgvector.py --metric ip --hnsw-ef-search 20,40,80,120
python3 pgvector/benchmark_pgvector.py --metric cosine --ivfflat-probes 1,5,10,20,50
python3 pgvector/benchmark_pgvector.py --metric cosine --where-sql "genre = :g" --where-param g=Sports
```

qdrant:

```bash
python3 qdrant/benchmark_qdrant.py
python3 qdrant/benchmark_qdrant.py --k-values 1,5,10 --num-queries 500
python3 qdrant/benchmark_qdrant.py --hnsw-ef 32,64,128,256
python3 qdrant/benchmark_qdrant.py --exact-qdrant
```

weaviate (`--distance` added):

```bash
python3 weaviate/benchmark_weaviate.py
python3 weaviate/benchmark_weaviate.py --distance cosine --k-values 1,5,10 --num-queries 500
python3 weaviate/benchmark_weaviate.py --distance dot --hnsw-ef 32,64,128
python3 weaviate/benchmark_weaviate.py --distance l2 --exact-weaviate
```

## Environment Variables

pgvector scripts:
- `DATABASE_URL` (default: `postgresql+psycopg2://postgres:postgres@localhost:5432/appdb`)
- Benchmark-only: `TABLE_NAME`, `ID_COLUMN`, `EMBEDDING_COLUMN`, `GENRE_COLUMN`

qdrant benchmark:
- `QDRANT_URL` (default: `http://localhost:6333`)
- `COLLECTION_NAME` (default: `news_articles`)

weaviate scripts:
- `WEAVIATE_HTTP_HOST` (default: `localhost`)
- `WEAVIATE_HTTP_PORT` (default: `8080`)
- `WEAVIATE_GRPC_HOST` (default: `localhost`)
- `WEAVIATE_GRPC_PORT` (default: `50051`)
- `WEAVIATE_SECURE` (default: `false`)
- `WEAVIATE_COLLECTION` (default: `NewsArticle`)
- `WEAVIATE_SOURCE_ID_PROPERTY` (benchmark default: `sourceRowId`)

shared model override:
- `EMBEDDING_MODEL` (default: `intfloat/e5-large-v2`)

## Notes

- Keep ingest and benchmark model consistent.
- For Weaviate, use the same `--distance` during ingest and benchmark for aligned recall comparisons.
- `weaviate/benchmark_weaviate.py --distance` overrides inferred collection distance; omit it to auto-detect.
