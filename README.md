# Vector DB Embedding Ingest + Recall Benchmarks

This project ingests `train.csv`, generates embeddings with `intfloat/e5-large-v2` (1024 dims), stores vectors in multiple backends, and benchmarks Recall@K + latency.

Backends:
- PostgreSQL + pgvector
- Qdrant
- Weaviate
- Milvus
- FAISS

## Repository Layout

- `pgvector/insert-data-pgvector.py`
- `pgvector/benchmark_pgvector.py`
- `qdrant/insert-data-qdrant.py`
- `qdrant/benchmark_qdrant.py`
- `weaviate/insert-data-weaviate.py`
- `weaviate/benchmark_weaviate.py`
- `milvus/insert-data-milvus.py`
- `milvus/benchmark_milvus.py`
- `faiss/insert-data-faiss.py`
- `faiss/benchmark_faiss.py`
- `docker-compose-pgvector.yml`
- `docker-compose-qdrant.yml`
- `docker-compose-weviate.yml`
- `docker-compose-milvus.yml`
- `train.csv`

## Requirements

- Python 3.9+
- Docker + Docker Compose

Install dependencies:

```bash
pip install pandas sqlalchemy psycopg2-binary sentence-transformers numpy qdrant-client weaviate-client pymilvus faiss-cpu
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

Milvus:

```bash
docker compose -f docker-compose-milvus.yml up -d
```

## Ingest Data

pgvector:

```bash
python3 pgvector/insert-data-pgvector.py
```

Qdrant:

```bash
python3 qdrant/insert-data-qdrant.py
```

Weaviate:

```bash
python3 weaviate/insert-data-weaviate.py --distance cosine
python3 weaviate/insert-data-weaviate.py --distance dot
python3 weaviate/insert-data-weaviate.py --distance l2
```

Milvus:

```bash
python3 milvus/insert-data-milvus.py --distance cosine
python3 milvus/insert-data-milvus.py --distance dot
python3 milvus/insert-data-milvus.py --distance l2
```

FAISS:

```bash
python3 faiss/insert-data-faiss.py --distance cosine --index-type hnsw --overwrite
python3 faiss/insert-data-faiss.py --distance dot --index-type ivfflat --ivf-nlist 1024 --overwrite
python3 faiss/insert-data-faiss.py --distance l2 --index-type flat --overwrite
```

Milvus distance aliases:
- `dot` is mapped to Milvus `IP`
- `euclid` / `euclidean` are mapped to `L2`

## Run Benchmarks

pgvector:

```bash
python3 pgvector/benchmark_pgvector.py
python3 pgvector/benchmark_pgvector.py --metric cosine --k-values 1,5,10 --num-queries 500
python3 pgvector/benchmark_pgvector.py --metric ip --hnsw-ef-search 20,40,80,120
python3 pgvector/benchmark_pgvector.py --metric cosine --ivfflat-probes 1,5,10,20,50
python3 pgvector/benchmark_pgvector.py --metric cosine --where-sql "genre = :g" --where-param g=Sports
```

Qdrant:

```bash
python3 qdrant/benchmark_qdrant.py
python3 qdrant/benchmark_qdrant.py --k-values 1,5,10 --num-queries 500
python3 qdrant/benchmark_qdrant.py --hnsw-ef 32,64,128,256
python3 qdrant/benchmark_qdrant.py --exact-qdrant
```

Weaviate:

```bash
python3 weaviate/benchmark_weaviate.py
python3 weaviate/benchmark_weaviate.py --distance cosine --k-values 1,5,10 --num-queries 500
python3 weaviate/benchmark_weaviate.py --distance dot --hnsw-ef 32,64,128
python3 weaviate/benchmark_weaviate.py --distance l2 --exact-weaviate
```

Milvus:

```bash
python3 milvus/benchmark_milvus.py
python3 milvus/benchmark_milvus.py --distance cosine --consistency-level Bounded
python3 milvus/benchmark_milvus.py --distance cosine --hnsw-ef 32,64,128
python3 milvus/benchmark_milvus.py --distance cosine --nprobe 8,16,32
python3 milvus/benchmark_milvus.py --distance cosine --k-values 1,5,10 --num-queries 200
```

FAISS:

```bash
python3 faiss/benchmark_faiss.py
python3 faiss/benchmark_faiss.py --k-values 1,5,10 --num-queries 500
python3 faiss/benchmark_faiss.py --nprobe 4,8,16,32
python3 faiss/benchmark_faiss.py --hnsw-ef-search 32,64,128,256
```

## Environment Variables

Shared:
- `EMBEDDING_MODEL` (default: `intfloat/e5-large-v2`)

pgvector:
- `DATABASE_URL` (default: `postgresql+psycopg2://postgres:postgres@localhost:5432/appdb`)
- benchmark only: `TABLE_NAME`, `ID_COLUMN`, `EMBEDDING_COLUMN`, `GENRE_COLUMN`

Qdrant:
- `QDRANT_URL` (default: `http://localhost:6333`)
- `COLLECTION_NAME` (default: `news_articles`)

Weaviate:
- `WEAVIATE_HTTP_HOST` (default: `localhost`)
- `WEAVIATE_HTTP_PORT` (default: `8080`)
- `WEAVIATE_GRPC_HOST` (default: `localhost`)
- `WEAVIATE_GRPC_PORT` (default: `50051`)
- `WEAVIATE_SECURE` (default: `false`)
- `WEAVIATE_COLLECTION` (default: `NewsArticle`)
- `WEAVIATE_SOURCE_ID_PROPERTY` (benchmark default: `sourceRowId`)

Milvus:
- `MILVUS_HOST` (default: `localhost`)
- `MILVUS_PORT` (default: `19530`)
- ingest: `MILVUS_COLLECTION`, `MILVUS_INDEX_NAME`, `MILVUS_DISTANCE`/`MILVUS_METRIC`, `MILVUS_HNSW_M`, `MILVUS_HNSW_EF_CONSTRUCTION`, `BATCH_SIZE`, `RECREATE_COLLECTION`
- benchmark: `MILVUS_COLLECTION`, `MILVUS_DISTANCE`/`MILVUS_METRIC`, `MILVUS_CONSISTENCY_LEVEL`

FAISS:
- ingest: `FAISS_DISTANCE`, `FAISS_INDEX_TYPE`, `FAISS_IVF_NLIST`, `FAISS_HNSW_M`, `FAISS_OUTPUT_DIR`, `BATCH_SIZE`
- benchmark: `FAISS_OUTPUT_DIR`

## Notes

- Keep ingest and benchmark embedding model consistent.
- For Weaviate and Milvus, use the same distance during ingest and benchmark for valid comparisons.
- Milvus benchmark validates requested `--distance` against the collection index metric and fails early if they do not match.
