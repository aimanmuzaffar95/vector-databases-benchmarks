# Vector DB Embedding Ingest + Recall Benchmarks

This project ingests `train.csv`, generates embeddings with `intfloat/e5-large-v2` (1024 dims), stores vectors in multiple backends, and benchmarks Recall@K plus latency.

Backends:
- pgvector (PostgreSQL)
- Chroma
- Qdrant
- Weaviate
- Milvus
- FAISS

## Requirements

- Python 3.9+
- Docker + Docker Compose (for DB backends)

Install dependencies:

```bash
pip install pandas sqlalchemy psycopg2-binary sentence-transformers numpy chromadb qdrant-client weaviate-client pymilvus faiss-cpu
```

## Start Databases

pgvector:
```bash
docker compose -f docker-compose-pgvector.yml up -d
```

Chroma:
```bash
docker compose -f docker-compose-chroma.yml up -d
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

## Canonical CLI Flags

The scripts now use canonical names (breaking change):
- distance: `--distance`
- load cap in benchmark scripts: `--max-load-items`

Removed legacy flags:
- `--metric`
- `--max-load-rows`
- `--max-load-points`

## Ingest Commands

### pgvector
```bash
python3 pgvector/insert-data-pgvector.py
python3 pgvector/insert-data-pgvector.py --collection news_articles --distance cosine --batch-size 1000
python3 pgvector/insert-data-pgvector.py --distance ip
```

### Chroma
```bash
python3 chroma/insert-data-chroma.py
python3 chroma/insert-data-chroma.py --collection news_articles --distance cosine --batch-size 512
python3 chroma/insert-data-chroma.py --distance l2
```

### Qdrant
```bash
python3 qdrant/insert-data-qdrant.py
python3 qdrant/insert-data-qdrant.py --collection news_articles --distance cosine --batch-size 128
python3 qdrant/insert-data-qdrant.py --distance dot
```

### Weaviate
```bash
python3 weaviate/insert-data-weaviate.py
python3 weaviate/insert-data-weaviate.py --collection NewsArticle --distance cosine --batch-size 128
python3 weaviate/insert-data-weaviate.py --distance l2
```

### Milvus
```bash
python3 milvus/insert-data-milvus.py
python3 milvus/insert-data-milvus.py --collection news_articles --distance cosine --batch-size 1000
python3 milvus/insert-data-milvus.py --distance dot
```

### FAISS
```bash
python3 faiss/insert-data-faiss.py --distance cosine --index-type hnsw --overwrite
python3 faiss/insert-data-faiss.py --distance dot --index-type ivfflat --ivf-nlist 1024 --overwrite
python3 faiss/insert-data-faiss.py --distance l2 --index-type flat --overwrite
```

## Benchmark Commands

### pgvector
```bash
python3 pgvector/benchmark_pgvector.py
python3 pgvector/benchmark_pgvector.py --distance cosine --k-values 1,5,10 --num-queries 500
python3 pgvector/benchmark_pgvector.py --distance ip --hnsw-ef-search 20,40,80,120
python3 pgvector/benchmark_pgvector.py --distance cosine --ivfflat-probes 1,5,10,20,50
python3 pgvector/benchmark_pgvector.py --distance cosine --max-load-items 1000 --where-sql "genre = :g" --where-param g=Sports
```

### Chroma
```bash
python3 chroma/benchmark_chroma.py
python3 chroma/benchmark_chroma.py --distance cosine --k-values 1,5,10 --num-queries 500
python3 chroma/benchmark_chroma.py --distance cosine --max-load-items 1000
python3 chroma/benchmark_chroma.py --distance cosine --where-json '{"genre":"Sports"}'
```

### Qdrant
```bash
python3 qdrant/benchmark_qdrant.py
python3 qdrant/benchmark_qdrant.py --distance cosine --k-values 1,5,10 --num-queries 500
python3 qdrant/benchmark_qdrant.py --distance cosine --hnsw-ef 32,64,128,256
python3 qdrant/benchmark_qdrant.py --distance cosine --max-load-items 1000 --exact-qdrant
```

### Weaviate
```bash
python3 weaviate/benchmark_weaviate.py
python3 weaviate/benchmark_weaviate.py --distance cosine --k-values 1,5,10 --num-queries 500
python3 weaviate/benchmark_weaviate.py --distance dot --hnsw-ef 32,64,128
python3 weaviate/benchmark_weaviate.py --distance l2 --max-load-items 1000 --exact-weaviate
```

### Milvus
```bash
python3 milvus/benchmark_milvus.py
python3 milvus/benchmark_milvus.py --distance cosine --consistency-level Bounded
python3 milvus/benchmark_milvus.py --distance cosine --hnsw-ef 32,64,128
python3 milvus/benchmark_milvus.py --distance cosine --nprobe 8,16,32
python3 milvus/benchmark_milvus.py --distance cosine --k-values 1,5,10 --num-queries 200
```

### FAISS
```bash
python3 faiss/benchmark_faiss.py
python3 faiss/benchmark_faiss.py --distance cosine --k-values 1,5,10 --num-queries 500
python3 faiss/benchmark_faiss.py --distance dot --nprobe 4,8,16,32
python3 faiss/benchmark_faiss.py --distance cosine --hnsw-ef-search 32,64,128,256
```

## Useful Shared Insert Flags

Most insert scripts accept:
- `--csv-path`
- `--model`
- `--batch-size`
- `--collection`
- `--distance`

FAISS and Milvus have additional backend-specific flags (for example index settings).

## Environment Variables

Common:
- `EMBEDDING_MODEL` (default: `intfloat/e5-large-v2`)
- `CSV_PATH` (default: `train.csv`)
- `BATCH_SIZE` (backend-specific defaults)

pgvector:
- `DATABASE_URL`
- benchmark: `TABLE_NAME`, `ID_COLUMN`, `EMBEDDING_COLUMN`, `GENRE_COLUMN`

Chroma:
- `CHROMA_MODE`, `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_PERSIST_DIR`
- `COLLECTION_NAME`

Qdrant:
- `QDRANT_URL`
- `COLLECTION_NAME`

Weaviate:
- `WEAVIATE_HTTP_HOST`, `WEAVIATE_HTTP_PORT`
- `WEAVIATE_GRPC_HOST`, `WEAVIATE_GRPC_PORT`
- `WEAVIATE_SECURE`
- `WEAVIATE_COLLECTION`

Milvus:
- `MILVUS_HOST`, `MILVUS_PORT`
- `MILVUS_COLLECTION`, `MILVUS_INDEX_NAME`
- `MILVUS_DISTANCE` / `MILVUS_METRIC`
- `MILVUS_HNSW_M`, `MILVUS_HNSW_EF_CONSTRUCTION`
- `MILVUS_CONSISTENCY_LEVEL`

FAISS:
- `FAISS_DISTANCE`, `FAISS_INDEX_TYPE`
- `FAISS_IVF_NLIST`, `FAISS_HNSW_M`
- `FAISS_OUTPUT_DIR`

## Notes

- Keep ingest and benchmark embedding model consistent.
- For backend comparisons, use matching `--distance` during ingest and benchmark.
- Milvus benchmark validates that requested `--distance` matches collection index metric.
