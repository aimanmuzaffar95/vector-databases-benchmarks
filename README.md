# Vector DB Embedding Ingest + Recall Benchmarks

This project ingests `train.csv`, generates 1024-d embeddings using `intfloat/e5-large-v2`, stores vectors in multiple backends, and benchmarks Recall@K + latency.

Supported backends:
- `pgvector` (PostgreSQL)
- `chroma`
- `qdrant`
- `weaviate`
- `milvus`
- `faiss`

## Requirements

- Python 3.9+
- Docker + Docker Compose (for non-FAISS backends)

Install dependencies:

```bash
pip install pandas sqlalchemy psycopg2-binary sentence-transformers numpy chromadb qdrant-client weaviate-client pymilvus faiss-cpu
```

## Quick Start (Unified Runner)

Use `run_benchmarks.py` to orchestrate docker / insert / benchmark steps across one or more backends.

```bash
python3 run_benchmarks.py -dbname all --docker --insert --benchmark
```

Common examples:

```bash
# Start containers + benchmark all backends
python3 run_benchmarks.py -dbname all --docker --benchmark

# Insert only for pgvector and qdrant
python3 run_benchmarks.py -dbname pgvector,qdrant --insert

# Insert all backends (shared embeddings precomputed once, then reused)
python3 run_benchmarks.py -dbname all --insert

# Force rebuild of shared embedding cache before insert
python3 run_benchmarks.py -dbname all --insert --insert-args "--force-rebuild-embeddings"

# Use a custom embedding cache directory
python3 run_benchmarks.py -dbname all --insert --insert-args "--cache-dir .cache/embeddings"

# Disable shared cache and fall back to per-script embedding
python3 run_benchmarks.py -dbname all --insert --insert-args "--no-embedding-cache"

# Benchmark only faiss
python3 run_benchmarks.py -dbname faiss --benchmark

# Forward extra args to benchmark scripts
python3 run_benchmarks.py -dbname all --benchmark --benchmark-args "--k-values 1,5,10 --num-queries 300"
```

### Unified Runner Flags

- `-dbname, --dbname` required; comma-separated backend names or `all`
- `--docker` start backend containers (`faiss` is skipped as local backend)
- `--insert` run insert scripts
- `--benchmark` run benchmark scripts
- `--insert-args "..."` pass-through args for insert scripts (supports `--cache-dir`, `--force-rebuild-embeddings`, `--no-embedding-cache`)
- `--benchmark-args "..."` pass-through args for benchmark scripts
- When `--insert` is used, `run_benchmarks.py` precomputes shared embeddings once before per-backend inserts unless `--no-embedding-cache` is passed.

## Benchmark Output Format

Each `benchmark-*.py` script prints runs in a standardized format:

```text
===============
Benchmark Results
===============
Run: default
Distance: cosine
Measured queries: 480
-------------------------------------
Recall@1: 1.0000
Recall@5: 0.9880
Recall@10: 0.9860
-------------------------------------
Latency avg: 0.25 ms
Latency p50: 0.16 ms
Latency p95: 0.37 ms
-------------------------------------
```

After all benchmark scripts finish, `run_benchmarks.py` prints a consolidated final table:

- `DB`
- `Run`
- `Distance`
- `Queries`
- `Recall@1`, `Recall@5`, `Recall@10`
- `Lat avg (ms)`, `Lat p50 (ms)`, `Lat p95 (ms)`

`run_benchmarks.py` also prints a consolidated insert table:

- `DB`
- `Rows`
- `Embedding Source`
- `Embedding time (s)`
- `Write time (s)`
- `Build time (s)`

## Script Inventory (Current)

### Insert scripts
- `pgvector/insert-data-pgvector.py`
- `chroma/insert-data-chroma.py`
- `qdrant/insert-data-qdrant.py`
- `weaviate/insert-data-weaviate.py`
- `milvus/insert-data-milvus.py`
- `faiss/insert-data-faiss.py`

### Benchmark scripts
- `pgvector/benchmark-pgvector.py`
- `chroma/benchmark-chroma.py`
- `qdrant/benchmark-qdrant.py`
- `weaviate/benchmark-weaviate.py`
- `milvus/benchmark-milvus.py`
- `faiss/benchmark-faiss.py`

### Shared embedding utilities
- `shared/embedding_cache.py`
- `shared/precompute_embeddings.py`

## Docker Compose Files

- `docker-compose-pgvector.yml`
- `docker-compose-chroma.yml`
- `docker-compose-qdrant.yml`
- `docker-compose-weviate.yml`
- `docker-compose-milvus.yml`

## Backend Notes

- `run_benchmarks.py` defaults to `--distance cosine` for insert and benchmark if you do not pass `--distance`.
- `run_benchmarks.py` defaults benchmark scripts to `--num-queries 480` if you do not pass `--num-queries`.
- Benchmarks default `--warmup-queries` to `0`, so measured queries match requested queries by default.
- Shared embedding cache uses `NPZ + JSON`.
- Cache invalidates/rebuilds based on CSV content digest, model name, and embedding text prefix version.
- The same cached vectors are reused by all insert scripts in a run.
- Use the same `--distance` family during insert and benchmark for fair comparisons.
- Milvus benchmark now fails fast if requested `--distance` does not match the collection index metric.
- Weaviate benchmark includes retry-based connection startup handling for container readiness.
- FAISS runs locally and does not require Docker.

## Environment Variables (Highlights)

Common:
- `CSV_PATH` (default: `train.csv`)
- `EMBEDDING_MODEL` (default: `intfloat/e5-large-v2`)

pgvector benchmark:
- `DATABASE_URL`
- `TABLE_NAME`, `ID_COLUMN`, `EMBEDDING_COLUMN`, `GENRE_COLUMN`

chroma:
- `CHROMA_MODE`, `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_PERSIST_DIR`
- `COLLECTION_NAME`

qdrant:
- `QDRANT_URL`
- `COLLECTION_NAME`

weaviate:
- `WEAVIATE_HTTP_HOST`, `WEAVIATE_HTTP_PORT`
- `WEAVIATE_GRPC_HOST`, `WEAVIATE_GRPC_PORT`
- `WEAVIATE_SECURE`, `WEAVIATE_COLLECTION`

milvus:
- `MILVUS_HOST`, `MILVUS_PORT`
- `MILVUS_COLLECTION`, `MILVUS_INDEX_NAME`
- `MILVUS_DISTANCE` / `MILVUS_METRIC`

faiss:
- `FAISS_DISTANCE`, `FAISS_INDEX_TYPE`
- `FAISS_IVF_NLIST`, `FAISS_HNSW_M`
- `FAISS_OUTPUT_DIR`
