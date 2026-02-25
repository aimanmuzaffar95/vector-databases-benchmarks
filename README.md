# Vector DB Embedding Ingest + Recall/QPS Benchmarks

This project ingests `train.csv`, generates 1024-d embeddings using `intfloat/e5-large-v2`, stores vectors in multiple backends, and benchmarks Recall@K + latency plus QPS throughput.

Special thanks to [Tahir Saeed](https://github.com/tahir-arbisoft) for his collaboration on this project.

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
pip3 install -r requirements.txt
```

## Quick Start (Unified Runner)

Use `run_benchmarks.py` to orchestrate docker / insert / recall benchmark / QPS steps across one or more backends.

```bash
python3 run_benchmarks.py -d all -D -i -r -q -o outputs/benchmark.csv
```

## CLI Parameters (Complete Reference)

- `-d, --db-name` (required): target backends. Use comma-separated names (`pgvector,qdrant`) or `all`.
- `-D, --docker`: start backend containers using each backend's compose file. `faiss` is `N/A (local backend)`.
- `-i, --insert`: run backend insert scripts. If enabled, shared embeddings are precomputed once before the first backend insert unless disabled.
- `-r, --recall`: run recall/latency benchmark scripts (`benchmark-*.py`).
- `-q, --qps`: run throughput benchmark scripts (`benchmark-qps-*.py`).
- `-I, --insert-args "..."`: pass-through args only for insert scripts.
- `-R, --recall-args "..."`: pass-through args only for recall scripts.
- `-Q, --qps-args "..."`: pass-through args only for QPS scripts.
- `-o, --benchmark-csv-path`: output path for consolidated benchmark CSV (default: `benchmark.csv`).
- Backward-compatible aliases are still accepted: `-dbname`, `--dbname`, and `--recal`.

Parameter defaults and behavior:

- At least one action flag is required: `--docker`, `--insert`, `--recall`, or `--qps` (or short forms).
- `--db-name all` (or `-d all`) expands to: `pgvector,chroma,qdrant,weaviate,milvus,faiss`.
- `--distance cosine` is auto-added to insert/recall/qps arg groups if missing.
- `--num-queries 480` is auto-added to recall args if missing.
- `--insert` precompute step is skipped when `--insert-args` includes `--no-embedding-cache`.
- Invalid quoted arg strings (for example unmatched quotes) in `--insert-args`, `--recall-args`, or `--qps-args` will stop execution with a parser error.
- Docker handling:
  - Fatal and immediate stop: Docker daemon not running, or Docker CLI not found.
  - Per-DB docker/container issues are reported in `Summary`; that DB's benchmark steps are marked as skipped while other DBs continue.

## Action Combinations (All Valid Non-Empty Sets)

Template:

```bash
python3 run_benchmarks.py -d <db|db1,db2|all> <action flags>
```

Every possible action combination:

```bash
# 1 action
python3 run_benchmarks.py -d all -D
python3 run_benchmarks.py -d all -i
python3 run_benchmarks.py -d all -r
python3 run_benchmarks.py -d all -q

# 2 actions
python3 run_benchmarks.py -d all -D -i
python3 run_benchmarks.py -d all -D -r
python3 run_benchmarks.py -d all -D -q
python3 run_benchmarks.py -d all -i -r
python3 run_benchmarks.py -d all -i -q
python3 run_benchmarks.py -d all -r -q

# 3 actions
python3 run_benchmarks.py -d all -D -i -r
python3 run_benchmarks.py -d all -D -i -q
python3 run_benchmarks.py -d all -D -r -q
python3 run_benchmarks.py -d all -i -r -q

# 4 actions
python3 run_benchmarks.py -d all -D -i -r -q
```

Recommended practical examples:

```bash
# Insert only for pgvector and qdrant
python3 run_benchmarks.py -d pgvector,qdrant -i

# Force rebuild of shared embedding cache before insert
python3 run_benchmarks.py -d all -i -I "--force-rebuild-embeddings"

# Use a custom embedding cache directory
python3 run_benchmarks.py -d all -i -I "--cache-dir .cache/embeddings"

# Disable shared cache and fall back to per-script embedding
python3 run_benchmarks.py -d all -i -I "--no-embedding-cache"

# Recall benchmark only faiss
python3 run_benchmarks.py -d faiss -r

# Forward extra args to recall benchmark scripts
python3 run_benchmarks.py -d all -r -R "--k-values 1,5,10 --num-queries 300"

# Forward extra args to QPS scripts
python3 run_benchmarks.py -d qdrant -q -Q "--k 10 --seconds 20 --concurrency 8"
```

## QPS Benchmarks

QPS scripts can be run via `run_benchmarks.py -q` (or `--qps`) or standalone per backend.

Common example pattern:

```bash
python3 <backend>/benchmark-qps-<backend>.py --distance cosine --k 10 --seconds 20 --concurrency 8
```

Examples:

```bash
python3 pgvector/benchmark-qps-pgvector.py --distance cosine --k 10 --seconds 20 --concurrency 8
python3 chroma/benchmark-qps-chroma.py --distance cosine --k 10 --seconds 20 --concurrency 8
python3 qdrant/benchmark-qps-qdrant.py --distance cosine --k 10 --seconds 20 --concurrency 8
python3 weaviate/benchmark-qps-weaviate.py --distance cosine --k 10 --seconds 20 --concurrency 8
python3 milvus/benchmark-qps-milvus.py --distance cosine --k 10 --seconds 20 --concurrency 8
python3 faiss/benchmark-qps-faiss.py --distance cosine --k 10 --seconds 20 --concurrency 8
```

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

QPS scripts (`benchmark-qps-*.py`) print in this standardized format:

```text
===============
Benchmark Results
===============
Run: default
Distance: cosine
k: 10
-------------------------------------
Concurrency: 8
Duration (s): 20.00 (warmup 2.00s)
Measured queries: 394099
QPS: 21894.39
-------------------------------------
Latency avg: 0.36 ms
Latency p50: 0.30 ms
Latency p95: 0.71 ms
Latency p99: 1.36 ms
-------------------------------------
```

After all benchmark scripts finish, `run_benchmarks.py` prints separate final tables:

- `Final Recall Benchmark Table`
- `Final QPS Benchmark Table`

Recall table columns:
- `DB`
- `Run`
- `Distance`
- `Queries`
- `Recall@1`
- `Recall@5`
- `Recall@10`
- `Lat avg (ms)`
- `Lat p50 (ms)`
- `Lat p95 (ms)`

QPS table columns:
- `DB`
- `Run`
- `Distance`
- `Queries`
- `QPS`
- `Lat avg (ms)`
- `Lat p50 (ms)`
- `Lat p95 (ms)`
- `Lat p99 (ms)`

`run_benchmarks.py` also writes a consolidated benchmark CSV via `-o/--benchmark-csv-path` (default: `benchmark.csv`).

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

### QPS benchmark scripts
- `pgvector/benchmark-qps-pgvector.py`
- `chroma/benchmark-qps-chroma.py`
- `qdrant/benchmark-qps-qdrant.py`
- `weaviate/benchmark-qps-weaviate.py`
- `milvus/benchmark-qps-milvus.py`
- `faiss/benchmark-qps-faiss.py`

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

- `run_benchmarks.py` defaults to `--distance cosine` for insert and recall benchmark if you do not pass `--distance`.
- `run_benchmarks.py` defaults recall benchmark scripts to `--num-queries 480` if you do not pass `--num-queries`.
- Benchmarks default `--warmup-queries` to `0`, so measured queries match requested queries by default.
- Shared embedding cache uses `NPZ + JSON`.
- Cache invalidates/rebuilds based on CSV content digest, model name, and embedding text prefix version.
- The same cached vectors are reused by all insert scripts in a run.
- Use the same `--distance` family during insert and recall/qps benchmarking for fair comparisons.
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
