# PGVector + Qdrant News Embedding Demo

This project demonstrates how to:
- ingest a news dataset,
- generate sentence embeddings,
- store vectors in either **PostgreSQL + pgvector** or **Qdrant**,
- run semantic search, and
- benchmark retrieval quality (Recall@K) for pgvector.

The default embedding model is `intfloat/e5-large-v2` (1024 dimensions).

## Project Structure

- `train.csv`  
  Input dataset used by the pgvector pipeline (`Class Index`, `Title`, `Description`).
- `pgvector/main.py`  
  Loads `train.csv`, builds embeddings, recreates `news_articles`, inserts vectors, builds HNSW index.
- `pgvector/search_pgvector.py`  
  Runs semantic search against pgvector table using editable in-file variables.
- `pgvector/benchmark_pgvector_recallAtK.py`  
  Benchmarks pgvector retrieval recall/latency with optional ANN parameter sweeps.
- `qdrant/main-qdrant.py`  
  Ingests dataset and vectors into Qdrant collection.
- `docker-compose.yml`  
  Postgres + pgAdmin setup.
- `docker-compose-qdrant.yml`  
  Qdrant setup.
- `db/init/001_enable_pgvector.sql`  
  Enables pgvector extension.

## Requirements

- Python 3.9+
- Docker + Docker Compose
- Recommended Python packages:
  - `pandas`
  - `sqlalchemy`
  - `psycopg2-binary`
  - `sentence-transformers`
  - `numpy`
  - `qdrant-client`
  - `wget`

Install dependencies (inside your virtual environment):

```bash
pip install pandas sqlalchemy psycopg2-binary sentence-transformers numpy qdrant-client wget
```

## 1) Start PostgreSQL + pgvector

From the project root:

```bash
docker compose up -d
```

Default DB connection used by scripts:

`postgresql+psycopg2://postgres:postgres@localhost:5432/appdb`

Optional pgAdmin:
- URL: `http://localhost:5050`
- Email: `admin@example.com`
- Password: `admin`

## 2) Ingest vectors into pgvector

Run:

```bash
python3 pgvector/main.py
```

What it does:
- validates `train.csv` format,
- maps labels (1/2/3/4 -> World/Sports/Business/Sci/Tech),
- builds E5-style passage embeddings (`passage: <title + description>`),
- drops and recreates `news_articles`,
- inserts vectors and builds HNSW cosine index.

## 3) Search vectors in pgvector

Edit these variables in `pgvector/search_pgvector.py`:
- `QUERY_TEXT` or `QUERY_FILE`
- `TOP_K`
- `GENRE_FILTER` (optional)
- `MODEL_NAME` (must match ingestion model)

Then run:

```bash
python3 pgvector/search_pgvector.py
```

Notes:
- Query is encoded as E5-style `query: <text>`.
- Search uses cosine distance (`<=>`) and returns top-k rows with similarity score.

## 4) Benchmark pgvector Recall@K

Basic benchmark:

```bash
python3 pgvector/benchmark_pgvector_recallAtK.py
```

Custom settings:

```bash
python3 pgvector/benchmark_pgvector_recallAtK.py --k-values 1,5,10 --num-queries 500 --metric cosine
```

Optional sweeps:

```bash
python3 pgvector/benchmark_pgvector_recallAtK.py --metric cosine --hnsw-ef-search 20,40,80,120
python3 pgvector/benchmark_pgvector_recallAtK.py --metric cosine --ivfflat-probes 1,5,10,20,50
```

## Optional: Qdrant Pipeline

Start Qdrant:

```bash
docker compose -f docker-compose-qdrant.yml up -d
```

Run Qdrant ingest:

```bash
python3 qdrant/main-qdrant.py
```

This script downloads/uses `AG_news_samples.csv`, embeds descriptions, recreates `news_articles` collection, and upserts vectors in batches.

## Common Troubleshooting

- `ModuleNotFoundError: No module named 'psycopg2'`  
  Install driver: `pip install psycopg2-binary`.

- pgvector operator/type errors (`No operator matches...`)  
  Ensure query vector is cast to `vector` and connection points to the correct DB/table populated by `pgvector/main.py`.

- Poor search relevance  
  Re-run `pgvector/main.py` after changing embedding logic/model so stored vectors and query encoding stay aligned.

