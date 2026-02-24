#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

import sqlalchemy as sa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.embedding_cache import compute_without_cache, load_or_create_cache

CSV_PATH_DEFAULT = os.getenv("CSV_PATH", "train.csv")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/appdb")
COLLECTION_NAME_DEFAULT = os.getenv("COLLECTION_NAME", "news_articles")
MODEL_NAME_DEFAULT = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
EXPECTED_DIM = 1024
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE", "1000"))
DISTANCE_DEFAULT = os.getenv("PGVECTOR_DISTANCE", "cosine").strip().lower()
CACHE_DIR_DEFAULT = os.getenv("EMBEDDING_CACHE_DIR", ".cache/embeddings")


def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insert CSV news embeddings into pgvector")
    parser.add_argument("--csv-path", default=CSV_PATH_DEFAULT)
    parser.add_argument("--model", default=MODEL_NAME_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--collection", default=COLLECTION_NAME_DEFAULT)
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--force-rebuild-embeddings", action="store_true")
    parser.add_argument("--no-embedding-cache", action="store_true")
    parser.add_argument(
        "--distance",
        choices=["cosine", "l2", "ip"],
        default=DISTANCE_DEFAULT,
        help="Distance metric used for pgvector HNSW operator class.",
    )
    return parser.parse_args()


def distance_to_ops(distance: str) -> str:
    if distance == "cosine":
        return "vector_cosine_ops"
    if distance == "l2":
        return "vector_l2_ops"
    if distance == "ip":
        return "vector_ip_ops"
    raise ValueError(f"Unsupported distance: {distance}")


def cleanup_table(engine: sa.Engine, table_name: str, index_name: str) -> None:
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP INDEX IF EXISTS {index_name};"))
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {table_name};"))


def main():
    args = parse_args()
    csv_path = args.csv_path
    model_name = args.model
    batch_size = int(args.batch_size)
    table_name = args.collection
    distance = args.distance
    index_name = f"{table_name}_embedding_hnsw"
    distance_ops = distance_to_ops(distance)

    print("DB:", DATABASE_URL)
    print("Model:", model_name)
    print("Distance:", distance)
    print("Collection:", table_name)

    if args.no_embedding_cache:
        bundle = compute_without_cache(csv_path=csv_path, model_name=model_name, expected_dim=EXPECTED_DIM)
    else:
        bundle = load_or_create_cache(
            csv_path=csv_path,
            model_name=model_name,
            expected_dim=EXPECTED_DIM,
            cache_dir=args.cache_dir,
            cache_key_suffix="shared",
            force_rebuild=bool(args.force_rebuild_embeddings),
        )

    records_count = bundle.vectors.shape[0]
    print(f"Using local dataset: {csv_path}")
    print(f"Records: {records_count} | CSV size: {bundle.csv_size_mb:.2f} MB")
    print(f"Embedding source: {bundle.source}")

    emb_lists = [e.tolist() for e in bundle.vectors]
    payload = [
        {
            "title": bundle.titles[i],
            "description": bundle.descriptions[i],
            "genre": bundle.labels[i],
            "embedding": emb_lists[i],
        }
        for i in range(records_count)
    ]

    engine = sa.create_engine(DATABASE_URL, future=True)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))

    cleanup_table(engine, table_name, index_name)

    with engine.begin() as conn:
        conn.execute(
            sa.text(
                f"""
            CREATE TABLE {table_name} (
                id BIGSERIAL PRIMARY KEY,
                title TEXT,
                description TEXT,
                genre TEXT,
                embedding vector({EXPECTED_DIM})
            );
        """
            )
        )

    insert_stmt = sa.text(
        f"""
        INSERT INTO {table_name} (title, description, genre, embedding)
        VALUES (:title, :description, :genre, :embedding)
    """
    )

    print("Inserting into Postgres...")
    t_write0 = time.perf_counter()
    with engine.begin() as conn:
        for start in range(0, records_count, batch_size):
            end = min(start + batch_size, records_count)
            conn.execute(insert_stmt, payload[start:end])
    write_seconds = time.perf_counter() - t_write0

    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP INDEX IF EXISTS {index_name};"))

    print("Building HNSW index...")
    t_idx0 = time.perf_counter()
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                f"""
            CREATE INDEX {index_name}
            ON {table_name}
            USING hnsw (embedding {distance_ops});
        """
            )
        )
    index_seconds = time.perf_counter() - t_idx0

    with engine.connect() as conn:
        db_count = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table_name};")).scalar_one()
    approx_vec_mb = bytes_to_mb(db_count * EXPECTED_DIM * 4)

    print("\n==== SUMMARY ====")
    print(f"Rows inserted              : {db_count}")
    print(f"CSV size (MB)              : {bundle.csv_size_mb:.2f}")
    print(f"Approx embedding data (MB) : {approx_vec_mb:.2f} (float32 vectors only)")
    print(f"Embedding source           : {bundle.source}")
    print(f"Embedding time (s)         : {bundle.embed_seconds:.3f}")
    print(f"DB write time (s)          : {write_seconds:.3f}")
    print(f"Index build time (s)       : {index_seconds:.3f}")
    print("=================\n")


if __name__ == "__main__":
    main()
