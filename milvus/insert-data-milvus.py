#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.embedding_cache import load_or_create_cache, compute_without_cache

CSV_PATH_DEFAULT = "train.csv"
DEFAULT_TABLE_NAME = "news_articles"
DEFAULT_INDEX_NAME = "news_articles_embedding_hnsw"
DEFAULT_MODEL_NAME = "intfloat/e5-large-v2"
EXPECTED_DIM = 1024
POLL_INTERVAL_SEC = 0.5
MAX_WAIT_SEC = 60 * 20
CACHE_DIR_DEFAULT = os.getenv("EMBEDDING_CACHE_DIR", ".cache/embeddings")


def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def truncate_str(x: str, max_len: int) -> str:
    s = "" if x is None else str(x)
    return s[:max_len]


def normalize_metric_type(metric: str) -> str:
    m = str(metric or "").strip().upper()
    if m == "DOT":
        return "IP"
    if m in ("EUCLID", "EUCLIDEAN"):
        return "L2"
    if m in ("COSINE", "IP", "L2"):
        return m
    raise ValueError(f"Unsupported metric: {metric!r}. Use COSINE, DOT, IP, or L2.")


def cleanup_collection(table_name: str) -> None:
    if utility.has_collection(table_name):
        utility.drop_collection(table_name)


def wait_for_index(collection: Collection) -> None:
    deadline = time.time() + MAX_WAIT_SEC
    while True:
        try:
            if collection.indexes and len(collection.indexes) > 0:
                return
        except Exception:
            pass

        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for Milvus index to become visible.")
        time.sleep(POLL_INTERVAL_SEC)


def parse_args():
    parser = argparse.ArgumentParser(description="Insert CSV news embeddings into Milvus (CLI-first config)")
    parser.add_argument("--csv-path", default=os.getenv("CSV_PATH", CSV_PATH_DEFAULT))
    parser.add_argument("--host", default=os.getenv("MILVUS_HOST", "localhost"))
    parser.add_argument("--port", default=os.getenv("MILVUS_PORT", "19530"))
    parser.add_argument("--collection", default=os.getenv("MILVUS_COLLECTION", DEFAULT_TABLE_NAME))
    parser.add_argument("--index-name", default=os.getenv("MILVUS_INDEX_NAME", DEFAULT_INDEX_NAME))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME))
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--force-rebuild-embeddings", action="store_true")
    parser.add_argument("--no-embedding-cache", action="store_true")
    parser.add_argument(
        "--distance",
        default=os.getenv("MILVUS_DISTANCE", os.getenv("MILVUS_METRIC", "cosine")),
        help="Distance/metric for Milvus index: cosine | ip | dot | l2",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "1000")))
    parser.add_argument(
        "--recreate",
        choices=["true", "false"],
        default=os.getenv("RECREATE_COLLECTION", "true").lower(),
        help="Whether to drop and recreate the collection (default: true)",
    )
    parser.add_argument("--hnsw-m", type=int, default=int(os.getenv("MILVUS_HNSW_M", "16")))
    parser.add_argument("--hnsw-ef-construction", type=int, default=int(os.getenv("MILVUS_HNSW_EF_CONSTRUCTION", "200")))
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = args.csv_path
    host = args.host
    port = str(args.port)
    table_name = args.collection
    index_name = args.index_name
    model_name = args.model
    metric_type = normalize_metric_type(args.distance)
    batch_size = int(args.batch_size)
    recreate_collection = str(args.recreate).lower() == "true"

    print("DB:", f"milvus://{host}:{port}")
    print("Model:", model_name)
    print("Distance (Milvus metric):", metric_type)
    print("Collection:", table_name)
    print("Recreate:", recreate_collection)

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

    emb_lists = bundle.vectors.tolist()
    ids = list(range(records_count))
    titles = [truncate_str(x, 1024) for x in bundle.titles]
    descriptions = [truncate_str(x, 8192) for x in bundle.descriptions]
    genres = [truncate_str(x, 64) for x in bundle.labels]

    connections.connect(alias="default", host=host, port=port)

    if recreate_collection:
        cleanup_collection(table_name)
    elif utility.has_collection(table_name):
        raise RuntimeError(
            f"Collection '{table_name}' already exists. Use --recreate true to overwrite or choose a different --collection."
        )

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="genre", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EXPECTED_DIM),
    ]
    schema = CollectionSchema(fields=fields, description="News articles with embeddings")
    collection = Collection(name=table_name, schema=schema, using="default", shards_num=2)

    print("Inserting into Milvus...")
    t_write0 = time.perf_counter()

    for start in range(0, records_count, batch_size):
        end = min(start + batch_size, records_count)
        batch_data = [
            ids[start:end],
            titles[start:end],
            descriptions[start:end],
            genres[start:end],
            emb_lists[start:end],
        ]
        collection.insert(batch_data)

    collection.flush()
    write_seconds = time.perf_counter() - t_write0

    print("Building HNSW index...")
    t_idx0 = time.perf_counter()

    index_params = {
        "index_type": "HNSW",
        "metric_type": metric_type,
        "params": {
            "M": int(args.hnsw_m),
            "efConstruction": int(args.hnsw_ef_construction),
        },
    }

    collection.create_index(
        field_name="embedding",
        index_params=index_params,
        index_name=index_name,
    )

    wait_for_index(collection)

    try:
        print("Index metadata after creation:")
        for idx in collection.indexes:
            print(" -", getattr(idx, "params", None))
    except Exception as e:
        print("Could not inspect index metadata:", e)

    collection.load()
    index_seconds = time.perf_counter() - t_idx0
    db_count = int(collection.num_entities)
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

    connections.disconnect("default")


if __name__ == "__main__":
    main()
