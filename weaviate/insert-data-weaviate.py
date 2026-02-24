#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.data import DataObject

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.embedding_cache import compute_without_cache, load_or_create_cache

CSV_PATH_DEFAULT = os.getenv("CSV_PATH", "train.csv")
WEAVIATE_HTTP_HOST = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "localhost")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_SECURE = os.getenv("WEAVIATE_SECURE", "false").lower() == "true"
COLLECTION_NAME_DEFAULT = os.getenv("COLLECTION_NAME", os.getenv("WEAVIATE_COLLECTION", "NewsArticle"))
MODEL_NAME_DEFAULT = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
EXPECTED_DIM = 1024
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE", "128"))
POLL_INTERVAL_SEC = 0.5
MAX_WAIT_SEC = 60 * 10
CACHE_DIR_DEFAULT = os.getenv("EMBEDDING_CACHE_DIR", ".cache/embeddings")

DISTANCE_MAP = {
    "cosine": VectorDistances.COSINE,
    "dot": VectorDistances.DOT,
    "l2": VectorDistances.L2_SQUARED,
}


def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def chunk_ranges(total: int, batch: int):
    for i in range(0, total, batch):
        yield i, min(i + batch, total)


def connect_weaviate() -> weaviate.WeaviateClient:
    return weaviate.connect_to_custom(
        http_host=WEAVIATE_HTTP_HOST,
        http_port=WEAVIATE_HTTP_PORT,
        http_secure=WEAVIATE_SECURE,
        grpc_host=WEAVIATE_GRPC_HOST,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=WEAVIATE_SECURE,
    )


def wait_until_ready(client: weaviate.WeaviateClient, timeout_sec: int = 60) -> None:
    deadline = time.time() + timeout_sec
    while True:
        try:
            if client.is_ready():
                return
        except Exception:
            pass
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for Weaviate to become ready.")
        time.sleep(0.2)


def recreate_collection(client: weaviate.WeaviateClient, collection_name: str, distance: str) -> None:
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    client.collections.create(
        name=collection_name,
        description="News articles with externally generated sentence embeddings (E5).",
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(distance_metric=DISTANCE_MAP[distance])
        ),
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
            Property(name="genre", data_type=DataType.TEXT),
            Property(name="sourceRowId", data_type=DataType.INT),
        ],
    )


def wait_until_count(client: weaviate.WeaviateClient, collection_name: str, expected_count: int) -> float:
    start = time.perf_counter()
    deadline = start + MAX_WAIT_SEC
    collection = client.collections.get(collection_name)

    while True:
        try:
            agg = collection.aggregate.over_all(total_count=True)
            total_count = int(agg.total_count or 0)
            if total_count >= expected_count:
                return time.perf_counter() - start
        except Exception:
            pass

        if time.perf_counter() > deadline:
            raise TimeoutError(
                f"Timed out waiting for object count to reach {expected_count} in collection {collection_name!r}."
            )
        time.sleep(POLL_INTERVAL_SEC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default=CSV_PATH_DEFAULT)
    parser.add_argument("--model", default=MODEL_NAME_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--collection", default=COLLECTION_NAME_DEFAULT)
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--force-rebuild-embeddings", action="store_true")
    parser.add_argument("--no-embedding-cache", action="store_true")
    parser.add_argument(
        "--distance",
        choices=sorted(DISTANCE_MAP.keys()),
        default="cosine",
        help="Vector distance metric for Weaviate HNSW index.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = args.csv_path
    model_name = args.model
    batch_size = int(args.batch_size)
    collection_name = args.collection

    print("Weaviate HTTP:", f"http{'s' if WEAVIATE_SECURE else ''}://{WEAVIATE_HTTP_HOST}:{WEAVIATE_HTTP_PORT}")
    print("Weaviate gRPC:", f"{WEAVIATE_GRPC_HOST}:{WEAVIATE_GRPC_PORT}")
    print("Collection:", collection_name)
    print("Model:", model_name)
    print("Distance metric:", args.distance)

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
    client = connect_weaviate()

    try:
        wait_until_ready(client)
        print("Recreating Weaviate collection...")
        recreate_collection(client, collection_name, args.distance)

        collection = client.collections.get(collection_name)
        print(f"Inserting into Weaviate (batch_size={batch_size})...")
        t_write0 = time.perf_counter()

        for start, end in chunk_ranges(records_count, batch_size):
            batch_objects = []
            for i in range(start, end):
                props = {
                    "title": bundle.titles[i],
                    "description": bundle.descriptions[i],
                    "genre": bundle.labels[i],
                    "sourceRowId": int(i),
                }
                batch_objects.append(DataObject(properties=props, vector=emb_lists[i]))

            collection.data.insert_many(batch_objects)

        write_seconds = time.perf_counter() - t_write0
        print("Waiting for Weaviate object count to reach expected size...")
        index_seconds = wait_until_count(client, collection_name, records_count)

        agg = collection.aggregate.over_all(total_count=True)
        db_count = int(agg.total_count or 0)
        approx_vec_mb = bytes_to_mb(db_count * EXPECTED_DIM * 4)

        print("\n==== SUMMARY ====")
        print(f"Rows inserted              : {db_count}")
        print(f"CSV size (MB)              : {bundle.csv_size_mb:.2f}")
        print(f"Approx embedding data (MB) : {approx_vec_mb:.2f} (float32 vectors only)")
        print(f"Embedding source           : {bundle.source}")
        print(f"Embedding time (s)         : {bundle.embed_seconds:.3f}")
        print(f"DB write time (s)          : {write_seconds:.3f}")
        print(f"Index build time (s)       : {index_seconds:.3f} (count visibility/settling proxy)")
        print("=================\n")
    finally:
        client.close()


if __name__ == "__main__":
    main()
