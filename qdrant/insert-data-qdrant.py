#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.embedding_cache import compute_without_cache, load_or_create_cache

CSV_PATH_DEFAULT = os.getenv("CSV_PATH", "train.csv")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME_DEFAULT = os.getenv("COLLECTION_NAME", "news_articles")
DISTANCE_DEFAULT = os.getenv("QDRANT_DISTANCE", "cosine").strip().lower()
MODEL_NAME_DEFAULT = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
EXPECTED_DIM = 1024
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE", "128"))
POLL_INTERVAL_SEC = 0.5
MAX_WAIT_SEC = 60 * 30
CACHE_DIR_DEFAULT = os.getenv("EMBEDDING_CACHE_DIR", ".cache/embeddings")


def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def wait_until_collection_available(client: QdrantClient, collection: str, timeout_sec: int = 60) -> None:
    deadline = time.time() + timeout_sec
    while True:
        try:
            client.get_collection(collection)
            return
        except Exception:
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for collection to become available.")
            time.sleep(0.2)


def chunk_ranges(total: int, batch: int):
    for i in range(0, total, batch):
        yield i, min(i + batch, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insert CSV news embeddings into Qdrant")
    parser.add_argument("--csv-path", default=CSV_PATH_DEFAULT)
    parser.add_argument("--model", default=MODEL_NAME_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--collection", default=COLLECTION_NAME_DEFAULT)
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--force-rebuild-embeddings", action="store_true")
    parser.add_argument("--no-embedding-cache", action="store_true")
    parser.add_argument(
        "--distance",
        choices=["cosine", "dot", "euclid", "manhattan"],
        default=DISTANCE_DEFAULT,
        help="Distance metric for the Qdrant collection.",
    )
    return parser.parse_args()


def parse_distance(distance: str) -> qm.Distance:
    mapping = {
        "cosine": qm.Distance.COSINE,
        "dot": qm.Distance.DOT,
        "euclid": qm.Distance.EUCLID,
        "manhattan": qm.Distance.MANHATTAN,
    }
    if distance not in mapping:
        raise ValueError(f"Unsupported distance: {distance}")
    return mapping[distance]


def wait_until_optimized(client: QdrantClient, collection: str) -> float:
    start = time.perf_counter()
    deadline = start + MAX_WAIT_SEC

    while True:
        info = client.get_collection(collection)
        status = getattr(info, "optimizer_status", None)
        if status is None and hasattr(info, "dict"):
            status = info.dict().get("optimizer_status")

        if status == "ok":
            return time.perf_counter() - start

        if time.perf_counter() > deadline:
            raise TimeoutError(
                f"Timed out waiting for collection optimization (last optimizer_status={status})."
            )

        time.sleep(POLL_INTERVAL_SEC)


def main():
    args = parse_args()
    csv_path = args.csv_path
    model_name = args.model
    batch_size = int(args.batch_size)
    collection_name = args.collection
    distance = args.distance
    distance_enum = parse_distance(distance)

    print("Qdrant:", QDRANT_URL)
    print("Model:", model_name)
    print("Distance:", distance)
    print("Collection:", collection_name)

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

    vectors = [v.tolist() for v in bundle.vectors]
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(collection_name):
        print("Deleting existing collection...")
        client.delete_collection(collection_name)

    print("Creating collection...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(size=EXPECTED_DIM, distance=distance_enum),
    )

    print(f"Inserting into Qdrant (batch_size={batch_size})...")
    t_write0 = time.perf_counter()

    for start, end in chunk_ranges(records_count, batch_size):
        ids = list(range(start, end))
        payloads = [
            {
                "title": bundle.titles[i],
                "description": bundle.descriptions[i],
                "genre": bundle.labels[i],
            }
            for i in range(start, end)
        ]

        client.upsert(
            collection_name=collection_name,
            points=qm.Batch(
                ids=ids,
                vectors=vectors[start:end],
                payloads=payloads,
            ),
        )

    write_seconds = time.perf_counter() - t_write0

    wait_until_collection_available(client, collection_name)
    print("Building Qdrant index...")
    index_seconds = wait_until_optimized(client, collection_name)

    info = client.get_collection(collection_name)
    points_count = info.points_count or 0
    approx_vec_mb = bytes_to_mb(points_count * EXPECTED_DIM * 4)

    print("\n==== SUMMARY ====")
    print(f"Rows inserted              : {points_count}")
    print(f"CSV size (MB)              : {bundle.csv_size_mb:.2f}")
    print(f"Approx embedding data (MB) : {approx_vec_mb:.2f} (float32 vectors only)")
    print(f"Embedding source           : {bundle.source}")
    print(f"Embedding time (s)         : {bundle.embed_seconds:.3f}")
    print(f"DB write time (s)          : {write_seconds:.3f}")
    print(f"Index build time (s)       : {index_seconds:.3f}")
    print("=================\n")


if __name__ == "__main__":
    main()
