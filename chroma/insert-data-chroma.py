#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.embedding_cache import compute_without_cache, load_or_create_cache

CSV_PATH_DEFAULT = os.getenv("CSV_PATH", "train.csv")
CHROMA_MODE = os.getenv("CHROMA_MODE", "http").strip().lower()
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
COLLECTION_NAME_DEFAULT = os.getenv("COLLECTION_NAME", "news_articles")
MODEL_NAME_DEFAULT = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
EXPECTED_DIM = 1024
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE", "512"))
DISTANCE_DEFAULT = os.getenv("CHROMA_DISTANCE", "cosine").strip().lower()
CACHE_DIR_DEFAULT = os.getenv("EMBEDDING_CACHE_DIR", ".cache/embeddings")


def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def get_client():
    if CHROMA_MODE == "http":
        return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    if CHROMA_MODE == "persistent":
        return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    raise ValueError("CHROMA_MODE must be 'http' or 'persistent'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insert CSV news embeddings into Chroma")
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
        help="Chroma HNSW space metric for the collection.",
    )
    return parser.parse_args()


def cleanup_collection(client: chromadb.ClientAPI, collection_name: str) -> None:
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass


def main():
    args = parse_args()
    csv_path = args.csv_path
    model_name = args.model
    batch_size = int(args.batch_size)
    collection_name = args.collection
    distance = args.distance

    if CHROMA_MODE == "http":
        db_label = f"chroma://{CHROMA_HOST}:{CHROMA_PORT} (http)"
    else:
        db_label = f"chroma://{os.path.abspath(CHROMA_PERSIST_DIR)} (persistent)"

    print("DB:", db_label)
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

    emb_lists = [e.tolist() for e in bundle.vectors]

    client = get_client()
    cleanup_collection(client, collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance},
    )

    print("Inserting into Chroma...")
    t_write0 = time.perf_counter()

    for start in range(0, records_count, batch_size):
        end = min(start + batch_size, records_count)
        batch_ids = [str(i + 1) for i in range(start, end)]
        batch_embeddings = emb_lists[start:end]
        batch_metadatas = [
            {
                "title": bundle.titles[i],
                "description": bundle.descriptions[i],
                "genre": bundle.labels[i],
            }
            for i in range(start, end)
        ]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )

    write_seconds = time.perf_counter() - t_write0

    print("Building HNSW index...")
    t_idx0 = time.perf_counter()
    _ = collection.query(
        query_embeddings=[emb_lists[0]],
        n_results=1,
        include=["distances"],
    )
    index_seconds = time.perf_counter() - t_idx0

    db_count = collection.count()
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
