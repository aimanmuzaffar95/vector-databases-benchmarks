#!/usr/bin/env python3
import argparse
import os
import time
from typing import Iterable

import pandas as pd
from sentence_transformers import SentenceTransformer

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.data import DataObject


CSV_PATH = "train.csv"

LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

# Weaviate connection (matches your docker-compose)
WEAVIATE_HTTP_HOST = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "localhost")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_SECURE = os.getenv("WEAVIATE_SECURE", "false").lower() == "true"

# Weaviate collection/class name (PascalCase is conventional)
COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "NewsArticle")

# Pick a 1024-d model
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
EXPECTED_DIM = 1024

# Batch config
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))

# Polling for count visibility after ingest
POLL_INTERVAL_SEC = 0.5
MAX_WAIT_SEC = 60 * 10  # 10 minutes safety cap


DISTANCE_MAP = {
    "cosine": VectorDistances.COSINE,
    "dot": VectorDistances.DOT,
    "l2": VectorDistances.L2_SQUARED,
}


def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def build_passage_text(title: str, description: str) -> str:
    combined = f"{title.strip()} {description.strip()}".strip()
    return f"passage: {combined}"


def chunk_ranges(total: int, batch: int):
    for i in range(0, total, batch):
        yield i, min(i + batch, total)


def connect_weaviate() -> weaviate.WeaviateClient:
    """
    Connect to local/self-hosted Weaviate using your docker-compose ports.
    """
    client = weaviate.connect_to_custom(
        http_host=WEAVIATE_HTTP_HOST,
        http_port=WEAVIATE_HTTP_PORT,
        http_secure=WEAVIATE_SECURE,
        grpc_host=WEAVIATE_GRPC_HOST,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=WEAVIATE_SECURE,
    )
    return client


def wait_until_ready(client: weaviate.WeaviateClient, timeout_sec: int = 60) -> None:
    """
    Basic readiness check.
    """
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


def recreate_collection(client: weaviate.WeaviateClient, distance: str) -> None:
    """
    Delete existing collection and create a fresh one.
    We disable server-side vectorization because you are providing vectors manually.
    """
    # Delete if exists
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)

    # Create collection with HNSW vector index (default in Weaviate, but made explicit)
    client.collections.create(
        name=COLLECTION_NAME,
        description="News articles with externally generated sentence embeddings (E5).",
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=DISTANCE_MAP[distance]
            )
        ),
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
            Property(name="genre", data_type=DataType.TEXT),
            Property(name="sourceRowId", data_type=DataType.INT),  # helpful for debugging/benchmarking
        ],
    )


def wait_until_count(client: weaviate.WeaviateClient, expected_count: int) -> float:
    """
    Polls aggregate total_count until it matches expected_count.
    Returns elapsed seconds from start of polling.
    """
    start = time.perf_counter()
    deadline = start + MAX_WAIT_SEC

    collection = client.collections.get(COLLECTION_NAME)

    while True:
        try:
            agg = collection.aggregate.over_all(total_count=True)
            total_count = int(agg.total_count or 0)
            if total_count >= expected_count:
                return time.perf_counter() - start
        except Exception:
            # transient during heavy ingest/startup
            pass

        if time.perf_counter() > deadline:
            raise TimeoutError(
                f"Timed out waiting for object count to reach {expected_count} "
                f"in collection {COLLECTION_NAME!r}."
            )
        time.sleep(POLL_INTERVAL_SEC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distance",
        choices=sorted(DISTANCE_MAP.keys()),
        default="cosine",
        help="Vector distance metric for Weaviate HNSW index.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Weaviate HTTP:", f"http{'s' if WEAVIATE_SECURE else ''}://{WEAVIATE_HTTP_HOST}:{WEAVIATE_HTTP_PORT}")
    print("Weaviate gRPC:", f"{WEAVIATE_GRPC_HOST}:{WEAVIATE_GRPC_PORT}")
    print("Collection:", COLLECTION_NAME)
    print("Model:", MODEL_NAME)
    print("Distance metric:", args.distance)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Local dataset not found: {CSV_PATH}")
    print(f"Using local dataset: {CSV_PATH}")

    csv_size_mb = bytes_to_mb(os.path.getsize(CSV_PATH))

    # 1) Load data
    df = pd.read_csv(CSV_PATH)
    expected_columns = {"Class Index", "Title", "Description"}
    received_columns = set(df.columns)
    if not expected_columns.issubset(received_columns):
        raise ValueError(f"Missing column(s). Expected: {expected_columns}, Found: {received_columns}")

    df = df.rename(columns={
        "Class Index": "label",
        "Title": "title",
        "Description": "description",
    })

    # Normalize fields
    df["label"] = df["label"].map(LABEL_MAP).fillna("Unknown")
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

    records_count = len(df)
    print(f"Records: {records_count} | CSV size: {csv_size_mb:.2f} MB")

    # 2) Load model + verify dim
    print("Loading sentence-transformers model...")
    model = SentenceTransformer(MODEL_NAME)

    dim = model.get_sentence_embedding_dimension()
    print("Detected embedding dim:", dim)

    if dim != EXPECTED_DIM:
        raise RuntimeError(
            f"Model dimension is {dim}, expected {EXPECTED_DIM}. "
            f"Pick a 1024-d model (e.g., intfloat/e5-large-v2)."
        )

    # 3) Encode embeddings (E5 format: passage-prefixed title + description)
    texts = [
        build_passage_text(df.iloc[i]["title"], df.iloc[i]["description"])
        for i in range(records_count)
    ]

    print("Encoding embeddings...")
    t_emb0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        # normalize_embeddings=True,  # optional; keep consistent with your other DB tests
    )
    t_emb1 = time.perf_counter()
    embed_seconds = t_emb1 - t_emb0

    emb_lists = [e.tolist() for e in embeddings]

    # 4) Connect Weaviate + recreate collection
    client = connect_weaviate()
    try:
        wait_until_ready(client)

        print("Recreating Weaviate collection...")
        recreate_collection(client, args.distance)

        collection = client.collections.get(COLLECTION_NAME)

        # 5) Insert + time (batch)
        print(f"Inserting into Weaviate (batch_size={BATCH_SIZE})...")
        t_write0 = time.perf_counter()

        for start, end in chunk_ranges(records_count, BATCH_SIZE):
            batch_objects = []
            for i in range(start, end):
                # We store sourceRowId so you can later trace back to CSV row if needed
                props = {
                    "title": df.iloc[i]["title"],
                    "description": df.iloc[i]["description"],
                    "genre": df.iloc[i]["label"],
                    "sourceRowId": int(i),
                }

                batch_objects.append(
                    DataObject(
                        properties=props,
                        vector=emb_lists[i],
                    )
                )

            collection.data.insert_many(batch_objects)

        t_write1 = time.perf_counter()
        write_seconds = t_write1 - t_write0

        # 6) "Index build" timing in Weaviate
        # Weaviate maintains HNSW as data is ingested, so there is no separate CREATE INDEX step like PGVector.
        # We measure "post-insert visibility/settling" until expected object count is visible.
        print("Waiting for Weaviate object count to reach expected size...")
        index_seconds = wait_until_count(client, records_count)

        # 7) Verify inserted row count
        agg = collection.aggregate.over_all(total_count=True)
        db_count = int(agg.total_count or 0)

        # 8) Approx vector memory estimate (float32 only; DB/index overhead not included)
        approx_vec_mb = bytes_to_mb(db_count * EXPECTED_DIM * 4)

        print("\n==== SUMMARY ====")
        print(f"Rows inserted              : {db_count}")
        print(f"CSV size (MB)              : {csv_size_mb:.2f}")
        print(f"Approx embedding data (MB) : {approx_vec_mb:.2f} (float32 vectors only)")
        print(f"Embedding time (s)         : {embed_seconds:.3f}")
        print(f"DB write time (s)          : {write_seconds:.3f}")
        print(f'Index build time (s)       : {index_seconds:.3f} (count visibility/settling proxy)')
        print("=================\n")

    finally:
        client.close()


if __name__ == "__main__":
    main()
