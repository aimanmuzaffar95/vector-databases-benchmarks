#!/usr/bin/env python3
import os
import time
import pandas as pd
import wget

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

CSV_URL = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/AG_news_samples.csv"
CSV_PATH = "AG_news_samples.csv"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "news_articles"
DISTANCE = qm.Distance.COSINE

# Model / dim
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
EXPECTED_DIM = 1024

# Insert batching
BATCH_SIZE = 128

# Polling config for "ready/optimized after insert"
POLL_INTERVAL_SEC = 0.5
MAX_WAIT_SEC = 60 * 30  # 30 minutes safety cap

def wait_until_collection_available(client: QdrantClient, collection: str, timeout_sec: int = 60) -> None:
    """Wait until Qdrant responds for the collection (basic readiness)."""
    deadline = time.time() + timeout_sec
    while True:
        try:
            client.get_collection(collection)
            return
        except Exception:
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for collection to become available.")
            time.sleep(0.2)



def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def chunk_ranges(total: int, batch: int):
    for i in range(0, total, batch):
        yield i, min(i + batch, total)


def wait_until_optimized(client: QdrantClient, collection: str) -> float:
    """
    Measures time until Qdrant reports optimizer status OK after inserts.
    We start the timer when this function is called and stop when optimized.
    """
    start = time.perf_counter()
    deadline = start + MAX_WAIT_SEC

    while True:
        info = client.get_collection(collection)
        # Qdrant exposes optimizer status in collection_info
        # In python client, it's typically at: info.optimizer_status
        status = getattr(info, "optimizer_status", None)

        # Be defensive: if structure differs, try dict-like fallback
        if status is None and hasattr(info, "dict"):
            d = info.dict()
            status = d.get("optimizer_status")

        # "ok" means optimized/ready
        if status == "ok":
            return time.perf_counter() - start

        if time.perf_counter() > deadline:
            raise TimeoutError(
                f"Timed out waiting for collection optimization (last optimizer_status={status}). "
                f"Increase MAX_WAIT_SEC if needed."
            )

        time.sleep(POLL_INTERVAL_SEC)


def main():
    print("Qdrant:", QDRANT_URL)
    print("Collection:", COLLECTION_NAME)
    print("Distance:", "Cosine")
    print("Model:", MODEL_NAME)

    # 1) Download CSV
    if not os.path.exists(CSV_PATH):
        print(f"Downloading: {CSV_URL}")
        wget.download(CSV_URL, CSV_PATH)
        print("\nDownloaded.")
    else:
        print(f"CSV already exists: {CSV_PATH}")

    csv_size_mb = bytes_to_mb(os.path.getsize(CSV_PATH))

    # 2) Load data
    df = pd.read_csv(CSV_PATH)
    for col in ["title", "description", "label"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Found: {list(df.columns)}")

    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["label"] = df["label"].fillna("")

    records_count = len(df)
    print(f"Records: {records_count} | CSV size: {csv_size_mb:.2f} MB")

    # 3) Load model + verify dim
    print("Loading sentence-transformers model...")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print("Detected embedding dim:", dim)

    if dim != EXPECTED_DIM:
        raise RuntimeError(
            f"Model dimension is {dim}, expected {EXPECTED_DIM}. "
            f"Pick a 1024-d model (e.g., intfloat/e5-large-v2)."
        )

    # 4) Encode embeddings
    texts = df["description"].tolist()
    print("Encoding embeddings...")
    t_emb0 = time.perf_counter()
    embeddings = model.encode(texts, show_progress_bar=True)
    t_emb1 = time.perf_counter()
    embed_seconds = t_emb1 - t_emb0

    vectors = [v.tolist() for v in embeddings]

    # 5) Connect Qdrant
    client = QdrantClient(url=QDRANT_URL)

    # 6) Recreate collection each run (as requested)
    if client.collection_exists(COLLECTION_NAME):
        print("Deleting existing collection...")
        client.delete_collection(COLLECTION_NAME)

    print("Creating collection...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qm.VectorParams(size=EXPECTED_DIM, distance=DISTANCE),
    )

    # 7) Upsert + time to write
    print(f"Upserting into Qdrant (batch_size={BATCH_SIZE})...")
    t_write0 = time.perf_counter()

    for start, end in chunk_ranges(records_count, BATCH_SIZE):
        ids = list(range(start, end))  # stable numeric IDs for this dataset

        payloads = [
            {
                "title": df.iloc[i]["title"],
                "description": df.iloc[i]["description"],
                "genre": df.iloc[i]["label"],
            }
            for i in range(start, end)
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=qm.Batch(
                ids=ids,
                vectors=vectors[start:end],
                payloads=payloads,
            ),
        )

    t_write1 = time.perf_counter()
    write_seconds = t_write1 - t_write0

    # Ensure all writes are committed/visible before optimization timing
    # client.wait_collection_green(COLLECTION_NAME)
    wait_until_collection_available(client, COLLECTION_NAME)


    # 8) Index/build time definition: time until optimized/ready after insert
    print("Waiting for collection to be optimized/ready (timed)...")
    index_seconds = wait_until_optimized(client, COLLECTION_NAME)

    # 9) Verify count
    info = client.get_collection(COLLECTION_NAME)
    points_count = info.points_count or 0

    # 10) Size estimate of embeddings in MB (float32)
    approx_vec_mb = bytes_to_mb(points_count * EXPECTED_DIM * 4)

    print("\n==== SUMMARY ====")
    print(f"Points inserted            : {points_count}")
    print(f"CSV size (MB)              : {csv_size_mb:.2f}")
    print(f"Approx embedding data (MB) : {approx_vec_mb:.2f} (float32 vectors only)")
    print(f"Embedding time (s)         : {embed_seconds:.3f}")
    print(f"Write time (s)             : {write_seconds:.3f}")
    print(f"Index ready/optimized (s)  : {index_seconds:.3f}")
    print("=================\n")


if __name__ == "__main__":
    main()
