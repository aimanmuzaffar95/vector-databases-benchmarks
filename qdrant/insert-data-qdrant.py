#!/usr/bin/env python3
import argparse
import os
import time
import pandas as pd

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

CSV_PATH_DEFAULT = os.getenv("CSV_PATH", "train.csv")

LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME_DEFAULT = os.getenv("COLLECTION_NAME", "news_articles")
DISTANCE_DEFAULT = os.getenv("QDRANT_DISTANCE", "cosine").strip().lower()

# Model / dim
MODEL_NAME_DEFAULT = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
EXPECTED_DIM = 1024

# Insert batching
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE", "128"))

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


def build_passage_text(title: str, description: str) -> str:
    combined = f"{title.strip()} {description.strip()}".strip()
    return f"passage: {combined}"


def chunk_ranges(total: int, batch: int):
    for i in range(0, total, batch):
        yield i, min(i + batch, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insert CSV news embeddings into Qdrant")
    parser.add_argument("--csv-path", default=CSV_PATH_DEFAULT)
    parser.add_argument("--model", default=MODEL_NAME_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--collection", default=COLLECTION_NAME_DEFAULT)
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

    # 1) Load local CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Local dataset not found: {csv_path}")
    print(f"Using local dataset: {csv_path}")

    csv_size_mb = bytes_to_mb(os.path.getsize(csv_path))

    # 2) Load data
    df = pd.read_csv(csv_path)
    expected_columns = {"Class Index", "Title", "Description"}
    received_columns = set(df.columns)
    if not expected_columns.issubset(received_columns):
        raise ValueError(f"Missing column(s). Expected: {expected_columns}, Found: {received_columns}")

    df = df.rename(columns={
        "Class Index": "label",
        "Title": "title",
        "Description": "description",
    })

    df["label"] = df["label"].map(LABEL_MAP).fillna("Unknown")
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

    records_count = len(df)
    print(f"Records: {records_count} | CSV size: {csv_size_mb:.2f} MB")

    # 3) Load model + verify dim
    print("Loading sentence-transformers model...")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print("Detected embedding dim:", dim)

    if dim != EXPECTED_DIM:
        raise RuntimeError(
            f"Model dimension is {dim}, expected {EXPECTED_DIM}. "
            f"Pick a 1024-d model (e.g., intfloat/e5-large-v2)."
        )

    # 4) Encode embeddings (E5 format: passage-prefixed title + description)
    texts = [
        build_passage_text(df.iloc[i]["title"], df.iloc[i]["description"])
        for i in range(records_count)
    ]
    print("Encoding embeddings...")
    t_emb0 = time.perf_counter()
    embeddings = model.encode(texts, show_progress_bar=True)
    t_emb1 = time.perf_counter()
    embed_seconds = t_emb1 - t_emb0

    vectors = [v.tolist() for v in embeddings]

    # 5) Connect Qdrant
    client = QdrantClient(url=QDRANT_URL)

    # 6) Recreate collection each run (as requested)
    if client.collection_exists(collection_name):
        print("Deleting existing collection...")
        client.delete_collection(collection_name)

    print("Creating collection...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(size=EXPECTED_DIM, distance=distance_enum),
    )

    # 7) Upsert + time to write
    print(f"Inserting into Qdrant (batch_size={batch_size})...")
    t_write0 = time.perf_counter()

    for start, end in chunk_ranges(records_count, batch_size):
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
            collection_name=collection_name,
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
    wait_until_collection_available(client, collection_name)


    # 8) Index/build time definition: time until optimized/ready after insert
    print("Building Qdrant index...")
    index_seconds = wait_until_optimized(client, collection_name)

    # 9) Verify count
    info = client.get_collection(collection_name)
    points_count = info.points_count or 0

    # 10) Size estimate of embeddings in MB (float32)
    approx_vec_mb = bytes_to_mb(points_count * EXPECTED_DIM * 4)

    print("\n==== SUMMARY ====")
    print(f"Rows inserted              : {points_count}")
    print(f"CSV size (MB)              : {csv_size_mb:.2f}")
    print(f"Approx embedding data (MB) : {approx_vec_mb:.2f} (float32 vectors only)")
    print(f"Embedding time (s)         : {embed_seconds:.3f}")
    print(f"DB write time (s)          : {write_seconds:.3f}")
    print(f"Index build time (s)       : {index_seconds:.3f}")
    print("=================\n")


if __name__ == "__main__":
    main()
