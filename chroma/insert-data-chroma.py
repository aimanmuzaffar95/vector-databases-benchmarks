#!/usr/bin/env python3
import argparse
import os
import time
import pandas as pd

from sentence_transformers import SentenceTransformer

import chromadb

CSV_PATH_DEFAULT = os.getenv("CSV_PATH", "train.csv")

LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}
CHROMA_MODE = os.getenv("CHROMA_MODE", "http").strip().lower()  # "http" or "persistent"
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")

COLLECTION_NAME_DEFAULT = os.getenv("COLLECTION_NAME", "news_articles")
MODEL_NAME_DEFAULT = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
EXPECTED_DIM = 1024
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE", "512"))
DISTANCE_DEFAULT = os.getenv("CHROMA_DISTANCE", "cosine").strip().lower()

def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)

def build_passage_text(title: str, description: str) -> str:
    combined = f"{title.strip()} {description.strip()}".strip()
    return f"passage: {combined}"

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
    parser.add_argument(
        "--distance",
        choices=["cosine", "l2", "ip"],
        default=DISTANCE_DEFAULT,
        help="Chroma HNSW space metric for the collection.",
    )
    return parser.parse_args()


def cleanup_collection(client: chromadb.ClientAPI, collection_name: str) -> None:
    """Remove existing collection so every run starts clean."""
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        # If it doesn't exist, ignore
        pass

def main():
    args = parse_args()
    csv_path = args.csv_path
    model_name = args.model
    batch_size = int(args.batch_size)
    collection_name = args.collection
    distance = args.distance

    # Keep output label "DB:" the same
    if CHROMA_MODE == "http":
        db_label = f"chroma://{CHROMA_HOST}:{CHROMA_PORT} (http)"
    else:
        db_label = f"chroma://{os.path.abspath(CHROMA_PERSIST_DIR)} (persistent)"

    print("DB:", db_label)
    print("Model:", model_name)
    print("Distance:", distance)
    print("Collection:", collection_name)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Local dataset not found: {csv_path}")
    print(f"Using local dataset: {csv_path}")

    csv_size_mb = bytes_to_mb(os.path.getsize(csv_path))
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
    print("Loading sentence-transformers model...")
    model = SentenceTransformer(model_name)

    dim = model.get_sentence_embedding_dimension()
    print("Detected embedding dim:", dim)

    if dim != EXPECTED_DIM:
        raise RuntimeError(
            f"Model dimension is {dim}, expected {EXPECTED_DIM}. "
            f"Pick a 1024-d model (e.g., intfloat/e5-large-v2)."
        )
    texts = [
        build_passage_text(df.iloc[i]["title"], df.iloc[i]["description"])
        for i in range(records_count)
    ]

    print("Encoding embeddings...")
    t_emb0 = time.perf_counter()
    embeddings = model.encode(texts, show_progress_bar=True)
    t_emb1 = time.perf_counter()
    embed_seconds = t_emb1 - t_emb0

    # Convert to lists for Chroma
    emb_lists = [e.tolist() for e in embeddings]
    client = get_client()
    cleanup_collection(client, collection_name)

    # Chroma metric is chosen at collection creation time via metadata.
    # Distance is chosen at collection creation time via hnsw:space metadata.
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance},
    )
    print("Inserting into Chroma...")
    t_write0 = time.perf_counter()

    # Chroma ids must be strings; we’ll use "1..N" as stable ids
    # (If you already have stable IDs, replace this with your own id column.)
    for start in range(0, records_count, batch_size):
        end = min(start + batch_size, records_count)

        batch_ids = [str(i + 1) for i in range(start, end)]
        batch_embeddings = emb_lists[start:end]

        batch_metadatas = [
            {
                "title": df.iloc[i]["title"],
                "description": df.iloc[i]["description"],
                "genre": df.iloc[i]["label"],
            }
            for i in range(start, end)
        ]

        # If you also want to store the raw text, you can add `documents=[...]`
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )

    t_write1 = time.perf_counter()
    write_seconds = t_write1 - t_write0
    # Chroma doesn't have an explicit CREATE INDEX step like pgvector.
    # To keep the same output format, we measure time until the first query runs after insertion.
    print("Building HNSW index...")
    t_idx0 = time.perf_counter()
    _ = collection.query(
        query_embeddings=[emb_lists[0]],
        n_results=1,
        include=["distances"],
    )
    t_idx1 = time.perf_counter()
    index_seconds = t_idx1 - t_idx0
    db_count = collection.count()
    approx_vec_mb = bytes_to_mb(db_count * EXPECTED_DIM * 4)

    print("\n==== SUMMARY ====")
    print(f"Rows inserted              : {db_count}")
    print(f"CSV size (MB)              : {csv_size_mb:.2f}")
    print(f"Approx embedding data (MB) : {approx_vec_mb:.2f} (float32 vectors only)")
    print(f"Embedding time (s)         : {embed_seconds:.3f}")
    print(f"DB write time (s)          : {write_seconds:.3f}")
    print(f"Index build time (s)       : {index_seconds:.3f}")
    print("=================\n")

if __name__ == "__main__":
    main()
