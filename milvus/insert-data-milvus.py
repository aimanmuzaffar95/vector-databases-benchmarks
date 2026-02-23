#!/usr/bin/env python3
import os
import time
import argparse
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

CSV_PATH_DEFAULT = "train.csv"

LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

DEFAULT_TABLE_NAME = "news_articles"
DEFAULT_INDEX_NAME = "news_articles_embedding_hnsw"

DEFAULT_MODEL_NAME = "intfloat/e5-large-v2"
EXPECTED_DIM = 1024

POLL_INTERVAL_SEC = 0.5
MAX_WAIT_SEC = 60 * 20  # 20 min safety cap


def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def build_passage_text(title: str, description: str) -> str:
    combined = f"{str(title).strip()} {str(description).strip()}".strip()
    return f"passage: {combined}"


def truncate_str(x: str, max_len: int) -> str:
    s = "" if x is None else str(x)
    return s[:max_len]


def normalize_metric_type(metric: str) -> str:
    """
    Normalize aliases to Milvus metric names.
    Accepted: cosine, ip, dot, l2, euclid, euclidean
    Returns one of: COSINE, IP, L2
    """
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

    # CLI-first (env vars used only as defaults)
    parser.add_argument("--csv-path", default=os.getenv("CSV_PATH", CSV_PATH_DEFAULT))
    parser.add_argument("--host", default=os.getenv("MILVUS_HOST", "localhost"))
    parser.add_argument("--port", default=os.getenv("MILVUS_PORT", "19530"))
    parser.add_argument("--collection", default=os.getenv("MILVUS_COLLECTION", DEFAULT_TABLE_NAME))
    parser.add_argument("--index-name", default=os.getenv("MILVUS_INDEX_NAME", DEFAULT_INDEX_NAME))

    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME))
    parser.add_argument(
        "--distance",
        default=os.getenv("MILVUS_DISTANCE", os.getenv("MILVUS_METRIC", "cosine")),
        help="Distance/metric for Milvus index: cosine | ip | dot | l2",
    )

    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "1000")))

    # safer behavior: explicit recreate toggle
    parser.add_argument(
        "--recreate",
        choices=["true", "false"],
        default=os.getenv("RECREATE_COLLECTION", "true").lower(),
        help="Whether to drop and recreate the collection (default: true)",
    )

    # HNSW params (optional tuning)
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

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Local dataset not found: {csv_path}")
    print(f"Using local dataset: {csv_path}")

    csv_size_mb = bytes_to_mb(os.path.getsize(csv_path))

    # 1) Load data
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

    # 2) Load embedding model
    print("Loading sentence-transformers model...")
    model = SentenceTransformer(model_name)

    dim = model.get_sentence_embedding_dimension()
    print("Detected embedding dim:", dim)
    if dim != EXPECTED_DIM:
        raise RuntimeError(
            f"Model dimension is {dim}, expected {EXPECTED_DIM}. "
            f"Pick a 1024-d model (e.g., intfloat/e5-large-v2)."
        )

    # 3) Encode embeddings (E5 passage format)
    texts = [build_passage_text(t, d) for t, d in zip(df["title"], df["description"])]

    print("Encoding embeddings...")
    t_emb0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    t_emb1 = time.perf_counter()
    embed_seconds = t_emb1 - t_emb0

    if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_DIM:
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")

    emb_lists = embeddings.tolist()

    # 4) Prepare scalar columns
    ids = list(range(records_count))  # stable numeric IDs for benchmarking
    titles = [truncate_str(x, 1024) for x in df["title"].tolist()]
    descriptions = [truncate_str(x, 8192) for x in df["description"].tolist()]
    genres = [truncate_str(x, 64) for x in df["label"].tolist()]

    # 5) Connect Milvus
    connections.connect(alias="default", host=host, port=port)

    # 6) Recreate or protect existing collection
    if recreate_collection:
        cleanup_collection(table_name)
    else:
        if utility.has_collection(table_name):
            raise RuntimeError(
                f"Collection '{table_name}' already exists. "
                "Use --recreate true to overwrite or choose a different --collection."
            )

    # 7) Create collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="genre", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EXPECTED_DIM),
    ]
    schema = CollectionSchema(fields=fields, description="News articles with embeddings")

    collection = Collection(name=table_name, schema=schema, using="default", shards_num=2)

    # 8) Insert + time
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

    t_write1 = time.perf_counter()
    write_seconds = t_write1 - t_write0

    # 9) Build HNSW index + time
    print("Building HNSW index...")
    t_idx0 = time.perf_counter()

    index_params = {
        "index_type": "HNSW",
        "metric_type": metric_type,  # COSINE / IP / L2
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

    # Verify created index metadata (helps avoid metric mismatch confusion)
    try:
        print("Index metadata after creation:")
        for idx in collection.indexes:
            print(" -", getattr(idx, "params", None))
    except Exception as e:
        print("Could not inspect index metadata:", e)

    # Load collection into memory for later search benchmarking
    collection.load()

    t_idx1 = time.perf_counter()
    index_seconds = t_idx1 - t_idx0

    # 10) Summary
    db_count = int(collection.num_entities)
    approx_vec_mb = bytes_to_mb(db_count * EXPECTED_DIM * 4)

    print("\n==== SUMMARY ====")
    print(f"Rows inserted              : {db_count}")
    print(f"CSV size (MB)              : {csv_size_mb:.2f}")
    print(f"Approx embedding data (MB) : {approx_vec_mb:.2f} (float32 vectors only)")
    print(f"Embedding time (s)         : {embed_seconds:.3f}")
    print(f"DB write time (s)          : {write_seconds:.3f}")
    print(f"Index build time (s)       : {index_seconds:.3f}")
    print("=================\n")

    connections.disconnect("default")


if __name__ == "__main__":
    main()