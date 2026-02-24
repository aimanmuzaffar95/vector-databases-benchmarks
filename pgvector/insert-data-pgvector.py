#!/usr/bin/env python3
import argparse
import os
import time
import pandas as pd
import sqlalchemy as sa

from sentence_transformers import SentenceTransformer

CSV_PATH_DEFAULT = os.getenv("CSV_PATH", "train.csv")

LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/appdb")
COLLECTION_NAME_DEFAULT = os.getenv("COLLECTION_NAME", "news_articles")

# Pick a 1024-d model
MODEL_NAME_DEFAULT = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
EXPECTED_DIM = 1024
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE", "1000"))
DISTANCE_DEFAULT = os.getenv("PGVECTOR_DISTANCE", "cosine").strip().lower()

def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)

def build_passage_text(title: str, description: str) -> str:
    combined = f"{title.strip()} {description.strip()}".strip()
    return f"passage: {combined}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insert CSV news embeddings into pgvector")
    parser.add_argument("--csv-path", default=CSV_PATH_DEFAULT)
    parser.add_argument("--model", default=MODEL_NAME_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--collection", default=COLLECTION_NAME_DEFAULT)
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
    """Remove the existing table/index so every run starts clean."""
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

    # Normalize label names so downstream code still uses text genres.
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

    # Convert vectors to lists for DB binding
    emb_lists = [e.tolist() for e in embeddings]

    # Prepare records
    payload = [
        {"title": df.iloc[i]["title"],
         "description": df.iloc[i]["description"],
         "genre": df.iloc[i]["label"],
         "embedding": emb_lists[i]}
        for i in range(records_count)
    ]
    engine = sa.create_engine(DATABASE_URL, future=True)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))

    cleanup_table(engine, table_name, index_name)

    with engine.begin() as conn:
        conn.execute(sa.text(f"""
            CREATE TABLE {table_name} (
                id BIGSERIAL PRIMARY KEY,
                title TEXT,
                description TEXT,
                genre TEXT,
                embedding vector({EXPECTED_DIM})
            );
        """))
    insert_stmt = sa.text(f"""
        INSERT INTO {table_name} (title, description, genre, embedding)
        VALUES (:title, :description, :genre, :embedding)
    """)

    print("Inserting into Postgres...")
    t_write0 = time.perf_counter()
    with engine.begin() as conn:
        for start in range(0, records_count, batch_size):
            end = min(start + batch_size, records_count)
            conn.execute(insert_stmt, payload[start:end])
    t_write1 = time.perf_counter()
    write_seconds = t_write1 - t_write0
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP INDEX IF EXISTS {index_name};"))

    print("Building HNSW index...")
    t_idx0 = time.perf_counter()
    with engine.begin() as conn:
        conn.execute(sa.text(f"""
            CREATE INDEX {index_name}
            ON {table_name}
            USING hnsw (embedding {distance_ops});
        """))
    t_idx1 = time.perf_counter()
    index_seconds = t_idx1 - t_idx0
    with engine.connect() as conn:
        db_count = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table_name};")).scalar_one()
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
