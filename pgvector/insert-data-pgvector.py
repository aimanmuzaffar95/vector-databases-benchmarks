#!/usr/bin/env python3
import os
import time
import pandas as pd
import sqlalchemy as sa

from sentence_transformers import SentenceTransformer

CSV_PATH = "train.csv"

LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/appdb")

TABLE_NAME = "news_articles"
INDEX_NAME = "news_articles_embedding_hnsw"

# Pick a 1024-d model
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
EXPECTED_DIM = 1024

def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)

def build_passage_text(title: str, description: str) -> str:
    combined = f"{title.strip()} {description.strip()}".strip()
    return f"passage: {combined}"


def cleanup_table(engine: sa.Engine) -> None:
    """Remove the existing table/index so every run starts clean."""
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP INDEX IF EXISTS {INDEX_NAME};"))
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {TABLE_NAME};"))

def main():
    print("DB:", DATABASE_URL)
    print("Model:", MODEL_NAME)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Local dataset not found: {CSV_PATH}")
    print(f"Using local dataset: {CSV_PATH}")

    csv_size_mb = bytes_to_mb(os.path.getsize(CSV_PATH))

    # 2) Load data
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

    # Normalize label names so downstream code still uses text genres.
    df["label"] = df["label"].map(LABEL_MAP).fillna("Unknown")
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

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

    # 5) Connect DB
    engine = sa.create_engine(DATABASE_URL, future=True)

    # 6) Schema
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))

    cleanup_table(engine)

    with engine.begin() as conn:
        conn.execute(sa.text(f"""
            CREATE TABLE {TABLE_NAME} (
                id BIGSERIAL PRIMARY KEY,
                title TEXT,
                description TEXT,
                genre TEXT,
                embedding vector({EXPECTED_DIM})
            );
        """))

    # 7) Insert + time
    insert_stmt = sa.text(f"""
        INSERT INTO {TABLE_NAME} (title, description, genre, embedding)
        VALUES (:title, :description, :genre, :embedding)
    """)

    print("Inserting into Postgres...")
    t_write0 = time.perf_counter()
    with engine.begin() as conn:
        conn.execute(insert_stmt, payload)
    t_write1 = time.perf_counter()
    write_seconds = t_write1 - t_write0

    # 8) Build index + time (HNSW cosine)
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP INDEX IF EXISTS {INDEX_NAME};"))

    print("Building HNSW index...")
    t_idx0 = time.perf_counter()
    with engine.begin() as conn:
        conn.execute(sa.text(f"""
            CREATE INDEX {INDEX_NAME}
            ON {TABLE_NAME}
            USING hnsw (embedding vector_cosine_ops);
        """))
    t_idx1 = time.perf_counter()
    index_seconds = t_idx1 - t_idx0

    # 9) Verify inserted row count
    with engine.connect() as conn:
        db_count = conn.execute(sa.text(f"SELECT COUNT(*) FROM {TABLE_NAME};")).scalar_one()

    # 10) Size estimate of embeddings in MB (float32)
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
