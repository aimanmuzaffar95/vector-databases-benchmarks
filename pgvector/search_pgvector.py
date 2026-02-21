#!/usr/bin/env python3
"""Search the `news_articles` table in Postgres/PGVector."""

from __future__ import annotations

import os
from pathlib import Path

import sqlalchemy as sa
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/appdb")
TABLE_NAME = os.getenv("TABLE_NAME", "news_articles")
EMBEDDING_COLUMN = os.getenv("EMBEDDING_COLUMN", "embedding")

# Edit these values before running.
MODEL_NAME = DEFAULT_MODEL
# QUERY_TEXT = "Fed minutes show dissent over inflation. Retail sales bounced back in July and jobless claims fell, indicating the economy is improving from a midsummer slump."
QUERY_TEXT = "Federal Reserve meeting notes show disagreement on inflation as retail sales and jobless claims suggest economic recovery."
QUERY_FILE: str | None = None
TOP_K = 5
GENRE_FILTER: str | None = None


def load_model(model_name: str) -> SentenceTransformer:
    print("Loading embedding model:", model_name)
    return SentenceTransformer(model_name)


def build_query_text(query_text: str, query_file: str | None) -> str:
    if query_file:
        return Path(query_file).read_text(encoding="utf-8").strip()
    return query_text

def build_query_input(query_text: str) -> str:
    return f"query: {query_text.strip()}"


def search(engine: sa.Engine, vector: list[float], limit: int, genre: str | None) -> list[dict[str, object]]:
    filter_clause = "" if genre is None else "AND genre = :genre"
    stmt = sa.text(f"""
        SELECT title, description, genre,
               1 - ( {EMBEDDING_COLUMN} <=> CAST(:vector AS vector) ) AS similarity
        FROM {TABLE_NAME}
        WHERE 1=1 {filter_clause}
        ORDER BY {EMBEDDING_COLUMN} <=> CAST(:vector AS vector)
        LIMIT :limit
    """)
    vector_literal = "[" + ",".join(str(value) for value in vector) + "]"
    params = {"vector": vector_literal, "limit": limit}
    if genre:
        params["genre"] = genre
    with engine.connect() as conn:
        rows = conn.execute(stmt, params).mappings().all()
    return [dict(row) for row in rows]


def main() -> None:
    if TOP_K <= 0:
        raise ValueError(f"TOP_K must be > 0. Received: {TOP_K}")

    query_text = build_query_text(QUERY_TEXT, QUERY_FILE)
    encoded_query_text = build_query_input(query_text)
    model = load_model(MODEL_NAME)
    vector = model.encode(encoded_query_text).tolist()

    engine = sa.create_engine(DATABASE_URL, future=True)
    results = search(engine, vector, TOP_K, GENRE_FILTER)

    print(f"\nQuery: {query_text}")
    print(f"Model: {MODEL_NAME} | K={TOP_K} | Genre filter: {GENRE_FILTER or 'none'}\n")
    if not results:
        print("No results returned.")
        return
    for i, row in enumerate(results, 1):
        print(f"{i}. [{row['genre']}] {row['title']} (score={row['similarity']:.4f})")
        print(f"   {row['description']}\n")


if __name__ == "__main__":
    main()
