#!/usr/bin/env python3
"""
Benchmark Recall@K for a pgvector-backed table.

What it does:
1) Loads ids + embeddings from Postgres
2) Samples query rows from the same dataset (with fixed seed)
3) Builds exact ground truth top-K in NumPy (excluding self-match)
4) Runs pgvector search for each query vector
5) Computes Recall@K + latency stats (p50/p95/avg)
6) Optionally sweeps ANN runtime params (ivfflat.probes / hnsw.ef_search)

Usage examples:
  python3 benchmark_recall_pgvector.py
  python3 benchmark_recall_pgvector.py --k-values 1,5,10 --num-queries 500
  python3 benchmark_recall_pgvector.py --distance cosine --ivfflat-probes 1,5,10,20,50
  python3 benchmark_recall_pgvector.py --distance cosine --hnsw-ef-search 20,40,80,120
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import sqlalchemy as sa


# -----------------------------
# Config (overridable via env)
# -----------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/appdb",
)
TABLE_NAME = os.getenv("TABLE_NAME", "news_articles")
ID_COLUMN = os.getenv("ID_COLUMN", "id")
EMBEDDING_COLUMN = os.getenv("EMBEDDING_COLUMN", "embedding")
# Optional filter column if you want benchmark under a predicate:
GENRE_COLUMN = os.getenv("GENRE_COLUMN", "genre")


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(name: str, label: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid {label}: {name!r}")
    return name


@dataclass
class RowVec:
    row_id: int
    vector: np.ndarray


@dataclass
class SearchResult:
    ids: list[int]
    latency_ms: float


@dataclass
class BenchmarkRun:
    label: str
    metric: str
    k_values: list[int]
    num_queries: int
    avg_recall_at_k: dict[int, float]
    p50_latency_ms: float
    p95_latency_ms: float
    avg_latency_ms: float


# -----------------------------
# Metric helpers
# -----------------------------
def parse_k_values(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k <= 0:
            raise ValueError(f"K must be > 0, got {k}")
        values.append(k)
    if not values:
        raise ValueError("No k-values provided")
    # de-dup + sort
    return sorted(set(values))


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def percentile(values: list[float], p: float) -> float:
    """Simple percentile using numpy for convenience."""
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def recall_at_k(pred_ids: list[int], gt_ids: list[int], k: int) -> float:
    gt_topk = gt_ids[:k]
    if not gt_topk:
        return 0.0
    pred_topk = pred_ids[:k]
    hit_count = len(set(pred_topk) & set(gt_topk))
    return hit_count / len(gt_topk)


def metric_to_pgvector_operator(metric: str) -> str:
    # pgvector operators:
    # <->  L2 distance
    # <=>  cosine distance
    # <#>  (negative) inner product distance-like operator ordering by ascending gives highest IP first
    metric = metric.lower()
    if metric == "cosine":
        return "<=>"
    if metric == "l2":
        return "<->"
    if metric in {"ip", "inner_product", "inner-product"}:
        return "<#>"
    raise ValueError(f"Unsupported metric: {metric}")


# -----------------------------
# DB loading and search
# -----------------------------
def create_engine(url: str) -> sa.Engine:
    return sa.create_engine(url, future=True, pool_pre_ping=True)


def load_all_vectors(
    engine: sa.Engine,
    table_name: str,
    id_column: str,
    embedding_column: str,
    *,
    where_sql: str | None = None,
    where_params: dict[str, Any] | None = None,
    limit: int | None = None,
) -> list[RowVec]:
    """
    Load all ids + embeddings from Postgres.
    embeddings are cast to text and parsed as '[...]'
    """
    table_name = validate_identifier(table_name, "table name")
    id_column = validate_identifier(id_column, "id column")
    embedding_column = validate_identifier(embedding_column, "embedding column")

    sql = f"""
        SELECT {id_column} AS id,
               {embedding_column}::text AS embedding_text
        FROM {table_name}
    """
    if where_sql:
        sql += f" WHERE {where_sql} "
    sql += f" ORDER BY {id_column} "
    if limit is not None:
        sql += " LIMIT :_limit "

    params = dict(where_params or {})
    if limit is not None:
        params["_limit"] = int(limit)

    rows_out: list[RowVec] = []
    with engine.connect() as conn:
        rows = conn.execute(sa.text(sql), params).mappings().all()

    for row in rows:
        row_id = int(row["id"])
        emb_text = row["embedding_text"]
        vec = parse_pgvector_text(emb_text)
        rows_out.append(RowVec(row_id=row_id, vector=vec))

    if not rows_out:
        raise RuntimeError("No vectors loaded from database.")
    return rows_out


def parse_pgvector_text(text: str) -> np.ndarray:
    """
    Parse pgvector text representation like '[0.1,0.2,-0.3]'.
    """
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"Unexpected vector text format (truncated): {s[:80]!r}")
    body = s[1:-1].strip()
    if not body:
        return np.array([], dtype=np.float32)
    arr = np.fromstring(body, sep=",", dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Failed to parse vector text into floats.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Vector contains NaN/Inf.")
    return arr


def vector_to_pgvector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in vector.tolist()) + "]"


def search_pgvector(
    conn: sa.Connection,
    *,
    table_name: str,
    id_column: str,
    embedding_column: str,
    query_vector: np.ndarray,
    limit: int,
    metric: str,
    exclude_id: int | None = None,
    ivfflat_probes: int | None = None,
    hnsw_ef_search: int | None = None,
) -> SearchResult:
    table_name = validate_identifier(table_name, "table name")
    id_column = validate_identifier(id_column, "id column")
    embedding_column = validate_identifier(embedding_column, "embedding column")
    op = metric_to_pgvector_operator(metric)

    where_parts = []
    params: dict[str, Any] = {"limit": int(limit), "vector": vector_to_pgvector_literal(query_vector)}
    if exclude_id is not None:
        where_parts.append(f"{id_column} != :exclude_id")
        params["exclude_id"] = int(exclude_id)

    where_clause = ""
    if where_parts:
        where_clause = "WHERE " + " AND ".join(where_parts)

    # Return IDs only for benchmark (faster, cleaner).
    stmt = sa.text(f"""
        SELECT {id_column} AS id
        FROM {table_name}
        {where_clause}
        ORDER BY {embedding_column} {op} CAST(:vector AS vector)
        LIMIT :limit
    """)

    if ivfflat_probes is not None and hnsw_ef_search is not None:
        raise ValueError("Set only one of ivfflat_probes or hnsw_ef_search")

    if ivfflat_probes is not None:
        conn.execute(sa.text("SET LOCAL ivfflat.probes = :p"), {"p": int(ivfflat_probes)})
    if hnsw_ef_search is not None:
        conn.execute(sa.text("SET LOCAL hnsw.ef_search = :e"), {"e": int(hnsw_ef_search)})

    t0 = time.perf_counter()
    rows = conn.execute(stmt, params).mappings().all()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    ids = [int(r["id"]) for r in rows]
    return SearchResult(ids=ids, latency_ms=latency_ms)


# -----------------------------
# Exact ground truth (NumPy)
# -----------------------------
def maybe_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def build_matrix(rows: list[RowVec]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array([r.row_id for r in rows], dtype=np.int64)
    mat = np.stack([r.vector for r in rows]).astype(np.float32, copy=False)
    return ids, mat


def exact_topk_ids(
    *,
    base_ids: np.ndarray,
    base_matrix: np.ndarray,
    query_vector: np.ndarray,
    metric: str,
    k: int,
    exclude_id: int | None = None,
) -> list[int]:
    """
    Compute exact top-k via brute force in NumPy.
    Returns sorted neighbor IDs according to metric ranking.
    """
    metric = metric.lower()
    q = query_vector.astype(np.float32, copy=False)

    # Scores for sorting:
    # - cosine: larger cosine sim is better
    # - ip: larger dot product is better
    # - l2: smaller distance is better
    if metric == "cosine":
        # Assume vectors are already in original DB form. For exact cosine, normalize both at benchmark time.
        qn = q / (np.linalg.norm(q) or 1.0)
        bn = maybe_normalize(base_matrix)
        scores = bn @ qn  # higher is better
        order = np.argsort(-scores)
    elif metric in {"ip", "inner_product", "inner-product"}:
        scores = base_matrix @ q
        order = np.argsort(-scores)
    elif metric == "l2":
        # squared L2 is enough for ranking
        diffs = base_matrix - q
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        order = np.argsort(d2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    result_ids: list[int] = []
    for idx in order:
        rid = int(base_ids[idx])
        if exclude_id is not None and rid == exclude_id:
            continue
        result_ids.append(rid)
        if len(result_ids) >= k:
            break
    return result_ids


# -----------------------------
# Benchmark runner
# -----------------------------
def benchmark_once(
    *,
    engine: sa.Engine,
    rows: list[RowVec],
    metric: str,
    k_values: list[int],
    num_queries: int,
    seed: int,
    warmup_queries: int,
    ivfflat_probes: int | None = None,
    hnsw_ef_search: int | None = None,
) -> BenchmarkRun:
    if not rows:
        raise ValueError("rows cannot be empty")

    max_k = max(k_values)
    if len(rows) <= max_k + 1:
        raise ValueError(f"Need more rows than max_k+1. rows={len(rows)}, max_k={max_k}")

    # Build base matrix from all rows (queries sampled from same table; self-match excluded)
    base_ids, base_matrix = build_matrix(rows)

    # Deterministic query sampling
    rng = random.Random(seed)
    sample_idx = list(range(len(rows)))
    rng.shuffle(sample_idx)
    sample_idx = sample_idx[: min(num_queries, len(rows))]
    if not sample_idx:
        raise ValueError("No query rows selected")

    # Precompute exact GT for all queries first (fair timing separation)
    gt_by_query_id: dict[int, list[int]] = {}
    for i in sample_idx:
        qr = rows[i]
        gt_ids = exact_topk_ids(
            base_ids=base_ids,
            base_matrix=base_matrix,
            query_vector=qr.vector,
            metric=metric,
            k=max_k,
            exclude_id=qr.row_id,
        )
        gt_by_query_id[qr.row_id] = gt_ids

    # Warm-up runs (ignored from metrics)
    warmup_count = min(warmup_queries, len(sample_idx))
    measured_indices = sample_idx[warmup_count:]

    latencies: list[float] = []
    recalls: dict[int, list[float]] = {k: [] for k in k_values}

    with engine.connect() as conn:
        for j, i in enumerate(sample_idx):
            qr = rows[i]
            result = search_pgvector(
                conn,
                table_name=TABLE_NAME,
                id_column=ID_COLUMN,
                embedding_column=EMBEDDING_COLUMN,
                query_vector=qr.vector,
                limit=max_k,
                metric=metric,
                exclude_id=qr.row_id,
                ivfflat_probes=ivfflat_probes,
                hnsw_ef_search=hnsw_ef_search,
            )

            if j < warmup_count:
                continue

            latencies.append(result.latency_ms)
            gt_ids = gt_by_query_id[qr.row_id]
            for k in k_values:
                recalls[k].append(recall_at_k(result.ids, gt_ids, k))

    if not latencies:
        raise RuntimeError("No measured queries after warm-up. Reduce warmup_queries.")

    avg_recall_at_k = {
        k: float(sum(vals) / len(vals)) if vals else float("nan")
        for k, vals in recalls.items()
    }

    label_parts = []
    if ivfflat_probes is not None:
        label_parts.append(f"ivfflat.probes={ivfflat_probes}")
    if hnsw_ef_search is not None:
        label_parts.append(f"hnsw.ef_search={hnsw_ef_search}")
    if not label_parts:
        label_parts.append("default")
    label = ", ".join(label_parts)

    return BenchmarkRun(
        label=label,
        metric=metric,
        k_values=k_values,
        num_queries=len(measured_indices),
        avg_recall_at_k=avg_recall_at_k,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        avg_latency_ms=float(statistics.mean(latencies)),
    )


def print_run(run: BenchmarkRun) -> None:
    print("=" * 80)
    print(f"Run: {run.label}")
    print(f"Distance: {run.metric}")
    print(f"Measured queries: {run.num_queries}")
    for k in run.k_values:
        print(f"Recall@{k}: {run.avg_recall_at_k[k]:.4f}")
    print(f"Latency avg: {run.avg_latency_ms:.2f} ms")
    print(f"Latency p50: {run.p50_latency_ms:.2f} ms")
    print(f"Latency p95: {run.p95_latency_ms:.2f} ms")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark Recall@K for pgvector")
    p.add_argument("--distance", default="cosine", choices=["cosine", "l2", "ip"], help="Distance/similarity metric")
    p.add_argument("--k-values", default="1,5,10", help="Comma-separated K values, e.g. 1,5,10")
    p.add_argument("--num-queries", type=int, default=500, help="Number of query rows to sample")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--warmup-queries", type=int, default=20, help="Warm-up queries excluded from stats")
    p.add_argument("--max-load-items", type=int, default=None, help="Optional cap when loading items (debugging)")
    p.add_argument("--where-sql", type=str, default=None, help="Optional SQL predicate for loading rows, e.g. genre = :g")
    p.add_argument("--where-param", action="append", default=[], help="Optional params for --where-sql as key=value (repeatable)")
    # Runtime ANN sweeps (optional)
    p.add_argument("--ivfflat-probes", type=str, default=None, help="Comma-separated probes values to sweep")
    p.add_argument("--hnsw-ef-search", type=str, default=None, help="Comma-separated ef_search values to sweep")
    return p


def parse_where_params(items: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --where-param {item!r}, expected key=value")
        k, v = item.split("=", 1)
        params[k] = v
    return params


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be > 0")
    if args.warmup_queries < 0:
        raise ValueError("--warmup-queries must be >= 0")

    ivf_sweep = parse_int_list(args.ivfflat_probes)
    hnsw_sweep = parse_int_list(args.hnsw_ef_search)
    if ivf_sweep and hnsw_sweep:
        raise ValueError("Choose either --ivfflat-probes or --hnsw-ef-search, not both")

    engine = create_engine(DATABASE_URL)

    print("Loading vectors from database...")
    rows = load_all_vectors(
        engine,
        TABLE_NAME,
        ID_COLUMN,
        EMBEDDING_COLUMN,
        where_sql=args.where_sql,
        where_params=parse_where_params(args.where_param),
        limit=args.max_load_items,
    )
    dim = int(rows[0].vector.shape[0])
    print(f"Loaded {len(rows)} vectors | dim={dim} | table={TABLE_NAME}")

    # Validate dimensions are consistent
    for r in rows:
        if r.vector.shape[0] != dim:
            raise ValueError(f"Inconsistent vector dims. Row id={r.row_id} has dim={r.vector.shape[0]} expected={dim}")

    runs: list[BenchmarkRun] = []
    if ivf_sweep:
        for probes in ivf_sweep:
            run = benchmark_once(
                engine=engine,
                rows=rows,
                metric=args.distance,
                k_values=k_values,
                num_queries=args.num_queries,
                seed=args.seed,
                warmup_queries=args.warmup_queries,
                ivfflat_probes=probes,
            )
            runs.append(run)
    elif hnsw_sweep:
        for ef in hnsw_sweep:
            run = benchmark_once(
                engine=engine,
                rows=rows,
                metric=args.distance,
                k_values=k_values,
                num_queries=args.num_queries,
                seed=args.seed,
                warmup_queries=args.warmup_queries,
                hnsw_ef_search=ef,
            )
            runs.append(run)
    else:
        run = benchmark_once(
            engine=engine,
            rows=rows,
            metric=args.distance,
            k_values=k_values,
            num_queries=args.num_queries,
            seed=args.seed,
            warmup_queries=args.warmup_queries,
        )
        runs.append(run)

    print("\nBenchmark Results")
    for run in runs:
        print_run(run)

    print("\nDone.")


if __name__ == "__main__":
    main()
