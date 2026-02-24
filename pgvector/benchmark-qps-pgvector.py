"""
Benchmark QPS (Queries Per Second) for a pgvector-backed table.

What it does:
1) Loads ids + embeddings from Postgres (for query sampling)
2) Samples query vectors (fixed seed)
3) Runs a timed throughput test with configurable concurrency
4) Uses pgvector ORDER BY with chosen distance metric
5) Reports QPS + latency stats (avg/p50/p95/p99)
6) Optionally sweeps ANN runtime params (ivfflat.probes / hnsw.ef_search)

Usage examples:
  python3 benchmark-qps-pgvector.py
  python3 benchmark-qps-pgvector.py --distance cosine --k 10 --seconds 10 --concurrency 8
  python3 benchmark-qps-pgvector.py --distance cosine --ivfflat-probes 1,5,10,20
  python3 benchmark-qps-pgvector.py --distance cosine --hnsw-ef-search 20,40,80,120

Notes:
- This measures client-side timing around the DB query (includes network + driver overhead).
- Uses a separate DB connection per worker thread for realistic concurrency.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

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
    distance: str
    k: int
    concurrency: int
    seconds: float
    warmup_seconds: float
    measured_queries: int
    qps: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


# -----------------------------
# Helpers
# -----------------------------
def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def metric_to_pgvector_operator(distance: str) -> str:
    distance = distance.lower()
    if distance == "cosine":
        return "<=>"
    if distance == "l2":
        return "<->"
    if distance in {"ip", "inner_product", "inner-product"}:
        return "<#>"
    raise ValueError(f"Unsupported distance: {distance}")


def create_engine(url: str) -> sa.Engine:
    # pool_pre_ping helps avoid stale connections in long runs
    return sa.create_engine(url, future=True, pool_pre_ping=True)


def parse_pgvector_text(text: str) -> np.ndarray:
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

    with engine.connect() as conn:
        rows = conn.execute(sa.text(sql), params).mappings().all()

    out: list[RowVec] = []
    for r in rows:
        out.append(RowVec(row_id=int(r["id"]), vector=parse_pgvector_text(r["embedding_text"])))

    if not out:
        raise RuntimeError("No vectors loaded from database.")
    return out


def search_pgvector(
    conn: sa.Connection,
    *,
    table_name: str,
    id_column: str,
    embedding_column: str,
    query_vector: np.ndarray,
    limit: int,
    distance: str,
    exclude_id: int | None = None,
    ivfflat_probes: int | None = None,
    hnsw_ef_search: int | None = None,
) -> SearchResult:
    table_name = validate_identifier(table_name, "table name")
    id_column = validate_identifier(id_column, "id column")
    embedding_column = validate_identifier(embedding_column, "embedding column")
    op = metric_to_pgvector_operator(distance)

    where_parts = []
    params: dict[str, Any] = {"limit": int(limit), "vector": vector_to_pgvector_literal(query_vector)}
    if exclude_id is not None:
        where_parts.append(f"{id_column} != :exclude_id")
        params["exclude_id"] = int(exclude_id)

    where_clause = ""
    if where_parts:
        where_clause = "WHERE " + " AND ".join(where_parts)

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


def parse_where_params(items: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --where-param {item!r}, expected key=value")
        k, v = item.split("=", 1)
        params[k] = v
    return params


# -----------------------------
# Throughput runner
# -----------------------------
def worker_loop(
    *,
    engine: sa.Engine,
    queries: list[RowVec],
    k: int,
    distance: str,
    table_name: str,
    id_column: str,
    embedding_column: str,
    stop_time: float,
    warmup_stop_time: float,
    ivfflat_probes: int | None,
    hnsw_ef_search: int | None,
) -> tuple[int, list[float]]:
    """
    Runs queries repeatedly until stop_time.
    Returns (measured_query_count, measured_latencies_ms).
    Warmup queries (before warmup_stop_time) are excluded from stats.
    """
    latencies: list[float] = []
    measured = 0

    # each worker uses its own connection
    with engine.connect() as conn:
        # round-robin through query vectors
        idx = 0
        n = len(queries)
        while True:
            now = time.perf_counter()
            if now >= stop_time:
                break

            qr = queries[idx]
            idx = (idx + 1) % n

            res = search_pgvector(
                conn,
                table_name=table_name,
                id_column=id_column,
                embedding_column=embedding_column,
                query_vector=qr.vector,
                limit=k,
                distance=distance,
                exclude_id=qr.row_id,
                ivfflat_probes=ivfflat_probes,
                hnsw_ef_search=hnsw_ef_search,
            )

            if now >= warmup_stop_time:
                measured += 1
                latencies.append(res.latency_ms)

    return measured, latencies


def run_throughput_test(
    *,
    engine: sa.Engine,
    sampled_queries: list[RowVec],
    k: int,
    distance: str,
    concurrency: int,
    seconds: float,
    warmup_seconds: float,
    ivfflat_probes: int | None,
    hnsw_ef_search: int | None,
) -> tuple[int, float, list[float]]:
    start = time.perf_counter()
    warmup_stop = start + float(warmup_seconds)
    stop = start + float(seconds)

    # distribute queries to workers (each gets same list to avoid bias)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(
                worker_loop,
                engine=engine,
                queries=sampled_queries,
                k=k,
                distance=distance,
                table_name=TABLE_NAME,
                id_column=ID_COLUMN,
                embedding_column=EMBEDDING_COLUMN,
                stop_time=stop,
                warmup_stop_time=warmup_stop,
                ivfflat_probes=ivfflat_probes,
                hnsw_ef_search=hnsw_ef_search,
            )
            for _ in range(concurrency)
        ]

        total_measured = 0
        all_latencies: list[float] = []
        for f in as_completed(futures):
            cnt, lats = f.result()
            total_measured += cnt
            all_latencies.extend(lats)

    measured_window = max(0.001, float(seconds) - float(warmup_seconds))
    qps = total_measured / measured_window
    return total_measured, qps, all_latencies


def print_run(run: BenchmarkRun) -> None:
    print("===============")
    print("Benchmark Results")
    print("===============")
    print(f"Run: {run.label}")
    print(f"Distance: {run.distance}")
    print(f"k: {run.k}")
    print("-------------------------------------")
    print(f"Concurrency: {run.concurrency}")
    print(f"Duration (s): {run.seconds:.2f} (warmup {run.warmup_seconds:.2f}s)")
    print(f"Measured queries: {run.measured_queries}")
    print(f"QPS: {run.qps:.2f}")
    print("-------------------------------------")
    print(f"Latency avg: {run.avg_latency_ms:.2f} ms")
    print(f"Latency p50: {run.p50_latency_ms:.2f} ms")
    print(f"Latency p95: {run.p95_latency_ms:.2f} ms")
    print(f"Latency p99: {run.p99_latency_ms:.2f} ms")
    print("-------------------------------------")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark QPS for pgvector")
    p.add_argument("--distance", default="cosine", choices=["cosine", "l2", "ip"], help="Distance/similarity metric")
    p.add_argument("--k", type=int, default=10, help="Top-k returned per query")
    p.add_argument("--seconds", type=float, default=10.0, help="Total test duration in seconds (includes warmup)")
    p.add_argument("--warmup-seconds", type=float, default=2.0, help="Warm-up duration excluded from stats")
    p.add_argument("--concurrency", type=int, default=8, help="Number of concurrent workers")
    p.add_argument("--query-pool", type=int, default=200, help="How many query vectors to sample (reused by workers)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--max-load-items", type=int, default=None, help="Optional cap when loading items (debugging)")
    p.add_argument("--where-sql", type=str, default=None, help="Optional SQL predicate for loading rows, e.g. genre = :g")
    p.add_argument("--where-param", action="append", default=[], help="Optional params for --where-sql as key=value")

    # Runtime ANN sweeps (optional)
    p.add_argument("--ivfflat-probes", type=str, default=None, help="Comma-separated probes values to sweep")
    p.add_argument("--hnsw-ef-search", type=str, default=None, help="Comma-separated ef_search values to sweep")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.seconds <= 0:
        raise ValueError("--seconds must be > 0")
    if args.warmup_seconds < 0 or args.warmup_seconds >= args.seconds:
        raise ValueError("--warmup-seconds must be >=0 and < --seconds")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if args.query_pool <= 0:
        raise ValueError("--query-pool must be > 0")

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

    # sample query pool
    rng = random.Random(args.seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    idxs = idxs[: min(args.query_pool, len(rows))]
    sampled_queries = [rows[i] for i in idxs]

    runs: list[BenchmarkRun] = []

    def one_run(label: str, ivfflat_probes: int | None, hnsw_ef_search: int | None) -> BenchmarkRun:
        measured, qps, latencies = run_throughput_test(
            engine=engine,
            sampled_queries=sampled_queries,
            k=args.k,
            distance=args.distance,
            concurrency=args.concurrency,
            seconds=args.seconds,
            warmup_seconds=args.warmup_seconds,
            ivfflat_probes=ivfflat_probes,
            hnsw_ef_search=hnsw_ef_search,
        )
        return BenchmarkRun(
            label=label,
            distance=args.distance,
            k=args.k,
            concurrency=args.concurrency,
            seconds=args.seconds,
            warmup_seconds=args.warmup_seconds,
            measured_queries=measured,
            qps=qps,
            avg_latency_ms=float(statistics.mean(latencies)) if latencies else float("nan"),
            p50_latency_ms=percentile(latencies, 50),
            p95_latency_ms=percentile(latencies, 95),
            p99_latency_ms=percentile(latencies, 99),
        )

    if ivf_sweep:
        for probes in ivf_sweep:
            runs.append(one_run(label=f"ivfflat.probes={probes}", ivfflat_probes=probes, hnsw_ef_search=None))
    elif hnsw_sweep:
        for ef in hnsw_sweep:
            runs.append(one_run(label=f"hnsw.ef_search={ef}", ivfflat_probes=None, hnsw_ef_search=ef))
    else:
        runs.append(one_run(label="default", ivfflat_probes=None, hnsw_ef_search=None))

    for r in runs:
        print_run(r)

    print("\nDone.")


if __name__ == "__main__":
    main()
