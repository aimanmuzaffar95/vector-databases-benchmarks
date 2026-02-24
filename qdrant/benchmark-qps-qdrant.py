#!/usr/bin/env python3
"""
Benchmark QPS (Queries Per Second) for a Qdrant collection.

What it does:
1) Loads ids + vectors from Qdrant (for query sampling)
2) Samples query vectors (fixed seed)
3) Runs a timed throughput test with configurable concurrency
4) Uses Qdrant search with chosen distance (collection must be created with that distance)
5) Reports QPS + latency stats (avg/p50/p95/p99)
6) Optionally sweeps ANN runtime param (hnsw_ef via --hnsw-ef-search)

Usage examples:
  python3 benchmark_qps_qdrant.py
  python3 benchmark_qps_qdrant.py --distance cosine --k 10 --seconds 10 --concurrency 8
  python3 benchmark_qps_qdrant.py --distance cosine --hnsw-ef-search 20,40,80,120

Notes:
- Qdrant distance is set at collection creation time. This script uses --distance only for labeling.
- Qdrant doesn't have ivfflat.probes; --ivfflat-probes is accepted but ignored.
- Each worker uses its own QdrantClient instance for realistic concurrency.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
except ImportError as e:
    raise SystemExit("Missing dependency. Install with: pip install qdrant-client") from e


# -----------------------------
# Config (overridable via env)
# -----------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "news_articles")
# Name of vector field in Qdrant ("" or None means default unnamed vector in some setups)
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "embedding").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)


@dataclass
class RowVec:
    row_id: int | str
    vector: np.ndarray


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


def make_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _vector_param_name() -> str | None:
    # If env var is empty, treat as "default vector"
    return QDRANT_VECTOR_NAME if QDRANT_VECTOR_NAME else None


# -----------------------------
# Load vectors from Qdrant (for sampling)
# -----------------------------
def load_all_vectors(
    client: QdrantClient,
    *,
    limit: int | None = None,
) -> list[RowVec]:
    """
    Load ids + vectors from Qdrant using scroll.

    NOTE: Scrolling ALL vectors can be heavy for huge collections.
    Use --max-load-items or limit env if needed.
    """
    vec_name = _vector_param_name()

    rows: list[RowVec] = []
    next_page = None
    remaining = limit if limit is not None else None

    while True:
        page_limit = 256
        if remaining is not None:
            page_limit = min(page_limit, remaining)
            if page_limit <= 0:
                break

        points, next_page = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            limit=page_limit,
            with_payload=False,
            with_vectors=True,
            offset=next_page,
        )

        if not points:
            break

        for p in points:
            pid = p.id  # int or str
            # vectors can be dict (named vectors) or list (single vector)
            vec = None
            if isinstance(p.vector, dict):
                if vec_name is None:
                    # pick the first vector if no name specified
                    vec = next(iter(p.vector.values()))
                else:
                    vec = p.vector.get(vec_name)
            else:
                vec = p.vector

            if vec is None:
                continue

            v = np.asarray(vec, dtype=np.float32)
            if v.size == 0 or not np.all(np.isfinite(v)):
                continue
            rows.append(RowVec(row_id=pid, vector=v))

        if remaining is not None:
            remaining -= len(points)

        if next_page is None:
            break

    if not rows:
        raise RuntimeError("No vectors loaded from Qdrant. Check collection name and vector field name.")
    return rows


# -----------------------------
# Search (Qdrant)
# -----------------------------
def search_qdrant(
    client: QdrantClient,
    *,
    query_vector: np.ndarray,
    limit: int,
    hnsw_ef: int | None = None,
) -> float:
    """
    Perform a Qdrant search and return latency_ms.
    We don't need ids for QPS benchmark; only timing matters.
    """
    vec_name = _vector_param_name()

    params = qm.SearchParams(hnsw_ef=int(hnsw_ef)) if hnsw_ef is not None else None

    def _search_unnamed_vector() -> None:
        if hasattr(client, "query_points"):
            client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector.tolist(),
                limit=int(limit),
                with_payload=False,
                with_vectors=False,
                search_params=params,
            )
            return
        client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=int(limit),
            search_params=params,
            with_payload=False,
            with_vectors=False,
        )

    def _search_named_vector(name: str) -> None:
        if hasattr(client, "query_points"):
            client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector.tolist(),
                using=name,
                limit=int(limit),
                with_payload=False,
                with_vectors=False,
                search_params=params,
            )
            return
        client.search(
            collection_name=COLLECTION_NAME,
            query_vector=(name, query_vector.tolist()),
            limit=int(limit),
            search_params=params,
            with_payload=False,
            with_vectors=False,
        )

    t0 = time.perf_counter()
    if vec_name is None:
        _search_unnamed_vector()
    else:
        try:
            _search_named_vector(vec_name)
        except Exception:
            # Fall back for unnamed-vector collections.
            _search_unnamed_vector()
    return (time.perf_counter() - t0) * 1000.0


# -----------------------------
# Throughput runner
# -----------------------------
def worker_loop(
    *,
    queries: list[RowVec],
    k: int,
    stop_time: float,
    warmup_stop_time: float,
    hnsw_ef: int | None,
) -> tuple[int, list[float]]:
    latencies: list[float] = []
    measured = 0

    client = make_client()
    idx = 0
    n = len(queries)

    while True:
        now = time.perf_counter()
        if now >= stop_time:
            break

        qr = queries[idx]
        idx = (idx + 1) % n

        latency_ms = search_qdrant(
            client,
            query_vector=qr.vector,
            limit=k,
            hnsw_ef=hnsw_ef,
        )

        if now >= warmup_stop_time:
            measured += 1
            latencies.append(latency_ms)

    return measured, latencies


def run_throughput_test(
    *,
    sampled_queries: list[RowVec],
    k: int,
    concurrency: int,
    seconds: float,
    warmup_seconds: float,
    hnsw_ef: int | None,
) -> tuple[int, float, list[float]]:
    start = time.perf_counter()
    warmup_stop = start + float(warmup_seconds)
    stop = start + float(seconds)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(
                worker_loop,
                queries=sampled_queries,
                k=k,
                stop_time=stop,
                warmup_stop_time=warmup_stop,
                hnsw_ef=hnsw_ef,
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
    p = argparse.ArgumentParser(description="Benchmark QPS for Qdrant")
    p.add_argument("--distance", default="cosine", choices=["cosine", "l2", "ip"], help="Label only; Qdrant metric is set at collection creation")
    p.add_argument("--k", type=int, default=10, help="Top-k returned per query")
    p.add_argument("--seconds", type=float, default=10.0, help="Total test duration in seconds (includes warmup)")
    p.add_argument("--warmup-seconds", type=float, default=2.0, help="Warm-up duration excluded from stats")
    p.add_argument("--concurrency", type=int, default=8, help="Number of concurrent workers")
    p.add_argument("--query-pool", type=int, default=200, help="How many query vectors to sample (reused by workers)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--max-load-items", type=int, default=None, help="Optional cap when loading vectors (debugging)")

    # Qdrant tuning
    p.add_argument("--hnsw-ef-search", type=str, default=None, help="Comma-separated hnsw_ef values to sweep")
    # Kept for compatibility with pgvector scripts (ignored)
    p.add_argument("--ivfflat-probes", type=str, default=None, help="Ignored for Qdrant (kept for compatibility)")
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

    hnsw_sweep = parse_int_list(args.hnsw_ef_search)
    ivf_sweep = parse_int_list(args.ivfflat_probes)
    if ivf_sweep:
        print("[WARN] Qdrant does not use ivfflat.probes. --ivfflat-probes is ignored.")

    client = make_client()

    print("Loading vectors from database...")
    rows = load_all_vectors(client, limit=args.max_load_items)
    dim = int(rows[0].vector.shape[0])
    print(f"Loaded {len(rows)} vectors | dim={dim} | collection={COLLECTION_NAME}")

    rng = random.Random(args.seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    idxs = idxs[: min(args.query_pool, len(rows))]
    sampled_queries = [rows[i] for i in idxs]

    runs: list[BenchmarkRun] = []

    def one_run(label: str, hnsw_ef: int | None) -> BenchmarkRun:
        measured, qps, latencies = run_throughput_test(
            sampled_queries=sampled_queries,
            k=args.k,
            concurrency=args.concurrency,
            seconds=args.seconds,
            warmup_seconds=args.warmup_seconds,
            hnsw_ef=hnsw_ef,
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

    if hnsw_sweep:
        for ef in hnsw_sweep:
            runs.append(one_run(label=f"hnsw.ef_search={ef}", hnsw_ef=ef))
    else:
        runs.append(one_run(label="default", hnsw_ef=None))

    for r in runs:
        print_run(r)

    print("\nDone.")


if __name__ == "__main__":
    main()
