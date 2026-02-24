"""
Benchmark QPS (Queries Per Second) for a Weaviate collection.

What it does:
1) Loads ids + vectors from Weaviate (for query sampling)
2) Samples query vectors (fixed seed)
3) Runs a timed throughput test with configurable concurrency
4) Uses Weaviate near_vector search
5) Reports QPS + latency stats (avg/p50/p95/p99)
6) Accepts ANN sweep flags for CLI compatibility
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
import weaviate


WEAVIATE_HTTP_HOST = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "localhost")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_SECURE = os.getenv("WEAVIATE_SECURE", "false").lower() == "true"
COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "NewsArticle")
SOURCE_ID_PROPERTY = os.getenv("WEAVIATE_SOURCE_ID_PROPERTY", "sourceRowId")


@dataclass
class RowVec:
    row_id: int
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


def connect_weaviate() -> weaviate.WeaviateClient:
    timeout_sec = float(os.getenv("WEAVIATE_CONNECT_TIMEOUT_SEC", "90"))
    retry_sleep_sec = float(os.getenv("WEAVIATE_CONNECT_RETRY_SEC", "2"))
    deadline = time.time() + max(timeout_sec, 1.0)
    last_err: Exception | None = None

    while time.time() < deadline:
        try:
            return weaviate.connect_to_custom(
                http_host=WEAVIATE_HTTP_HOST,
                http_port=WEAVIATE_HTTP_PORT,
                http_secure=WEAVIATE_SECURE,
                grpc_host=WEAVIATE_GRPC_HOST,
                grpc_port=WEAVIATE_GRPC_PORT,
                grpc_secure=WEAVIATE_SECURE,
            )
        except Exception as err:
            last_err = err
            time.sleep(max(retry_sleep_sec, 0.2))

    raise RuntimeError(
        f"Could not connect to Weaviate within {timeout_sec:.0f}s at {WEAVIATE_HTTP_HOST}:{WEAVIATE_HTTP_PORT}"
    ) from last_err


def load_all_vectors(client: weaviate.WeaviateClient, collection_name: str, limit: int | None = None) -> list[RowVec]:
    collection = client.collections.get(collection_name)
    rows: list[RowVec] = []

    iterator_obj = None
    try:
        iterator_obj = collection.iterator(include_vector=True)
    except TypeError:
        try:
            iterator_obj = collection.iterator(return_vector=True)
        except TypeError:
            iterator_obj = collection.iterator()

    for obj in iterator_obj:
        props = getattr(obj, "properties", {}) or {}
        if SOURCE_ID_PROPERTY not in props:
            continue

        try:
            row_id = int(props[SOURCE_ID_PROPERTY])
        except Exception:
            continue

        vec = getattr(obj, "vector", None)
        if vec is None:
            continue
        if isinstance(vec, dict):
            vec = vec.get("default", next(iter(vec.values())))

        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1 or not np.all(np.isfinite(arr)):
            continue

        rows.append(RowVec(row_id=row_id, vector=arr))
        if limit is not None and len(rows) >= int(limit):
            break

    if not rows:
        raise RuntimeError("No vectors loaded from Weaviate collection.")

    rows.sort(key=lambda r: r.row_id)
    return rows


def search_weaviate(
    client: weaviate.WeaviateClient,
    *,
    query_vector: np.ndarray,
    limit: int,
) -> float:
    collection = client.collections.get(COLLECTION_NAME)

    t0 = time.perf_counter()
    collection.query.near_vector(
        near_vector=query_vector.tolist(),
        limit=int(limit),
        return_properties=[SOURCE_ID_PROPERTY],
    )
    return (time.perf_counter() - t0) * 1000.0


def worker_loop(
    *,
    queries: list[RowVec],
    k: int,
    stop_time: float,
    warmup_stop_time: float,
) -> tuple[int, list[float]]:
    latencies: list[float] = []
    measured = 0

    client = connect_weaviate()
    idx = 0
    n = len(queries)

    try:
        while True:
            now = time.perf_counter()
            if now >= stop_time:
                break

            qr = queries[idx]
            idx = (idx + 1) % n

            latency_ms = search_weaviate(
                client,
                query_vector=qr.vector,
                limit=k,
            )

            if now >= warmup_stop_time:
                measured += 1
                latencies.append(latency_ms)
    finally:
        client.close()

    return measured, latencies


def run_throughput_test(
    *,
    sampled_queries: list[RowVec],
    k: int,
    concurrency: int,
    seconds: float,
    warmup_seconds: float,
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
    p = argparse.ArgumentParser(description="Benchmark QPS for Weaviate")
    p.add_argument("--distance", default="cosine", choices=["cosine", "l2", "dot"], help="Label only; Weaviate metric is set at collection creation")
    p.add_argument("--k", type=int, default=10, help="Top-k returned per query")
    p.add_argument("--seconds", type=float, default=10.0, help="Total test duration in seconds (includes warmup)")
    p.add_argument("--warmup-seconds", type=float, default=2.0, help="Warm-up duration excluded from stats")
    p.add_argument("--concurrency", type=int, default=8, help="Number of concurrent workers")
    p.add_argument("--query-pool", type=int, default=200, help="How many query vectors to sample (reused by workers)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--max-load-items", type=int, default=None, help="Optional cap when loading vectors (debugging)")
    p.add_argument("--hnsw-ef-search", type=str, default=None, help="Ignored for Weaviate (kept for compatibility)")
    p.add_argument("--ivfflat-probes", type=str, default=None, help="Ignored for Weaviate (kept for compatibility)")
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

    if parse_int_list(args.ivfflat_probes):
        print("[WARN] Weaviate does not use ivfflat.probes. --ivfflat-probes is ignored.")
    if parse_int_list(args.hnsw_ef_search):
        print("[WARN] Weaviate does not expose per-query hnsw.ef_search override in this script.")

    setup_client = connect_weaviate()
    try:
        print("Loading vectors from database...")
        rows = load_all_vectors(setup_client, COLLECTION_NAME, limit=args.max_load_items)
        dim = int(rows[0].vector.shape[0])
        print(f"Loaded {len(rows)} vectors | dim={dim} | collection={COLLECTION_NAME}")
    finally:
        setup_client.close()

    rng = random.Random(args.seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    idxs = idxs[: min(args.query_pool, len(rows))]
    sampled_queries = [rows[i] for i in idxs]

    measured, qps, latencies = run_throughput_test(
        sampled_queries=sampled_queries,
        k=args.k,
        concurrency=args.concurrency,
        seconds=args.seconds,
        warmup_seconds=args.warmup_seconds,
    )

    run = BenchmarkRun(
        label="default",
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

    print_run(run)
    print("\nDone.")


if __name__ == "__main__":
    main()
