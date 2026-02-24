"""
Benchmark QPS (Queries Per Second) for a Chroma collection.

What it does:
1) Loads ids + embeddings from Chroma (for query sampling)
2) Samples query vectors (fixed seed)
3) Runs a timed throughput test with configurable concurrency
4) Uses Chroma vector query API
5) Reports QPS + latency stats (avg/p50/p95/p99)
6) Accepts ANN sweep flags for CLI compatibility (ignored by Chroma)

Usage examples:
  python3 chroma/benchmark-qps-chroma.py
  python3 chroma/benchmark-qps-chroma.py --distance cosine --k 10 --seconds 10 --concurrency 8
  python3 chroma/benchmark-qps-chroma.py --where-json '{"genre":"Sports"}'
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import chromadb
except ImportError as e:
    raise SystemExit("Missing dependency. Install with: pip install chromadb") from e


CHROMA_MODE = os.getenv("CHROMA_MODE", "http").strip().lower()
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "news_articles")
ID_TYPE = os.getenv("ID_TYPE", "int").strip().lower()


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


def get_client():
    if CHROMA_MODE == "http":
        return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    if CHROMA_MODE == "persistent":
        return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    raise ValueError("CHROMA_MODE must be 'http' or 'persistent'")


def coerce_id(x: str) -> int | str:
    if ID_TYPE == "int":
        try:
            return int(x)
        except ValueError as e:
            raise ValueError(f"ID_TYPE=int but found non-int id in Chroma: {x!r}") from e
    return x


def load_all_vectors(
    collection,
    *,
    where: dict[str, Any] | None = None,
    limit: int | None = None,
) -> list[RowVec]:
    data = collection.get(include=["embeddings"], where=where)
    ids_raw = data.get("ids")
    embs_raw = data.get("embeddings")

    if ids_raw is None or embs_raw is None:
        raise RuntimeError("No vectors loaded from Chroma collection.")

    rows: list[RowVec] = []
    for rid, emb in zip(ids_raw, embs_raw):
        row_id = coerce_id(rid)
        vec = np.asarray(emb, dtype=np.float32)
        if vec.size == 0 or not np.all(np.isfinite(vec)):
            continue
        rows.append(RowVec(row_id=row_id, vector=vec))

    if limit is not None:
        rows = rows[: int(limit)]

    if not rows:
        raise RuntimeError("No vectors loaded from Chroma (after limit/filter).")
    return rows


def search_chroma(
    collection,
    *,
    query_vector: np.ndarray,
    limit: int,
    where: dict[str, Any] | None = None,
) -> float:
    t0 = time.perf_counter()
    collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=int(limit),
        where=where,
        include=["distances"],
    )
    return (time.perf_counter() - t0) * 1000.0


def worker_loop(
    *,
    queries: list[RowVec],
    k: int,
    stop_time: float,
    warmup_stop_time: float,
    where: dict[str, Any] | None,
) -> tuple[int, list[float]]:
    latencies: list[float] = []
    measured = 0

    client = get_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    idx = 0
    n = len(queries)
    while True:
        now = time.perf_counter()
        if now >= stop_time:
            break

        qr = queries[idx]
        idx = (idx + 1) % n

        latency_ms = search_chroma(
            collection,
            query_vector=qr.vector,
            limit=k,
            where=where,
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
    where: dict[str, Any] | None,
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
                where=where,
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
    p = argparse.ArgumentParser(description="Benchmark QPS for Chroma")
    p.add_argument("--distance", default="cosine", choices=["cosine", "l2", "ip"], help="Label only; Chroma metric is set at collection creation")
    p.add_argument("--k", type=int, default=10, help="Top-k returned per query")
    p.add_argument("--seconds", type=float, default=10.0, help="Total test duration in seconds (includes warmup)")
    p.add_argument("--warmup-seconds", type=float, default=2.0, help="Warm-up duration excluded from stats")
    p.add_argument("--concurrency", type=int, default=8, help="Number of concurrent workers")
    p.add_argument("--query-pool", type=int, default=200, help="How many query vectors to sample (reused by workers)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--max-load-items", type=int, default=None, help="Optional cap when loading vectors (debugging)")
    p.add_argument("--where-json", type=str, default=None, help="Optional Chroma where filter as JSON")
    p.add_argument("--hnsw-ef-search", type=str, default=None, help="Ignored for Chroma (kept for compatibility)")
    p.add_argument("--ivfflat-probes", type=str, default=None, help="Ignored for Chroma (kept for compatibility)")
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

    where = None
    if args.where_json:
        where = json.loads(args.where_json)
        if not isinstance(where, dict):
            raise ValueError("--where-json must decode to a JSON object")

    if parse_int_list(args.ivfflat_probes):
        print("[WARN] Chroma does not use ivfflat.probes. --ivfflat-probes is ignored.")
    if parse_int_list(args.hnsw_ef_search):
        print("[WARN] Chroma does not expose hnsw.ef_search override. --hnsw-ef-search is ignored.")

    client = get_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    print("Loading vectors from database...")
    rows = load_all_vectors(collection, where=where, limit=args.max_load_items)
    dim = int(rows[0].vector.shape[0])
    print(f"Loaded {len(rows)} vectors | dim={dim} | collection={COLLECTION_NAME}")

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
        where=where,
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
