"""
Benchmark QPS (Queries Per Second) for a Milvus collection.

What it does:
1) Loads ids + vectors from Milvus (for query sampling)
2) Samples query vectors (fixed seed)
3) Runs a timed throughput test with configurable concurrency
4) Uses Milvus ANN search
5) Reports QPS + latency stats (avg/p50/p95/p99)
6) Optionally sweeps ANN runtime params (nprobe / hnsw ef)
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
from pymilvus import Collection, connections, utility


MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "news_articles")
MILVUS_CONSISTENCY_LEVEL = os.getenv("MILVUS_CONSISTENCY_LEVEL", "Bounded")


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


def normalize_distance_name(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in {"ip", "dot", "inner_product", "inner-product"}:
        return "ip"
    if v in {"l2", "euclid", "euclidean"}:
        return "l2"
    return "cosine"


def connect(alias: str) -> None:
    connections.connect(alias=alias, host=MILVUS_HOST, port=str(MILVUS_PORT))


def load_all_vectors(collection: Collection, limit: int | None = None, batch_size: int = 5000) -> list[RowVec]:
    rows: list[RowVec] = []
    last_id = -1

    while True:
        expr = "" if last_id < 0 else f"id > {last_id}"
        result = collection.query(
            expr=expr,
            output_fields=["id", "embedding"],
            limit=batch_size,
            consistency_level=MILVUS_CONSISTENCY_LEVEL,
        )
        if not result:
            break

        result.sort(key=lambda r: int(r["id"]))
        for r in result:
            rid = int(r["id"])
            vec = np.asarray(r["embedding"], dtype=np.float32)
            if vec.ndim != 1 or not np.all(np.isfinite(vec)):
                continue
            rows.append(RowVec(row_id=rid, vector=vec))
            last_id = rid
            if limit is not None and len(rows) >= int(limit):
                return rows

        if len(result) < batch_size:
            break

    if not rows:
        raise RuntimeError("No vectors loaded from Milvus collection.")
    return rows


def search_milvus(
    collection: Collection,
    *,
    query_vector: np.ndarray,
    limit: int,
    distance: str,
    hnsw_ef: int | None = None,
    nprobe: int | None = None,
) -> float:
    metric = normalize_distance_name(distance).upper()
    if metric == "IP":
        metric = "IP"
    if metric == "COSINE":
        metric = "COSINE"

    search_params: dict[str, object] = {"metric_type": metric, "params": {}}
    params = search_params["params"]
    assert isinstance(params, dict)
    if hnsw_ef is not None:
        params["ef"] = int(hnsw_ef)
    if nprobe is not None:
        params["nprobe"] = int(nprobe)

    t0 = time.perf_counter()
    collection.search(
        data=[query_vector.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=int(limit),
        output_fields=["id"],
        consistency_level=MILVUS_CONSISTENCY_LEVEL,
    )
    return (time.perf_counter() - t0) * 1000.0


def worker_loop(
    *,
    queries: list[RowVec],
    k: int,
    distance: str,
    stop_time: float,
    warmup_stop_time: float,
    hnsw_ef: int | None,
    nprobe: int | None,
    worker_id: int,
) -> tuple[int, list[float]]:
    latencies: list[float] = []
    measured = 0

    alias = f"qps_worker_{worker_id}"
    connect(alias)
    collection = Collection(MILVUS_COLLECTION, using=alias)
    collection.load()

    idx = 0
    n = len(queries)
    try:
        while True:
            now = time.perf_counter()
            if now >= stop_time:
                break

            qr = queries[idx]
            idx = (idx + 1) % n

            latency_ms = search_milvus(
                collection,
                query_vector=qr.vector,
                limit=k,
                distance=distance,
                hnsw_ef=hnsw_ef,
                nprobe=nprobe,
            )

            if now >= warmup_stop_time:
                measured += 1
                latencies.append(latency_ms)
    finally:
        try:
            collection.release()
        except Exception:
            pass
        connections.disconnect(alias)

    return measured, latencies


def run_throughput_test(
    *,
    sampled_queries: list[RowVec],
    k: int,
    distance: str,
    concurrency: int,
    seconds: float,
    warmup_seconds: float,
    hnsw_ef: int | None,
    nprobe: int | None,
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
                distance=distance,
                stop_time=stop,
                warmup_stop_time=warmup_stop,
                hnsw_ef=hnsw_ef,
                nprobe=nprobe,
                worker_id=i,
            )
            for i in range(concurrency)
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
    p = argparse.ArgumentParser(description="Benchmark QPS for Milvus")
    p.add_argument("--distance", default="cosine", choices=["cosine", "l2", "ip", "dot"], help="Distance/similarity metric")
    p.add_argument("--k", type=int, default=10, help="Top-k returned per query")
    p.add_argument("--seconds", type=float, default=10.0, help="Total test duration in seconds (includes warmup)")
    p.add_argument("--warmup-seconds", type=float, default=2.0, help="Warm-up duration excluded from stats")
    p.add_argument("--concurrency", type=int, default=8, help="Number of concurrent workers")
    p.add_argument("--query-pool", type=int, default=200, help="How many query vectors to sample (reused by workers)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--max-load-items", type=int, default=None, help="Optional cap when loading vectors (debugging)")
    p.add_argument("--ivfflat-probes", type=str, default=None, help="Comma-separated nprobe values to sweep")
    p.add_argument("--hnsw-ef-search", type=str, default=None, help="Comma-separated ef values to sweep")
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

    collection: Collection | None = None
    connect("default")
    try:
        if not utility.has_collection(MILVUS_COLLECTION):
            raise RuntimeError(f"Collection not found: {MILVUS_COLLECTION}")

        collection = Collection(MILVUS_COLLECTION, using="default")
        collection.load()

        print("Loading vectors from database...")
        rows = load_all_vectors(collection, limit=args.max_load_items)
        dim = int(rows[0].vector.shape[0])
        print(f"Loaded {len(rows)} vectors | dim={dim} | collection={MILVUS_COLLECTION}")

        rng = random.Random(args.seed)
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        idxs = idxs[: min(args.query_pool, len(rows))]
        sampled_queries = [rows[i] for i in idxs]

        runs: list[BenchmarkRun] = []

        def one_run(label: str, nprobe: int | None, hnsw_ef: int | None) -> BenchmarkRun:
            measured, qps, latencies = run_throughput_test(
                sampled_queries=sampled_queries,
                k=args.k,
                distance=args.distance,
                concurrency=args.concurrency,
                seconds=args.seconds,
                warmup_seconds=args.warmup_seconds,
                hnsw_ef=hnsw_ef,
                nprobe=nprobe,
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
            for nprobe in ivf_sweep:
                runs.append(one_run(label=f"nprobe={nprobe}", nprobe=nprobe, hnsw_ef=None))
        elif hnsw_sweep:
            for ef in hnsw_sweep:
                runs.append(one_run(label=f"hnsw.ef_search={ef}", nprobe=None, hnsw_ef=ef))
        else:
            runs.append(one_run(label="default", nprobe=None, hnsw_ef=None))

        for r in runs:
            print_run(r)

        print("\nDone.")
    finally:
        if collection is not None:
            try:
                collection.release()
            except Exception:
                pass
        connections.disconnect("default")


if __name__ == "__main__":
    main()
