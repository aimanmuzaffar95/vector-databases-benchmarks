"""
Benchmark QPS (Queries Per Second) for a local FAISS index.

What it does:
1) Loads FAISS artifacts (index + vectors + ids)
2) Samples query vectors (fixed seed)
3) Runs a timed throughput test with configurable concurrency
4) Uses FAISS index.search()
5) Reports QPS + latency stats (avg/p50/p95/p99)
6) Optionally sweeps ANN runtime params (nprobe / hnsw efSearch)
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
from pathlib import Path

import faiss
import numpy as np


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


def unwrap_index(idx: faiss.Index) -> faiss.Index:
    cur = idx
    while hasattr(cur, "index"):
        cur = cur.index
    return cur


def set_runtime_param(index: faiss.Index, *, nprobe: int | None = None, ef_search: int | None = None) -> str:
    base = unwrap_index(index)

    if nprobe is not None and ef_search is not None:
        raise ValueError("Set only one of nprobe or ef_search")

    if nprobe is not None:
        if hasattr(base, "nprobe"):
            base.nprobe = int(nprobe)
            return f"nprobe={int(nprobe)}"
        return f"nprobe={int(nprobe)} (ignored: unsupported index)"

    if ef_search is not None:
        if hasattr(base, "hnsw") and hasattr(base.hnsw, "efSearch"):
            base.hnsw.efSearch = int(ef_search)
            return f"efSearch={int(ef_search)}"
        return f"efSearch={int(ef_search)} (ignored: unsupported index)"

    return "default"


def search_faiss(index: faiss.Index, *, query_vector: np.ndarray, limit: int) -> float:
    q = query_vector.reshape(1, -1).astype(np.float32, copy=False)
    t0 = time.perf_counter()
    index.search(q, int(limit))
    return (time.perf_counter() - t0) * 1000.0


def worker_loop(
    *,
    index: faiss.Index,
    queries: list[RowVec],
    k: int,
    stop_time: float,
    warmup_stop_time: float,
) -> tuple[int, list[float]]:
    latencies: list[float] = []
    measured = 0

    idx = 0
    n = len(queries)
    while True:
        now = time.perf_counter()
        if now >= stop_time:
            break

        qr = queries[idx]
        idx = (idx + 1) % n

        latency_ms = search_faiss(index, query_vector=qr.vector, limit=k)

        if now >= warmup_stop_time:
            measured += 1
            latencies.append(latency_ms)

    return measured, latencies


def run_throughput_test(
    *,
    index: faiss.Index,
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
                index=index,
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
    p = argparse.ArgumentParser(description="Benchmark QPS for FAISS")
    p.add_argument("--artifacts-dir", default=os.getenv("FAISS_OUTPUT_DIR", "faiss/artifacts"))
    p.add_argument("--distance", default="cosine", choices=["cosine", "l2", "dot"], help="Label only; metric comes from built index")
    p.add_argument("--k", type=int, default=10, help="Top-k returned per query")
    p.add_argument("--seconds", type=float, default=10.0, help="Total test duration in seconds (includes warmup)")
    p.add_argument("--warmup-seconds", type=float, default=2.0, help="Warm-up duration excluded from stats")
    p.add_argument("--concurrency", type=int, default=8, help="Number of concurrent workers")
    p.add_argument("--query-pool", type=int, default=200, help="How many query vectors to sample (reused by workers)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--ivfflat-probes", type=str, default=None, help="Comma-separated nprobe values to sweep")
    p.add_argument("--hnsw-ef-search", type=str, default=None, help="Comma-separated efSearch values to sweep")
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

    artifacts_dir = Path(args.artifacts_dir)
    index_path = artifacts_dir / "index.faiss"
    vectors_path = artifacts_dir / "vectors.npy"
    ids_path = artifacts_dir / "ids.npy"
    metadata_path = artifacts_dir / "metadata.json"

    for p in [index_path, vectors_path, ids_path, metadata_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    print("Loading vectors from database...")
    index = faiss.read_index(str(index_path))
    vectors = np.load(vectors_path).astype(np.float32, copy=False)
    ids = np.load(ids_path).astype(np.int64, copy=False)

    if vectors.ndim != 2:
        raise ValueError(f"vectors.npy must be 2D, got shape={vectors.shape}")
    if ids.ndim != 1:
        raise ValueError(f"ids.npy must be 1D, got shape={ids.shape}")
    if vectors.shape[0] != ids.shape[0]:
        raise ValueError(f"vectors/ids row mismatch: vectors={vectors.shape[0]} ids={ids.shape[0]}")

    rows = [RowVec(row_id=int(ids[i]), vector=vectors[i]) for i in range(vectors.shape[0])]
    dim = int(rows[0].vector.shape[0])
    print(f"Loaded {len(rows)} vectors | dim={dim} | index={metadata.get('index_type', 'unknown')}")

    rng = random.Random(args.seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    idxs = idxs[: min(args.query_pool, len(rows))]
    sampled_queries = [rows[i] for i in idxs]

    runs: list[BenchmarkRun] = []

    def one_run(label: str, nprobe: int | None, ef_search: int | None) -> BenchmarkRun:
        _ = set_runtime_param(index, nprobe=nprobe, ef_search=ef_search)
        measured, qps, latencies = run_throughput_test(
            index=index,
            sampled_queries=sampled_queries,
            k=args.k,
            concurrency=args.concurrency,
            seconds=args.seconds,
            warmup_seconds=args.warmup_seconds,
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
            runs.append(one_run(label=f"nprobe={nprobe}", nprobe=nprobe, ef_search=None))
    elif hnsw_sweep:
        for ef in hnsw_sweep:
            runs.append(one_run(label=f"hnsw.ef_search={ef}", nprobe=None, ef_search=ef))
    else:
        runs.append(one_run(label="default", nprobe=None, ef_search=None))

    for r in runs:
        print_run(r)

    print("\nDone.")


if __name__ == "__main__":
    main()
