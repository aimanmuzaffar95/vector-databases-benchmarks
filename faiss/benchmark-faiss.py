#!/usr/bin/env python3
"""
Benchmark Recall@K for a local FAISS index using exact NumPy ground truth.

Expected artifacts from faiss/insert-data-faiss.py:
- index.faiss
- vectors.npy
- ids.npy
- metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


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
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float


def parse_k_values(text: str) -> list[int]:
    vals: list[int] = []
    for p in text.split(","):
        p = p.strip()
        if not p:
            continue
        k = int(p)
        if k <= 0:
            raise ValueError(f"K must be > 0, got {k}")
        vals.append(k)
    if not vals:
        raise ValueError("No k-values provided")
    return sorted(set(vals))


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    out = []
    for p in text.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def recall_at_k(pred_ids: list[int], gt_ids: list[int], k: int) -> float:
    gt_topk = gt_ids[:k]
    if not gt_topk:
        return 0.0
    pred_topk = pred_ids[:k]
    hits = len(set(pred_topk) & set(gt_topk))
    return hits / len(gt_topk)


def maybe_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Recall@K for local FAISS index")

    p.add_argument("--artifacts-dir", default=os.getenv("FAISS_OUTPUT_DIR", "faiss/artifacts"))
    p.add_argument(
        "--distance",
        default="cosine",
        choices=["cosine", "dot", "l2"],
        help="Distance metric for benchmark and exact NumPy ground truth.",
    )
    p.add_argument("--k-values", default="1,5,10")
    p.add_argument("--num-queries", type=int, default=480)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nprobe", default=None, help="Comma-separated nprobe values for IVF")
    p.add_argument("--hnsw-ef-search", default=None, help="Comma-separated efSearch values for HNSW")

    return p.parse_args()


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


def exact_topk_ids(
    *,
    vectors: np.ndarray,
    ids: np.ndarray,
    query_vec: np.ndarray,
    query_id: int,
    metric: str,
    topk: int,
) -> list[int]:
    if metric == "cosine":
        q = maybe_normalize(query_vec.reshape(1, -1).astype(np.float32, copy=False))
        base = maybe_normalize(vectors)
        scores = (base @ q.T).ravel()
        order = np.argsort(-scores)
    elif metric == "dot":
        q = query_vec.reshape(1, -1).astype(np.float32, copy=False)
        scores = (vectors @ q.T).ravel()
        order = np.argsort(-scores)
    elif metric == "l2":
        q = query_vec.reshape(1, -1).astype(np.float32, copy=False)
        d2 = np.sum((vectors - q) ** 2, axis=1)
        order = np.argsort(d2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    ranked_ids = ids[order].tolist()
    ranked_ids = [rid for rid in ranked_ids if rid != query_id]
    return ranked_ids[:topk]


def faiss_search(index: faiss.Index, query_vector: np.ndarray, *, limit: int, exclude_id: int | None) -> SearchResult:
    request_limit = limit + (1 if exclude_id is not None else 0)

    q = query_vector.reshape(1, -1).astype(np.float32, copy=False)

    t0 = time.perf_counter()
    _distances, labels = index.search(q, int(request_limit))
    latency_ms = (time.perf_counter() - t0) * 1000.0

    ids: list[int] = []
    for rid in labels[0].tolist():
        if rid < 0:
            continue
        rid_int = int(rid)
        if exclude_id is not None and rid_int == int(exclude_id):
            continue
        ids.append(rid_int)
        if len(ids) >= limit:
            break

    return SearchResult(ids=ids, latency_ms=latency_ms)


def benchmark_once(
    *,
    index: faiss.Index,
    vectors: np.ndarray,
    ids: np.ndarray,
    metric: str,
    k_values: list[int],
    num_queries: int,
    seed: int,
    param_label: str,
) -> BenchmarkRun:
    max_k = max(k_values)

    if num_queries > len(ids):
        raise ValueError(f"num_queries ({num_queries}) cannot exceed dataset size ({len(ids)})")

    idxs = list(range(len(ids)))
    random.Random(seed).shuffle(idxs)
    query_idxs = idxs[:num_queries]

    recall_acc = {k: 0.0 for k in k_values}
    latencies_ms: list[float] = []

    for qi in query_idxs:
        qid = int(ids[qi])
        qv = vectors[qi]

        gt_ids = exact_topk_ids(
            vectors=vectors,
            ids=ids,
            query_vec=qv,
            query_id=qid,
            metric=metric,
            topk=max_k,
        )

        sr = faiss_search(index, qv, limit=max_k, exclude_id=qid)
        latencies_ms.append(sr.latency_ms)

        for k in k_values:
            recall_acc[k] += recall_at_k(sr.ids, gt_ids, k)

    avg_recall_at_k = {k: recall_acc[k] / num_queries for k in k_values}

    return BenchmarkRun(
        label=param_label,
        metric=metric,
        k_values=k_values,
        num_queries=num_queries,
        avg_recall_at_k=avg_recall_at_k,
        avg_latency_ms=statistics.mean(latencies_ms) if latencies_ms else float("nan"),
        p50_latency_ms=percentile(latencies_ms, 50),
        p95_latency_ms=percentile(latencies_ms, 95),
    )


def print_run(run: BenchmarkRun) -> None:
    print("===============")
    print("Benchmark Results")
    print("===============")
    print(f"Run: {run.label}")
    print(f"Distance: {run.metric}")
    print(f"Measured queries: {run.num_queries}")
    print("-------------------------------------")
    for k in run.k_values:
        print(f"Recall@{k}: {run.avg_recall_at_k[k]:.4f}")
    print("-------------------------------------")
    print(f"Latency avg: {run.avg_latency_ms:.2f} ms")
    print(f"Latency p50: {run.p50_latency_ms:.2f} ms")
    print(f"Latency p95: {run.p95_latency_ms:.2f} ms")
    print("-------------------------------------")


def main() -> None:
    args = parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    index_path = artifacts_dir / "index.faiss"
    vectors_path = artifacts_dir / "vectors.npy"
    ids_path = artifacts_dir / "ids.npy"
    metadata_path = artifacts_dir / "metadata.json"

    for p in [index_path, vectors_path, ids_path, metadata_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metric = str(metadata.get("distance", "cosine")).lower()
    if args.distance is not None:
        metric = args.distance.lower()

    print("Loading FAISS artifacts...")
    index = faiss.read_index(str(index_path))
    vectors = np.load(vectors_path).astype(np.float32, copy=False)
    ids = np.load(ids_path).astype(np.int64, copy=False)

    if vectors.ndim != 2:
        raise ValueError(f"vectors.npy must be 2D, got shape={vectors.shape}")
    if ids.ndim != 1:
        raise ValueError(f"ids.npy must be 1D, got shape={ids.shape}")
    if vectors.shape[0] != ids.shape[0]:
        raise ValueError(
            f"vectors/ids row mismatch: vectors={vectors.shape[0]} ids={ids.shape[0]}"
        )

    print(f"Rows: {vectors.shape[0]} | Dim: {vectors.shape[1]} | Metric: {metric}")
    print(f"Index type (metadata): {metadata.get('index_type', 'unknown')}")

    k_values = parse_k_values(args.k_values)
    nprobe_list = parse_int_list(args.nprobe)
    ef_list = parse_int_list(args.hnsw_ef_search)

    if nprobe_list and ef_list:
        raise ValueError("Please set only one sweep: --nprobe or --hnsw-ef-search")

    runs: list[BenchmarkRun] = []

    if nprobe_list:
        for nprobe in nprobe_list:
            label = set_runtime_param(index, nprobe=nprobe)
            run = benchmark_once(
                index=index,
                vectors=vectors,
                ids=ids,
                metric=metric,
                k_values=k_values,
                num_queries=args.num_queries,
                seed=args.seed,
                param_label=label,
            )
            runs.append(run)
    elif ef_list:
        for ef in ef_list:
            label = set_runtime_param(index, ef_search=ef)
            run = benchmark_once(
                index=index,
                vectors=vectors,
                ids=ids,
                metric=metric,
                k_values=k_values,
                num_queries=args.num_queries,
                seed=args.seed,
                param_label=label,
            )
            runs.append(run)
    else:
        label = set_runtime_param(index)
        run = benchmark_once(
            index=index,
            vectors=vectors,
            ids=ids,
            metric=metric,
            k_values=k_values,
            num_queries=args.num_queries,
            seed=args.seed,
            param_label=label,
        )
        runs.append(run)

    for run in runs:
        print_run(run)

    print("\nDone.")


if __name__ == "__main__":
    main()
