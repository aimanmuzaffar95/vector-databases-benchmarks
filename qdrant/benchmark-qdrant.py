#!/usr/bin/env python3
"""
Benchmark Recall@K for a Qdrant collection using exact NumPy ground truth.

What it does:
1) Scrolls all points (ids + vectors) from Qdrant
2) Samples query points from the same collection (fixed seed)
3) Computes exact top-K in NumPy (excluding self-match)
4) Runs Qdrant search (ANN or exact=True)
5) Computes Recall@K and latency stats (avg / p50 / p95)

Tested conceptually for qdrant-client APIs that support client.search(...)
If your client version prefers query_points(), see notes at bottom.

Examples:
  python3 benchmark_recall_qdrant.py
  python3 benchmark_recall_qdrant.py --k-values 1,5,10 --num-queries 480
  python3 benchmark_recall_qdrant.py --hnsw-ef 32,64,128,256
  python3 benchmark_recall_qdrant.py --exact-qdrant
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "news_articles")


@dataclass
class PointVec:
    point_id: int
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


def infer_distance_name(collection_info: Any) -> str:
    """
    Returns one of: cosine, euclid, dot, manhattan
    (we support exact ground truth for cosine/euclid/dot below)
    """
    vectors_cfg = collection_info.config.params.vectors

    # Qdrant may return single vector config or dict for named vectors.
    # Your insert script uses single unnamed vector config, so handle that first.
    distance = None
    if hasattr(vectors_cfg, "distance"):
        distance = vectors_cfg.distance
    elif isinstance(vectors_cfg, dict):
        # named vectors case; pick the first
        first_cfg = next(iter(vectors_cfg.values()))
        distance = first_cfg.distance

    if distance is None:
        raise RuntimeError("Could not infer Qdrant distance from collection config")

    # qdrant enum -> normalized string
    if distance == qm.Distance.COSINE:
        return "cosine"
    if distance == qm.Distance.EUCLID:
        return "euclid"
    if distance == qm.Distance.DOT:
        return "dot"
    if distance == qm.Distance.MANHATTAN:
        return "manhattan"
    return str(distance).lower()
def load_all_points(client: QdrantClient, collection_name: str) -> list[PointVec]:
    """
    Scroll the entire collection and fetch vectors.
    Assumes point IDs are numeric (true in your insert script).
    """
    all_points: list[PointVec] = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_vectors=True,
            with_payload=False,
        )

        for p in points:
            pid = p.id
            if isinstance(pid, bool):
                raise ValueError(f"Unexpected boolean point id: {pid}")
            if isinstance(pid, (int, np.integer)):
                point_id = int(pid)
            elif isinstance(pid, str) and pid.isdigit():
                point_id = int(pid)
            else:
                raise ValueError(f"Non-numeric point id found: {pid!r}. This benchmark assumes numeric IDs.")

            vec = p.vector
            # unnamed vector => list[float]
            if isinstance(vec, dict):
                # named-vector collections: pick first vector
                vec = next(iter(vec.values()))
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"Unexpected vector shape for point {point_id}: {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Point {point_id} contains NaN/Inf in vector")
            all_points.append(PointVec(point_id=point_id, vector=arr))

        if next_offset is None:
            break
        offset = next_offset

    if not all_points:
        raise RuntimeError("No points loaded from Qdrant collection")

    # Stable ordering by point id (helps reproducibility)
    all_points.sort(key=lambda x: x.point_id)
    return all_points


def qdrant_search(
    client: QdrantClient,
    *,
    collection_name: str,
    query_vector: np.ndarray,
    limit: int,
    exclude_id: int | None,
    hnsw_ef: int | None = None,
    exact: bool = False,
) -> SearchResult:
    """
    Run a Qdrant vector search and return only IDs + latency.
    Uses a filter to exclude the query point itself.
    """
    query_filter = None
    if exclude_id is not None:
        # Exclude by point ID (not payload field).
        try:
            query_filter = qm.Filter(must_not=[qm.HasIdCondition(has_id=[exclude_id])])
        except Exception:
            # Older clients may not expose HasIdCondition; keep client-side exclusion below.
            query_filter = None

    search_params = qm.SearchParams(
        hnsw_ef=hnsw_ef,
        exact=exact,
    )

    request_limit = limit + (1 if exclude_id is not None else 0)  # extra slot in case self comes back
    t0 = time.perf_counter()

    # qdrant-client API changed from search(...) -> query_points(...).
    # Support both to keep this benchmark runnable across versions.
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            query_filter=query_filter,
            limit=request_limit,
            with_payload=False,
            with_vectors=False,
            search_params=search_params,
        )
        hits = response.points
    else:
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            query_filter=query_filter,
            limit=request_limit,
            with_payload=False,
            with_vectors=False,
            search_params=search_params,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    ids: list[int] = []
    for h in hits:
        pid = h.id
        if isinstance(pid, str) and pid.isdigit():
            pid = int(pid)
        if not isinstance(pid, int):
            continue
        if exclude_id is not None and int(pid) == exclude_id:
            continue
        ids.append(int(pid))
        if len(ids) >= limit:
            break

    return SearchResult(ids=ids, latency_ms=latency_ms)
def build_matrix(points: list[PointVec]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array([p.point_id for p in points], dtype=np.int64)
    mat = np.stack([p.vector for p in points]).astype(np.float32, copy=False)
    return ids, mat


def exact_topk_ids(
    *,
    base_ids: np.ndarray,
    base_matrix: np.ndarray,
    query_vector: np.ndarray,
    distance_name: str,
    k: int,
    exclude_id: int | None = None,
) -> list[int]:
    """
    Exact top-k ranking in NumPy using the same metric family as Qdrant.
    """
    d = distance_name.lower()

    if d == "cosine":
        bn = maybe_normalize(base_matrix)
        q = query_vector.astype(np.float32, copy=False)
        qn = q / (np.linalg.norm(q) or 1.0)
        scores = bn @ qn  # higher better
        order = np.argsort(-scores)

    elif d == "dot":
        q = query_vector.astype(np.float32, copy=False)
        scores = base_matrix @ q  # higher better
        order = np.argsort(-scores)

    elif d == "euclid":
        q = query_vector.astype(np.float32, copy=False)
        diff = base_matrix - q
        d2 = np.einsum("ij,ij->i", diff, diff)  # lower better
        order = np.argsort(d2)

    elif d == "manhattan":
        q = query_vector.astype(np.float32, copy=False)
        d1 = np.abs(base_matrix - q).sum(axis=1)  # lower better
        order = np.argsort(d1)

    else:
        raise ValueError(f"Unsupported distance for exact GT: {distance_name}")

    out: list[int] = []
    for idx in order:
        rid = int(base_ids[idx])
        if exclude_id is not None and rid == exclude_id:
            continue
        out.append(rid)
        if len(out) >= k:
            break
    return out
def benchmark_once(
    *,
    client: QdrantClient,
    collection_name: str,
    points: list[PointVec],
    distance_name: str,
    k_values: list[int],
    num_queries: int,
    seed: int,
    warmup_queries: int,
    hnsw_ef: int | None = None,
    exact_qdrant: bool = False,
) -> BenchmarkRun:
    max_k = max(k_values)
    if len(points) <= max_k + 1:
        raise ValueError(f"Need > max_k+1 points. Have {len(points)}, max_k={max_k}")

    base_ids, base_matrix = build_matrix(points)

    rng = random.Random(seed)
    indices = list(range(len(points)))
    rng.shuffle(indices)
    indices = indices[: min(num_queries, len(points))]

    # Build exact GT first (outside measured latency)
    gt_map: dict[int, list[int]] = {}
    for i in indices:
        p = points[i]
        gt_map[p.point_id] = exact_topk_ids(
            base_ids=base_ids,
            base_matrix=base_matrix,
            query_vector=p.vector,
            distance_name=distance_name,
            k=max_k,
            exclude_id=p.point_id,
        )

    warmup_count = min(warmup_queries, len(indices))
    latencies: list[float] = []
    recalls: dict[int, list[float]] = {k: [] for k in k_values}

    for j, i in enumerate(indices):
        p = points[i]
        result = qdrant_search(
            client,
            collection_name=collection_name,
            query_vector=p.vector,
            limit=max_k,
            exclude_id=p.point_id,
            hnsw_ef=hnsw_ef,
            exact=exact_qdrant,
        )

        if j < warmup_count:
            continue

        latencies.append(result.latency_ms)
        gt_ids = gt_map[p.point_id]
        for k in k_values:
            recalls[k].append(recall_at_k(result.ids, gt_ids, k))

    if not latencies:
        raise RuntimeError("No measured queries after warm-up")

    label_parts = []
    if exact_qdrant:
        label_parts.append("exact=True")
    else:
        label_parts.append("exact=False")
    if hnsw_ef is not None:
        label_parts.append(f"hnsw_ef={hnsw_ef}")
    label = ", ".join(label_parts)

    return BenchmarkRun(
        label=label,
        metric=distance_name,
        k_values=k_values,
        num_queries=len(indices) - warmup_count,
        avg_recall_at_k={k: float(sum(v) / len(v)) for k, v in recalls.items()},
        avg_latency_ms=float(statistics.mean(latencies)),
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
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
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark Recall@K for a Qdrant collection")
    p.add_argument("--k-values", default="1,5,10", help="Comma-separated K values")
    p.add_argument("--num-queries", type=int, default=480, help="How many query points to sample")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--warmup-queries", type=int, default=0, help="Warm-up queries not counted")
    p.add_argument("--max-load-items", type=int, default=None, help="Optional cap when loading items (debug)")
    p.add_argument(
        "--distance",
        choices=["cosine", "dot", "euclid", "manhattan"],
        default="cosine",
        help="Distance metric for exact NumPy ground truth.",
    )
    p.add_argument("--hnsw-ef", type=str, default=None, help="Comma-separated hnsw_ef values to sweep")
    p.add_argument("--exact-qdrant", action="store_true", help="Force Qdrant exact=True (sanity baseline)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    k_values = parse_k_values(args.k_values)
    hnsw_ef_values = parse_int_list(args.hnsw_ef)

    if args.num_queries <= 0:
        raise ValueError("--num-queries must be > 0")
    if args.warmup_queries < 0:
        raise ValueError("--warmup-queries must be >= 0")

    client = QdrantClient(url=QDRANT_URL)

    print(f"Qdrant: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    info = client.get_collection(COLLECTION_NAME)
    inferred_distance_name = infer_distance_name(info)
    distance_name = args.distance or inferred_distance_name
    if args.distance:
        print(f"Distance (cli): {distance_name}")
    else:
        print(f"Detected distance: {distance_name}")

    print("Loading points + vectors from Qdrant...")
    points = load_all_points(client, COLLECTION_NAME)
    if args.max_load_items is not None:
        points = points[: args.max_load_items]
    dim = int(points[0].vector.shape[0])

    for p in points:
        if p.vector.shape[0] != dim:
            raise ValueError(f"Inconsistent vector dim. point_id={p.point_id} dim={p.vector.shape[0]} expected={dim}")

    print(f"Loaded points: {len(points)} | dim={dim}")

    runs: list[BenchmarkRun] = []
    if args.exact_qdrant:
        runs.append(
            benchmark_once(
                client=client,
                collection_name=COLLECTION_NAME,
                points=points,
                distance_name=distance_name,
                k_values=k_values,
                num_queries=args.num_queries,
                seed=args.seed,
                warmup_queries=args.warmup_queries,
                hnsw_ef=None,
                exact_qdrant=True,
            )
        )
    if hnsw_ef_values:
        for ef in hnsw_ef_values:
            runs.append(
                benchmark_once(
                    client=client,
                    collection_name=COLLECTION_NAME,
                    points=points,
                    distance_name=distance_name,
                    k_values=k_values,
                    num_queries=args.num_queries,
                    seed=args.seed,
                    warmup_queries=args.warmup_queries,
                    hnsw_ef=ef,
                    exact_qdrant=False,
                )
            )
    elif not args.exact_qdrant:
        runs.append(
            benchmark_once(
                client=client,
                collection_name=COLLECTION_NAME,
                points=points,
                distance_name=distance_name,
                k_values=k_values,
                num_queries=args.num_queries,
                seed=args.seed,
                warmup_queries=args.warmup_queries,
                hnsw_ef=None,
                exact_qdrant=False,
            )
        )

    for run in runs:
        print_run(run)

    print("\nDone.")


if __name__ == "__main__":
    main()
