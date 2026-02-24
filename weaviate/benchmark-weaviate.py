#!/usr/bin/env python3
"""
Benchmark Recall@K for a Weaviate collection using exact NumPy ground truth.

What it does:
1) Loads all objects (sourceRowId + vectors) from Weaviate
2) Samples query points from the same collection (fixed seed)
3) Computes exact top-K in NumPy (excluding self-match)
4) Runs Weaviate vector search
5) Computes Recall@K and latency stats (avg / p50 / p95)

Examples:
  python3 benchmark_recall_weaviate.py
  python3 benchmark_recall_weaviate.py --k-values 1,5,10 --num-queries 480
  python3 benchmark_recall_weaviate.py --exact-weaviate
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
import weaviate

# v4 client imports (used in your earlier script style)
from weaviate.classes.query import Filter


WEAVIATE_HTTP_HOST = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "localhost")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_SECURE = os.getenv("WEAVIATE_SECURE", "false").lower() == "true"

COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "NewsArticle")

# We rely on this payload property added in the ingestion script
SOURCE_ID_PROPERTY = os.getenv("WEAVIATE_SOURCE_ID_PROPERTY", "sourceRowId")


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


def infer_distance_name(collection_config: Any) -> str:
    """
    Best-effort infer distance metric from Weaviate collection config.
    Returns one of: cosine, dot, l2 (euclidean), manhattan (if available), or 'cosine' fallback.
    """
    # Weaviate client object shapes can vary by version. We'll be defensive.
    text = str(collection_config).lower()

    if "cosine" in text:
        return "cosine"
    if "dot" in text:
        return "dot"
    if "l2" in text or "euclidean" in text:
        return "euclid"
    if "manhattan" in text:
        return "manhattan"

    # Most local vector setups default to cosine; your ingestion script sets cosine explicitly.
    return "cosine"
def connect_weaviate() -> weaviate.WeaviateClient:
    timeout_sec = float(os.getenv("WEAVIATE_CONNECT_TIMEOUT_SEC", "90"))
    retry_sleep_sec = float(os.getenv("WEAVIATE_CONNECT_RETRY_SEC", "2"))
    deadline = time.time() + max(timeout_sec, 1.0)
    last_err: Exception | None = None

    while time.time() < deadline:
        try:
            client = weaviate.connect_to_custom(
                http_host=WEAVIATE_HTTP_HOST,
                http_port=WEAVIATE_HTTP_PORT,
                http_secure=WEAVIATE_SECURE,
                grpc_host=WEAVIATE_GRPC_HOST,
                grpc_port=WEAVIATE_GRPC_PORT,
                grpc_secure=WEAVIATE_SECURE,
            )
            return client
        except Exception as err:
            last_err = err
            time.sleep(max(retry_sleep_sec, 0.2))

    raise RuntimeError(
        f"Could not connect to Weaviate within {timeout_sec:.0f}s "
        f"at {WEAVIATE_HTTP_HOST}:{WEAVIATE_HTTP_PORT}"
    ) from last_err


def load_all_points(client: weaviate.WeaviateClient, collection_name: str) -> list[PointVec]:
    """
    Loads all objects with vectors from Weaviate.
    Assumes each object has integer payload property SOURCE_ID_PROPERTY (e.g., sourceRowId).
    """
    collection = client.collections.get(collection_name)
    all_points: list[PointVec] = []

    # Try iterator API variants across client versions
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
            raise ValueError(
                f"Missing required property '{SOURCE_ID_PROPERTY}' on object. "
                f"Store a stable int ID in payload for benchmarking."
            )

        pid_raw = props[SOURCE_ID_PROPERTY]
        try:
            point_id = int(pid_raw)
        except Exception as e:
            raise ValueError(f"Non-integer {SOURCE_ID_PROPERTY} value: {pid_raw!r}") from e

        # Vector extraction across possible client representations
        vec = getattr(obj, "vector", None)
        if vec is None:
            # Some client versions may require explicit vector return options
            raise ValueError("Object vector not returned. Ensure iterator includes vectors.")

        if isinstance(vec, dict):
            # Named or default vector map
            if "default" in vec:
                vec = vec["default"]
            else:
                vec = next(iter(vec.values()))

        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Unexpected vector shape for point {point_id}: {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Point {point_id} contains NaN/Inf in vector")

        all_points.append(PointVec(point_id=point_id, vector=arr))

    if not all_points:
        raise RuntimeError("No points loaded from Weaviate collection")

    # Stable ordering by sourceRowId (helps reproducibility)
    all_points.sort(key=lambda x: x.point_id)
    return all_points


def weaviate_search(
    client: weaviate.WeaviateClient,
    *,
    collection_name: str,
    query_vector: np.ndarray,
    limit: int,
    exclude_id: int | None,
    hnsw_ef: int | None = None,  # accepted for CLI compatibility; may be ignored depending on client/server support
    exact: bool = False,
) -> SearchResult:
    """
    Run a Weaviate vector search and return only SOURCE_ID_PROPERTY values + latency.

    exact:
      Weaviate doesn't expose a universal per-query 'exact=True' equivalent in the same way as Qdrant.
      For compatibility, this flag is accepted but ANN search is still used unless your setup supports a custom route.
      Exact ground truth is computed in NumPy anyway (the benchmark source of truth).
    """
    collection = client.collections.get(collection_name)

    request_limit = limit + (1 if exclude_id is not None else 0)

    filters = None
    if exclude_id is not None:
        # Exclude self by payload property (sourceRowId)
        filters = Filter.by_property(SOURCE_ID_PROPERTY).not_equal(exclude_id)

    # Note: Weaviate per-query HNSW ef override is not consistently exposed across client versions.
    # We keep the parameter for script similarity but do not hard-fail if unsupported.
    _ = hnsw_ef
    _ = exact

    t0 = time.perf_counter()
    res = collection.query.near_vector(
        near_vector=query_vector.tolist(),
        limit=request_limit,
        filters=filters,
        return_properties=[SOURCE_ID_PROPERTY],
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    ids: list[int] = []
    objects = getattr(res, "objects", []) or []
    for obj in objects:
        props = getattr(obj, "properties", {}) or {}
        if SOURCE_ID_PROPERTY not in props:
            continue
        try:
            pid = int(props[SOURCE_ID_PROPERTY])
        except Exception:
            continue

        if exclude_id is not None and pid == exclude_id:
            continue

        ids.append(pid)
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
    Exact top-k ranking in NumPy using the same metric family as Weaviate.
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

    elif d in {"euclid", "l2"}:
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
    client: weaviate.WeaviateClient,
    collection_name: str,
    points: list[PointVec],
    distance_name: str,
    k_values: list[int],
    num_queries: int,
    seed: int,
    warmup_queries: int,
    hnsw_ef: int | None = None,
    exact_qdrant: bool = False,  # kept name for structural similarity; maps to exact-weaviate semantics here
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
        result = weaviate_search(
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
    p = argparse.ArgumentParser(description="Benchmark Recall@K for a Weaviate collection")
    p.add_argument("--k-values", default="1,5,10", help="Comma-separated K values")
    p.add_argument("--num-queries", type=int, default=480, help="How many query points to sample")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--warmup-queries", type=int, default=0, help="Warm-up queries not counted")
    p.add_argument("--max-load-items", type=int, default=None, help="Optional cap when loading items (debug)")
    p.add_argument(
        "--distance",
        choices=["cosine", "dot", "l2"],
        default="cosine",
        help="Distance metric used for exact NumPy ground-truth computation.",
    )
    p.add_argument("--hnsw-ef", type=str, default=None, help="Comma-separated hnsw_ef values to sweep (kept for compatibility; may be ignored by client/server)")
    p.add_argument("--exact-weaviate", action="store_true", help="Label run as exact baseline (Weaviate still uses ANN query; exact GT is NumPy)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    k_values = parse_k_values(args.k_values)
    hnsw_ef_values = parse_int_list(args.hnsw_ef)

    if args.num_queries <= 0:
        raise ValueError("--num-queries must be > 0")
    if args.warmup_queries < 0:
        raise ValueError("--warmup-queries must be >= 0")

    client = connect_weaviate()
    try:
        print(f"Weaviate: http{'s' if WEAVIATE_SECURE else ''}://{WEAVIATE_HTTP_HOST}:{WEAVIATE_HTTP_PORT}")
        print(f"Collection: {COLLECTION_NAME}")

        collection = client.collections.get(COLLECTION_NAME)
        try:
            cfg = collection.config.get()
        except Exception:
            cfg = None
        inferred_distance_name = infer_distance_name(cfg)
        distance_name = args.distance or inferred_distance_name
        if args.distance:
            print(f"Distance (cli): {distance_name}")
        else:
            print(f"Detected distance: {distance_name}")

        print("Loading points + vectors from Weaviate...")
        points = load_all_points(client, COLLECTION_NAME)
        if args.max_load_items is not None:
            points = points[: args.max_load_items]
        dim = int(points[0].vector.shape[0])

        for p in points:
            if p.vector.shape[0] != dim:
                raise ValueError(f"Inconsistent vector dim. point_id={p.point_id} dim={p.vector.shape[0]} expected={dim}")

        print(f"Loaded points: {len(points)} | dim={dim}")

        runs: list[BenchmarkRun] = []
        if args.exact_weaviate:
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
                    exact_qdrant=True,  # label compatibility
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
        elif not args.exact_weaviate:
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
    finally:
        client.close()


if __name__ == "__main__":
    main()
