#!/usr/bin/env python3
"""
Milvus Recall@K Benchmark (NumPy Exact Ground Truth vs ANN search)

Benchmark goals:
- Load all vectors + IDs from Milvus collection
- Build exact ground truth in NumPy (cosine / dot / l2)
- Run ANN search in Milvus
- Exclude self-match
- Report Recall@K and latency (avg/p50/p95)

Compatible with local Milvus standalone (e.g. v2.3.x) using pymilvus.

Example:
  python3 milvus-recall-benchmark.py \
    --host localhost --port 19530 \
    --collection news_articles \
    --distance cosine \
    --num-queries 200 \
    --k-values 1,5,10 \
    --hnsw-ef 32,64,128,256

If your index is IVF-based, you can also sweep nprobe:
  python3 milvus-recall-benchmark.py --nprobe 8,16,32
"""

import os
import time
import math
import argparse
import statistics
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np

from pymilvus import connections, Collection, utility


# ----------------------------
# Helpers
# ----------------------------

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def percentile_ms(values_ms: List[float], p: float) -> float:
    if not values_ms:
        return float("nan")
    arr = np.array(values_ms, dtype=np.float64)
    return float(np.percentile(arr, p))

def format_ms(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.2f}"

def recall_at_k(gt_ids: List[int], ann_ids: List[int], k: int) -> float:
    gt_topk = set(gt_ids[:k])
    if not gt_topk:
        return 0.0
    ann_topk = set(ann_ids[:k])
    return len(gt_topk & ann_topk) / float(k)

def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms

def infer_index_type_and_metric(coll: Collection) -> Tuple[str, Optional[str]]:
    """
    Tries to infer index type and metric from collection index metadata.
    Falls back gracefully if API shape differs by pymilvus version.
    """
    idx_type = "UNKNOWN"
    metric = None
    try:
        idxs = coll.indexes
        if idxs:
            idx = idxs[0]
            params = getattr(idx, "params", None) or {}
            # Possible shapes vary by version:
            # {'index_type': 'HNSW', 'metric_type': 'COSINE', 'params': {'M': '16', 'efConstruction': '200'}}
            # or nested structures
            if isinstance(params, dict):
                idx_type = params.get("index_type", idx_type)
                metric = params.get("metric_type", metric)
                nested = params.get("params")
                if isinstance(nested, dict):
                    metric = nested.get("metric_type", metric)
            # Some versions expose dict via to_dict()
            if hasattr(idx, "to_dict"):
                d = idx.to_dict()
                if isinstance(d, dict):
                    idx_param = d.get("index_param", {}) or {}
                    idx_type = idx_param.get("index_type", idx_type)
                    metric = idx_param.get("metric_type", metric)
    except Exception:
        pass
    return idx_type, metric


def normalize_distance_name(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in ("ip", "dot", "inner_product", "inner-product"):
        return "dot"
    if v in ("l2", "euclid", "euclidean"):
        return "l2"
    return "cosine"

def print_run(
    engine_label: str,
    collection_name: str,
    db_metric: str,
    gt_metric: str,
    num_vectors: int,
    dim: int,
    num_queries: int,
    k_values: List[int],
    recall_map: Dict[int, float],
    latencies_ms: List[float],
    ann_param_label: str,
    ann_param_value,
):
    avg_ms = statistics.mean(latencies_ms) if latencies_ms else float("nan")
    p50_ms = percentile_ms(latencies_ms, 50)
    p95_ms = percentile_ms(latencies_ms, 95)

    print("\n" + "=" * 72)
    print("Benchmark Results")
    print("=" * 72)
    print(f"Engine              : {engine_label}")
    print(f"Collection           : {collection_name}")
    print(f"DB Metric            : {db_metric}")
    print(f"GT Metric (NumPy)    : {gt_metric}")
    print(f"Vectors Loaded       : {num_vectors}")
    print(f"Vector Dim           : {dim}")
    print(f"Queries Benchmarked  : {num_queries}")
    print(f"{ann_param_label:<20}: {ann_param_value}")
    print("-" * 72)

    for k in k_values:
        print(f"Recall@{k:<2}          : {recall_map.get(k, float('nan')):.4f}")

    print("-" * 72)
    print(f"Latency avg (ms)     : {format_ms(avg_ms)}")
    print(f"Latency p50 (ms)     : {format_ms(p50_ms)}")
    print(f"Latency p95 (ms)     : {format_ms(p95_ms)}")
    print("=" * 72)

# ----------------------------
# Exact Ground Truth (NumPy)
# ----------------------------

class ExactGroundTruth:
    def __init__(self, ids: np.ndarray, vectors: np.ndarray, metric: str):
        self.ids = ids.astype(np.int64)
        self.vectors = vectors.astype(np.float32)
        self.metric = metric.lower()

        if self.metric not in ("cosine", "dot", "ip", "l2", "euclid", "euclidean"):
            raise ValueError(f"Unsupported ground-truth metric: {metric}")

        if self.metric == "cosine":
            self.base = normalize_rows(self.vectors)
        else:
            self.base = self.vectors

    def topk(self, query_vec: np.ndarray, query_id: int, k: int) -> List[int]:
        q = query_vec.astype(np.float32).reshape(1, -1)

        if self.metric == "cosine":
            qn = normalize_rows(q)  # shape (1,d)
            scores = (self.base @ qn.T).ravel()  # higher is better
            order = np.argsort(-scores)
        elif self.metric in ("dot", "ip"):
            scores = (self.base @ q.T).ravel()   # higher is better
            order = np.argsort(-scores)
        else:  # l2 / euclid / euclidean
            diff = self.base - q
            dists = np.sum(diff * diff, axis=1)  # lower is better
            order = np.argsort(dists)

        ranked_ids = self.ids[order].tolist()

        # Exclude self-match
        ranked_ids = [rid for rid in ranked_ids if rid != query_id]

        return ranked_ids[:k]

# ----------------------------
# Milvus Data Loading
# ----------------------------

def load_all_ids_and_vectors(
    coll: Collection,
    batch_size: int = 10000,
    id_field: str = "id",
    vector_field: str = "embedding",
    consistency_level: str = "Bounded",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all IDs and vectors from Milvus.
    Uses query iterator-style pagination via `id >= last_id` ordering approach.

    Assumes stable integer primary keys and dense-ish ids (0..N-1 in your case).
    """
    expr = ""
    out_fields = [id_field, vector_field]

    all_ids: List[int] = []
    all_vecs: List[np.ndarray] = []

    # In your setup IDs are stable 0..N-1, so paging by id is simple and efficient.
    last_id = -1
    while True:
        if last_id < 0:
            expr = ""
        else:
            expr = f"{id_field} > {last_id}"

        rows = coll.query(
            expr=expr,
            output_fields=out_fields,
            limit=batch_size,
            consistency_level=consistency_level,
        )
        if not rows:
            break

        # Sort client-side by id to ensure deterministic paging if backend doesn't guarantee order
        rows.sort(key=lambda r: int(r[id_field]))

        ids_chunk = [int(r[id_field]) for r in rows]
        vecs_chunk = [r[vector_field] for r in rows]

        all_ids.extend(ids_chunk)
        all_vecs.extend(vecs_chunk)

        last_id = ids_chunk[-1]

        if len(rows) < batch_size:
            break

    ids_np = np.array(all_ids, dtype=np.int64)
    vecs_np = np.array(all_vecs, dtype=np.float32)

    if ids_np.size == 0:
        raise RuntimeError("No vectors found in collection.")

    return ids_np, vecs_np

# ----------------------------
# Milvus ANN Search
# ----------------------------

def milvus_search_single(
    coll: Collection,
    query_vec: np.ndarray,
    topn: int,
    metric: str,
    consistency_level: str,
    hnsw_ef: Optional[int] = None,
    nprobe: Optional[int] = None,
    exclude_id: Optional[int] = None,
    vector_field: str = "embedding",
) -> List[int]:
    """
    Run a single ANN search in Milvus and return IDs (self excluded client-side).
    """
    metric_upper = metric.upper()
    if metric_upper == "DOT":
        # Milvus typically uses IP (inner product) instead of DOT naming
        metric_upper = "IP"
    elif metric_upper == "COSINE":
        metric_upper = "COSINE"
    elif metric_upper in ("L2", "EUCLID", "EUCLIDEAN"):
        metric_upper = "L2"

    search_params = {"metric_type": metric_upper, "params": {}}

    # HNSW ef
    if hnsw_ef is not None:
        search_params["params"]["ef"] = int(hnsw_ef)

    # IVF nprobe
    if nprobe is not None:
        search_params["params"]["nprobe"] = int(nprobe)

    # Ask for a few extras because we will exclude self client-side
    limit = int(topn + 1)

    res = coll.search(
        data=[query_vec.tolist()],
        anns_field=vector_field,
        param=search_params,
        limit=limit,
        output_fields=["id"],  # id also available via hit.id, this keeps compatibility
        consistency_level=consistency_level,
    )

    ids: List[int] = []
    if res and len(res) > 0:
        for hit in res[0]:
            hid = int(getattr(hit, "id", hit.entity.get("id")))
            if exclude_id is not None and hid == int(exclude_id):
                continue
            ids.append(hid)
            if len(ids) >= topn:
                break

    return ids

# ----------------------------
# Main Benchmark
# ----------------------------

def run_benchmark_once(
    coll: Collection,
    all_ids: np.ndarray,
    all_vecs: np.ndarray,
    query_indices: np.ndarray,
    k_values: List[int],
    gt_metric: str,
    db_metric: str,
    consistency_level: str,
    hnsw_ef: Optional[int],
    nprobe: Optional[int],
):
    max_k = max(k_values)

    gt = ExactGroundTruth(all_ids, all_vecs, metric=gt_metric)

    latency_ms: List[float] = []
    recall_accum: Dict[int, List[float]] = {k: [] for k in k_values}

    for qi in query_indices:
        qid = int(all_ids[qi])
        qvec = all_vecs[qi]

        # Exact GT (NumPy)
        gt_ids = gt.topk(qvec, query_id=qid, k=max_k)

        # ANN search latency
        t0 = time.perf_counter()
        ann_ids = milvus_search_single(
            coll=coll,
            query_vec=qvec,
            topn=max_k,
            metric=db_metric,
            consistency_level=consistency_level,
            hnsw_ef=hnsw_ef,
            nprobe=nprobe,
            exclude_id=qid,
        )
        t1 = time.perf_counter()
        latency_ms.append((t1 - t0) * 1000.0)

        # Recall@K
        for k in k_values:
            recall_accum[k].append(recall_at_k(gt_ids, ann_ids, k))

    recall_map = {k: float(np.mean(v)) if v else float("nan") for k, v in recall_accum.items()}
    return recall_map, latency_ms

def main():
    parser = argparse.ArgumentParser(description="Milvus Recall@K benchmark with NumPy exact ground truth")
    parser.add_argument("--host", default=os.getenv("MILVUS_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MILVUS_PORT", "19530")))
    parser.add_argument("--collection", default=os.getenv("MILVUS_COLLECTION", "news_articles"))

    parser.add_argument(
        "--distance",
        default=os.getenv("MILVUS_DISTANCE", os.getenv("MILVUS_METRIC", "cosine")),
        help="Distance for both Milvus search + NumPy GT: cosine | dot | ip | l2",
    )
    parser.add_argument(
        "--consistency-level",
        default=os.getenv("MILVUS_CONSISTENCY_LEVEL", "Bounded"),
        help="Milvus consistency level (Strong|Session|Bounded|Eventually|Customized).",
    )

    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", default="1,5,10")

    # HNSW sweep (Milvus search param: ef)
    parser.add_argument(
        "--hnsw-ef",
        default="",
        help="Comma-separated HNSW ef values to sweep (e.g. 32,64,128). Leave empty for default.",
    )

    # IVF sweep (Milvus search param: nprobe)
    parser.add_argument(
        "--nprobe",
        default="",
        help="Comma-separated IVF nprobe values to sweep (e.g. 8,16,32). Leave empty for none.",
    )

    parser.add_argument("--load-batch-size", type=int, default=10000)
    args = parser.parse_args()

    distance = args.distance.lower()
    if distance == "ip":
        gt_metric = "dot"
        db_metric = "ip"
    elif distance == "dot":
        gt_metric = "dot"
        db_metric = "ip"
    elif distance in ("l2", "euclid", "euclidean"):
        gt_metric = "l2"
        db_metric = "l2"
    else:
        gt_metric = "cosine"
        db_metric = "cosine"

    k_values = sorted(parse_int_list(args.k_values))
    if not k_values:
        raise ValueError("--k-values cannot be empty")

    hnsw_efs = parse_int_list(args.hnsw_ef) if args.hnsw_ef.strip() else []
    nprobes = parse_int_list(args.nprobe) if args.nprobe.strip() else []

    # Connect Milvus
    connections.connect(alias="default", host=args.host, port=str(args.port))

    if not utility.has_collection(args.collection):
        raise RuntimeError(f"Collection not found: {args.collection}")

    coll = Collection(args.collection)
    coll.load()

    idx_type, idx_metric = infer_index_type_and_metric(coll)
    detected_distance = normalize_distance_name(idx_metric or "")

    print(f"Milvus: {args.host}:{args.port}")
    print(f"Collection: {args.collection}")
    print(f"Index Type (detected): {idx_type}")
    print(f"Index Metric (detected): {idx_metric or 'UNKNOWN'}")
    print(f"Benchmark Distance (requested): {args.distance}")
    print(f"Consistency Level: {args.consistency_level}")

    # Milvus requires query metric to match the index metric. Fail early with guidance.
    requested_distance = normalize_distance_name(args.distance)
    if idx_metric and requested_distance != detected_distance:
        raise ValueError(
            "Requested --distance does not match collection index metric. "
            f"requested={args.distance} (normalized={requested_distance}), "
            f"index_metric={idx_metric} (normalized={detected_distance}). "
            "Use matching --distance or recreate the collection/index with the target metric "
            "(e.g. set MILVUS_METRIC=IP when running milvus/insert-data-milvus.py for dot/IP benchmarking)."
        )

    print("Loading all IDs + vectors from Milvus...")
    t0 = time.perf_counter()
    all_ids, all_vecs = load_all_ids_and_vectors(
        coll,
        batch_size=args.load_batch_size,
        consistency_level=args.consistency_level,
    )
    t1 = time.perf_counter()

    n, dim = all_vecs.shape
    print(f"Loaded vectors: {n} | dim: {dim} | load time: {t1 - t0:.2f}s")

    # Query sampling
    rng = np.random.default_rng(args.seed)
    num_queries = min(args.num_queries, n)
    query_indices = rng.choice(n, size=num_queries, replace=False)

    # Decide sweep plan
    runs = []

    # If user specified both hnsw_ef and nprobe, sweep combinations.
    if hnsw_efs and nprobes:
        for ef in hnsw_efs:
            for npb in nprobes:
                runs.append(("hnsw_ef+nprobe", (ef, npb)))
    elif hnsw_efs:
        for ef in hnsw_efs:
            runs.append(("hnsw_ef", ef))
    elif nprobes:
        for npb in nprobes:
            runs.append(("nprobe", npb))
    else:
        runs.append(("search_params", "default"))

    for label, value in runs:
        if label == "hnsw_ef":
            hnsw_ef = int(value)
            nprobe = None
            ann_label = "hnsw_ef"
            ann_value = hnsw_ef
        elif label == "nprobe":
            hnsw_ef = None
            nprobe = int(value)
            ann_label = "nprobe"
            ann_value = nprobe
        elif label == "hnsw_ef+nprobe":
            hnsw_ef, nprobe = int(value[0]), int(value[1])
            ann_label = "hnsw_ef+nprobe"
            ann_value = f"{hnsw_ef}, {nprobe}"
        else:
            hnsw_ef = None
            nprobe = None
            ann_label = "search_params"
            ann_value = "default"

        recall_map, latency_ms = run_benchmark_once(
            coll=coll,
            all_ids=all_ids,
            all_vecs=all_vecs,
            query_indices=query_indices,
            k_values=k_values,
            gt_metric=gt_metric,
            db_metric=db_metric,
            consistency_level=args.consistency_level,
            hnsw_ef=hnsw_ef,
            nprobe=nprobe,
        )

        print_run(
            engine_label="Milvus",
            collection_name=args.collection,
            db_metric=db_metric,
            gt_metric=gt_metric,
            num_vectors=n,
            dim=dim,
            num_queries=num_queries,
            k_values=k_values,
            recall_map=recall_map,
            latencies_ms=latency_ms,
            ann_param_label=ann_label,
            ann_param_value=ann_value,
        )

    # Cleanup
    try:
        coll.release()
    except Exception:
        pass
    connections.disconnect("default")


if __name__ == "__main__":
    main()
