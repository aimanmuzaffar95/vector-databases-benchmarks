#!/usr/bin/env python3
"""
Benchmark Recall@K for a ChromaDB collection (server or local persistent).

What it does:
1) Loads ids + embeddings from Chroma collection
2) Samples query rows from the same dataset (with fixed seed)
3) Builds exact ground truth top-K in NumPy (excluding self-match)
4) Runs Chroma search for each query vector
5) Computes Recall@K + latency stats (p50/p95/avg)

Notes vs pgvector version:
- Chroma query metric is typically set at collection creation (e.g., hnsw:space).
- Chroma doesn't support per-query tuning knobs like ivfflat.probes or hnsw.ef_search via API.
  The CLI args remain accepted but are ignored with a warning to keep compatibility.

Usage examples:
  python3 benchmark_recall_chroma.py
  python3 benchmark_recall_chroma.py --k-values 1,5,10 --num-queries 480
  python3 benchmark_recall_chroma.py --distance cosine
  python3 benchmark_recall_chroma.py --where-json '{"genre":"sports"}'
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import chromadb
except ImportError as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install chromadb"
    ) from e
CHROMA_MODE = (  # "http" (server) or "persistent" (local disk)
    __import__("os").getenv("CHROMA_MODE", "http").strip().lower()
)

CHROMA_HOST = __import__("os").getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(__import__("os").getenv("CHROMA_PORT", "8000"))

# Used only if CHROMA_MODE="persistent"
CHROMA_PERSIST_DIR = __import__("os").getenv("CHROMA_PERSIST_DIR", "./chroma_data")

COLLECTION_NAME = __import__("os").getenv("COLLECTION_NAME", "news_articles")

# If your Chroma IDs are not ints, leave as "string"
ID_TYPE = __import__("os").getenv("ID_TYPE", "int").strip().lower()  # "int" or "string"


@dataclass
class RowVec:
    row_id: int | str
    vector: np.ndarray


@dataclass
class SearchResult:
    ids: list[int | str]
    latency_ms: float


@dataclass
class BenchmarkRun:
    label: str
    metric: str
    k_values: list[int]
    num_queries: int
    avg_recall_at_k: dict[int, float]
    p50_latency_ms: float
    p95_latency_ms: float
    avg_latency_ms: float
def parse_k_values(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k <= 0:
            raise ValueError(f"K must be > 0, got {k}")
        values.append(k)
    if not values:
        raise ValueError("No k-values provided")
    return sorted(set(values))


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


def recall_at_k(pred_ids: list[int | str], gt_ids: list[int | str], k: int) -> float:
    gt_topk = gt_ids[:k]
    if not gt_topk:
        return 0.0
    pred_topk = pred_ids[:k]
    hit_count = len(set(pred_topk) & set(gt_topk))
    return hit_count / len(gt_topk)
def maybe_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def build_matrix(rows: list[RowVec]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array([r.row_id for r in rows], dtype=object)
    mat = np.stack([r.vector for r in rows]).astype(np.float32, copy=False)
    return ids, mat


def exact_topk_ids(
    *,
    base_ids: np.ndarray,
    base_matrix: np.ndarray,
    query_vector: np.ndarray,
    metric: str,
    k: int,
    exclude_id: int | str | None = None,
) -> list[int | str]:
    metric = metric.lower()
    q = query_vector.astype(np.float32, copy=False)

    if metric == "cosine":
        qn = q / (np.linalg.norm(q) or 1.0)
        bn = maybe_normalize(base_matrix)
        scores = bn @ qn
        order = np.argsort(-scores)
    elif metric in {"ip", "inner_product", "inner-product"}:
        scores = base_matrix @ q
        order = np.argsort(-scores)
    elif metric == "l2":
        diffs = base_matrix - q
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        order = np.argsort(d2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    result_ids: list[int | str] = []
    for idx in order:
        rid = base_ids[idx]
        if exclude_id is not None and rid == exclude_id:
            continue
        result_ids.append(rid)
        if len(result_ids) >= k:
            break
    return result_ids
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
        except ValueError:
            raise ValueError(
                f"ID_TYPE=int but found non-int id in Chroma: {x!r}. "
                f"Set env ID_TYPE=string or ensure ids are integers."
            )
    return x


def load_all_vectors(
    collection,
    *,
    where: dict[str, Any] | None = None,
    limit: int | None = None,
) -> list[RowVec]:
    """
    Loads all ids + embeddings from Chroma.
    If limit is set, we still fetch all then slice (Chroma get pagination differs by version).
    """
    # Chroma returns ids as list[str] and embeddings as list[list[float]]
    data = collection.get(include=["embeddings"], where=where)
    ids_raw = data.get("ids")
    embs_raw = data.get("embeddings")

    if ids_raw is None or embs_raw is None:
        raise RuntimeError("No vectors loaded from Chroma collection.")
    if len(ids_raw) == 0 or len(embs_raw) == 0:
        raise RuntimeError("No vectors loaded from Chroma collection.")

    rows_out: list[RowVec] = []
    for rid, emb in zip(ids_raw, embs_raw):
        row_id = coerce_id(rid)
        vec = np.asarray(emb, dtype=np.float32)
        if vec.size == 0 or not np.all(np.isfinite(vec)):
            raise ValueError(f"Invalid embedding for id={rid!r}")
        rows_out.append(RowVec(row_id=row_id, vector=vec))

    if limit is not None:
        rows_out = rows_out[: int(limit)]

    if not rows_out:
        raise RuntimeError("No vectors loaded from Chroma (after limit).")
    return rows_out


def search_chroma(
    collection,
    *,
    query_vector: np.ndarray,
    limit: int,
    exclude_id: int | str | None = None,
    where: dict[str, Any] | None = None,
) -> SearchResult:
    """
    Chroma doesn't support exclude_id directly.
    We request limit+1 and remove exclude_id if present, then truncate.
    """
    n_results = int(limit) + (1 if exclude_id is not None else 0)

    t0 = time.perf_counter()
    res = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=n_results,
        where=where,
        include=["distances"],  # doesn't change output format; useful for debugging
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    ids = (res.get("ids") or [[]])[0]
    # coerce ids to int/str consistently
    ids2: list[int | str] = [coerce_id(x) for x in ids]

    if exclude_id is not None:
        ids2 = [x for x in ids2 if x != exclude_id]

    ids2 = ids2[: int(limit)]
    return SearchResult(ids=ids2, latency_ms=latency_ms)
def benchmark_once(
    *,
    collection,
    rows: list[RowVec],
    metric: str,
    k_values: list[int],
    num_queries: int,
    seed: int,
    warmup_queries: int,
    where: dict[str, Any] | None = None,
    # kept for CLI compatibility; ignored
    ivfflat_probes: int | None = None,
    hnsw_ef_search: int | None = None,
) -> BenchmarkRun:
    if not rows:
        raise ValueError("rows cannot be empty")

    if ivfflat_probes is not None or hnsw_ef_search is not None:
        print(
            "[WARN] Chroma does not expose per-query ivfflat.probes or hnsw.ef_search. "
            "These parameters are ignored."
        )

    max_k = max(k_values)
    if len(rows) <= max_k + 1:
        raise ValueError(f"Need more rows than max_k+1. rows={len(rows)}, max_k={max_k}")

    base_ids, base_matrix = build_matrix(rows)

    rng = random.Random(seed)
    sample_idx = list(range(len(rows)))
    rng.shuffle(sample_idx)
    sample_idx = sample_idx[: min(num_queries, len(rows))]
    if not sample_idx:
        raise ValueError("No query rows selected")

    gt_by_query_id: dict[int | str, list[int | str]] = {}
    for i in sample_idx:
        qr = rows[i]
        gt_ids = exact_topk_ids(
            base_ids=base_ids,
            base_matrix=base_matrix,
            query_vector=qr.vector,
            metric=metric,
            k=max_k,
            exclude_id=qr.row_id,
        )
        gt_by_query_id[qr.row_id] = gt_ids

    warmup_count = min(warmup_queries, len(sample_idx))
    measured_indices = sample_idx[warmup_count:]

    latencies: list[float] = []
    recalls: dict[int, list[float]] = {k: [] for k in k_values}

    for j, i in enumerate(sample_idx):
        qr = rows[i]
        result = search_chroma(
            collection,
            query_vector=qr.vector,
            limit=max_k,
            exclude_id=qr.row_id,
            where=where,
        )

        if j < warmup_count:
            continue

        latencies.append(result.latency_ms)
        gt_ids = gt_by_query_id[qr.row_id]
        for k in k_values:
            recalls[k].append(recall_at_k(result.ids, gt_ids, k))

    if not latencies:
        raise RuntimeError("No measured queries after warm-up. Reduce warmup_queries.")

    avg_recall_at_k = {
        k: float(sum(vals) / len(vals)) if vals else float("nan")
        for k, vals in recalls.items()
    }

    label = "default"
    return BenchmarkRun(
        label=label,
        metric=metric,
        k_values=k_values,
        num_queries=len(measured_indices),
        avg_recall_at_k=avg_recall_at_k,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        avg_latency_ms=float(statistics.mean(latencies)),
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
    p = argparse.ArgumentParser(description="Benchmark Recall@K for ChromaDB")
    p.add_argument(
        "--distance",
        default="cosine",
        choices=["cosine", "l2", "ip"],
        help="Distance metric for exact ground truth",
    )
    p.add_argument("--k-values", default="1,5,10", help="Comma-separated K values, e.g. 1,5,10")
    p.add_argument("--num-queries", type=int, default=480, help="Number of query rows to sample")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--warmup-queries", type=int, default=0, help="Warm-up queries excluded from stats")
    p.add_argument("--max-load-items", type=int, default=None, help="Optional cap when loading items (debugging)")

    # Chroma filter (recommended instead of SQL)
    p.add_argument(
        "--where-json",
        type=str,
        default=None,
        help='Optional Chroma "where" filter as JSON. Example: \'{"genre":"sports"}\'',
    )

    # Keep these args for compatibility; ignored in Chroma
    p.add_argument("--ivfflat-probes", type=str, default=None, help="Ignored for Chroma (kept for compatibility)")
    p.add_argument("--hnsw-ef-search", type=str, default=None, help="Ignored for Chroma (kept for compatibility)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be > 0")
    if args.warmup_queries < 0:
        raise ValueError("--warmup-queries must be >= 0")
    where = None
    if args.where_json:
        try:
            where = json.loads(args.where_json)
            if not isinstance(where, dict):
                raise ValueError("where-json must decode to a JSON object (dict).")
        except Exception as e:
            raise ValueError(f"Invalid --where-json: {e}") from e
    ivf_sweep = parse_int_list(args.ivfflat_probes)
    hnsw_sweep = parse_int_list(args.hnsw_ef_search)
    if ivf_sweep and hnsw_sweep:
        raise ValueError("Choose either --ivfflat-probes or --hnsw-ef-search, not both")
    if ivf_sweep or hnsw_sweep:
        print("[WARN] Sweep params provided, but Chroma ignores ivfflat/hnsw tuning knobs via API.")

    client = get_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    print("Loading vectors from database...")
    rows = load_all_vectors(collection, where=where, limit=args.max_load_items)
    dim = int(rows[0].vector.shape[0])
    print(f"Loaded {len(rows)} vectors | dim={dim} | collection={COLLECTION_NAME}")
    for r in rows:
        if r.vector.shape[0] != dim:
            raise ValueError(
                f"Inconsistent vector dims. Row id={r.row_id} has dim={r.vector.shape[0]} expected={dim}"
            )

    run = benchmark_once(
        collection=collection,
        rows=rows,
        metric=args.distance,
        k_values=k_values,
        num_queries=args.num_queries,
        seed=args.seed,
        warmup_queries=args.warmup_queries,
        where=where,
    )

    print_run(run)
    print("\nDone.")


if __name__ == "__main__":
    main()
