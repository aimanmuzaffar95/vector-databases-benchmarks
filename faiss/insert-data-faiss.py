#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.embedding_cache import compute_without_cache, load_or_create_cache

CSV_PATH_DEFAULT = "train.csv"
DEFAULT_MODEL_NAME = "intfloat/e5-large-v2"
EXPECTED_DIM = 1024
CACHE_DIR_DEFAULT = os.getenv("EMBEDDING_CACHE_DIR", ".cache/embeddings")


def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a local FAISS index from train.csv embeddings")
    p.add_argument("--csv-path", default=os.getenv("CSV_PATH", CSV_PATH_DEFAULT))
    p.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME))
    p.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    p.add_argument("--force-rebuild-embeddings", action="store_true")
    p.add_argument("--no-embedding-cache", action="store_true")
    p.add_argument(
        "--distance",
        default=os.getenv("FAISS_DISTANCE", "cosine"),
        choices=["cosine", "dot", "l2"],
        help="Distance metric for FAISS index",
    )
    p.add_argument(
        "--index-type",
        default=os.getenv("FAISS_INDEX_TYPE", "hnsw"),
        choices=["flat", "ivfflat", "hnsw"],
        help="FAISS ANN index family",
    )
    p.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "1000")))
    p.add_argument("--ivf-nlist", type=int, default=int(os.getenv("FAISS_IVF_NLIST", "1024")))
    p.add_argument("--hnsw-m", type=int, default=int(os.getenv("FAISS_HNSW_M", "32")))
    p.add_argument(
        "--output-dir",
        default=os.getenv("FAISS_OUTPUT_DIR", "faiss/artifacts"),
        help="Directory where index + vectors + metadata are saved",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files in output-dir")
    return p.parse_args()


def build_index(dim: int, index_type: str, metric_type: int, ivf_nlist: int, hnsw_m: int) -> faiss.Index:
    if index_type == "flat":
        base = faiss.IndexFlat(dim, metric_type)
        return faiss.IndexIDMap2(base)

    if index_type == "ivfflat":
        quantizer = faiss.IndexFlat(dim, metric_type)
        base = faiss.IndexIVFFlat(quantizer, dim, int(ivf_nlist), metric_type)
        return faiss.IndexIDMap2(base)

    if index_type == "hnsw":
        base = faiss.IndexHNSWFlat(dim, int(hnsw_m), metric_type)
        return faiss.IndexIDMap2(base)

    raise ValueError(f"Unsupported index type: {index_type}")


def unwrap_index(idx: faiss.Index) -> faiss.Index:
    cur = idx
    while hasattr(cur, "index"):
        cur = cur.index
    return cur


def main() -> None:
    args = parse_args()

    csv_path = args.csv_path
    model_name = args.model
    distance = args.distance
    index_type = args.index_type
    output_dir = Path(args.output_dir)

    print("Model:", model_name)
    print("Distance:", distance)
    print("Index type:", index_type)
    print("Output dir:", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "index.faiss"
    vectors_path = output_dir / "vectors.npy"
    ids_path = output_dir / "ids.npy"
    metadata_path = output_dir / "metadata.json"

    if not args.overwrite:
        existing = [p for p in [index_path, vectors_path, ids_path, metadata_path] if p.exists()]
        if existing:
            raise RuntimeError(
                "Output files already exist. Use --overwrite to replace them. Existing: "
                + ", ".join(str(p) for p in existing)
            )

    if args.no_embedding_cache:
        bundle = compute_without_cache(csv_path=csv_path, model_name=model_name, expected_dim=EXPECTED_DIM)
    else:
        bundle = load_or_create_cache(
            csv_path=csv_path,
            model_name=model_name,
            expected_dim=EXPECTED_DIM,
            cache_dir=args.cache_dir,
            cache_key_suffix="shared",
            force_rebuild=bool(args.force_rebuild_embeddings),
        )

    vectors = np.asarray(bundle.vectors, dtype=np.float32)
    records_count = vectors.shape[0]

    print(f"Using local dataset: {csv_path}")
    print(f"Records: {records_count} | CSV size: {bundle.csv_size_mb:.2f} MB")
    print(f"Embedding source: {bundle.source}")

    if vectors.ndim != 2 or vectors.shape[1] != EXPECTED_DIM:
        raise RuntimeError(f"Unexpected embedding shape: {vectors.shape}")

    metric_type = faiss.METRIC_L2
    if distance in {"cosine", "dot"}:
        metric_type = faiss.METRIC_INNER_PRODUCT
    if distance == "cosine":
        faiss.normalize_L2(vectors)

    ids = np.arange(records_count, dtype=np.int64)
    index = build_index(
        dim=EXPECTED_DIM,
        index_type=index_type,
        metric_type=metric_type,
        ivf_nlist=args.ivf_nlist,
        hnsw_m=args.hnsw_m,
    )
    base = unwrap_index(index)

    print("Building FAISS index...")
    t_idx0 = time.perf_counter()

    if not base.is_trained:
        base.train(vectors)

    batch_size = int(args.batch_size)
    for start in range(0, records_count, batch_size):
        end = min(start + batch_size, records_count)
        index.add_with_ids(vectors[start:end], ids[start:end])

    index_seconds = time.perf_counter() - t_idx0

    faiss.write_index(index, str(index_path))
    np.save(vectors_path, vectors)
    np.save(ids_path, ids)

    metadata = {
        "csv_path": csv_path,
        "model": model_name,
        "dim": int(EXPECTED_DIM),
        "distance": distance,
        "index_type": index_type,
        "rows": int(records_count),
        "ivf_nlist": int(args.ivf_nlist),
        "hnsw_m": int(args.hnsw_m),
        "normalized": bool(distance == "cosine"),
        "created_unix": time.time(),
        "embedding_source": bundle.source,
        "embedding_seconds": float(bundle.embed_seconds),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    approx_vec_mb = bytes_to_mb(records_count * EXPECTED_DIM * 4)

    print("\n==== SUMMARY ====")
    print(f"Rows indexed               : {index.ntotal}")
    print(f"CSV size (MB)              : {bundle.csv_size_mb:.2f}")
    print(f"Approx embedding data (MB) : {approx_vec_mb:.2f} (float32 vectors only)")
    print(f"Embedding source           : {bundle.source}")
    print(f"Embedding time (s)         : {bundle.embed_seconds:.3f}")
    print(f"FAISS build/add time (s)   : {index_seconds:.3f}")
    print(f"Saved index                : {index_path}")
    print(f"Saved vectors              : {vectors_path}")
    print(f"Saved ids                  : {ids_path}")
    print(f"Saved metadata             : {metadata_path}")
    print("=================\n")


if __name__ == "__main__":
    main()
