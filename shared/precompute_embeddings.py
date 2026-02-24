#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.embedding_cache import load_or_create_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute reusable embedding cache")
    p.add_argument("--csv-path", default=os.getenv("CSV_PATH", "train.csv"))
    p.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2"))
    p.add_argument("--expected-dim", type=int, default=1024)
    p.add_argument("--cache-dir", default=".cache/embeddings")
    p.add_argument("--force-rebuild", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_or_create_cache(
        csv_path=args.csv_path,
        model_name=args.model,
        expected_dim=int(args.expected_dim),
        cache_dir=args.cache_dir,
        cache_key_suffix="shared",
        force_rebuild=bool(args.force_rebuild),
    )
    print(f"Embedding source: {bundle.source}")
    print(f"Embedding time (s): {bundle.embed_seconds:.3f}")
    print(f"Rows: {bundle.vectors.shape[0]} | Dim: {bundle.vectors.shape[1]}")


if __name__ == "__main__":
    main()
