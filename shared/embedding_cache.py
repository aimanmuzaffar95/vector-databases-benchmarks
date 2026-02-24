#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}
PREFIX_VERSION = "passage-v1"
CACHE_VERSION = 1


@dataclass
class CacheBundle:
    vectors: np.ndarray
    titles: list[str]
    descriptions: list[str]
    labels: list[str]
    csv_size_mb: float
    embed_seconds: float
    source: str


def bytes_to_mb(size_bytes: int) -> float:
    return size_bytes / (1024 * 1024)


def build_texts(df: pd.DataFrame) -> list[str]:
    return [f"passage: {str(t).strip()} {str(d).strip()}".strip() for t, d in zip(df["title"], df["description"])]


def compute_csv_sha256(csv_path: str) -> str:
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Local dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expected_columns = {"Class Index", "Title", "Description"}
    received_columns = set(df.columns)
    if not expected_columns.issubset(received_columns):
        raise ValueError(f"Missing column(s). Expected: {expected_columns}, Found: {received_columns}")

    df = df.rename(
        columns={
            "Class Index": "label",
            "Title": "title",
            "Description": "description",
        }
    )

    df["label"] = df["label"].map(LABEL_MAP).fillna("Unknown")
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    return df


def validate_cache(
    meta: dict[str, Any],
    csv_sha256: str,
    model_name: str,
    expected_dim: int,
    row_count: int,
    prefix_version: str,
) -> None:
    if int(meta.get("version", -1)) != CACHE_VERSION:
        raise ValueError("Cache version mismatch")
    if meta.get("csv_sha256") != csv_sha256:
        raise ValueError("CSV digest mismatch")
    if meta.get("model_name") != model_name:
        raise ValueError("Model mismatch")
    if int(meta.get("expected_dim", -1)) != int(expected_dim):
        raise ValueError("Embedding dim mismatch")
    if int(meta.get("rows", -1)) != int(row_count):
        raise ValueError("Row count mismatch")
    if meta.get("prefix_version") != prefix_version:
        raise ValueError("Prefix version mismatch")


def _cache_key(csv_sha256: str, model_name: str, expected_dim: int, cache_key_suffix: str) -> str:
    raw = f"{csv_sha256}|{model_name}|{expected_dim}|{PREFIX_VERSION}|{cache_key_suffix}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _cache_paths(cache_dir: str, cache_key: str) -> tuple[Path, Path]:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"embeddings_{cache_key}.npz", root / f"embeddings_{cache_key}.json"


def load_cache(cache_npz_path: str, metadata_json_path: str) -> CacheBundle:
    meta = json.loads(Path(metadata_json_path).read_text(encoding="utf-8"))
    with np.load(cache_npz_path, allow_pickle=True) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        titles = [str(x) for x in data["title"].tolist()]
        descriptions = [str(x) for x in data["description"].tolist()]
        labels = [str(x) for x in data["label"].tolist()]

    if vectors.ndim != 2:
        raise ValueError(f"Invalid cached vectors shape: {vectors.shape}")

    return CacheBundle(
        vectors=vectors,
        titles=titles,
        descriptions=descriptions,
        labels=labels,
        csv_size_mb=float(meta.get("csv_size_mb", float("nan"))),
        embed_seconds=float(meta.get("embed_seconds", float("nan"))),
        source="cache",
    )


def compute_without_cache(csv_path: str, model_name: str, expected_dim: int) -> CacheBundle:
    df = load_dataset(csv_path)
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    if dim != expected_dim:
        raise RuntimeError(
            f"Model dimension is {dim}, expected {expected_dim}. "
            "Pick a 1024-d model (e.g., intfloat/e5-large-v2)."
        )

    texts = build_texts(df)
    t0 = time.perf_counter()
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    embed_seconds = time.perf_counter() - t0

    if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
        raise RuntimeError(f"Unexpected embedding shape: {vectors.shape}")

    return CacheBundle(
        vectors=vectors,
        titles=df["title"].tolist(),
        descriptions=df["description"].tolist(),
        labels=df["label"].tolist(),
        csv_size_mb=bytes_to_mb(os.path.getsize(csv_path)),
        embed_seconds=embed_seconds,
        source="computed",
    )


def load_or_create_cache(
    csv_path: str,
    model_name: str,
    expected_dim: int,
    cache_dir: str,
    cache_key_suffix: str,
    force_rebuild: bool = False,
) -> CacheBundle:
    df = load_dataset(csv_path)
    row_count = len(df)
    csv_sha256 = compute_csv_sha256(csv_path)
    key = _cache_key(csv_sha256, model_name, expected_dim, cache_key_suffix)
    cache_npz, cache_json = _cache_paths(cache_dir, key)

    if (not force_rebuild) and cache_npz.exists() and cache_json.exists():
        try:
            meta = json.loads(cache_json.read_text(encoding="utf-8"))
            validate_cache(meta, csv_sha256, model_name, expected_dim, row_count, PREFIX_VERSION)
            bundle = load_cache(str(cache_npz), str(cache_json))
            if bundle.vectors.shape[0] != row_count or bundle.vectors.shape[1] != expected_dim:
                raise ValueError("Cached vector shape mismatch")
            return bundle
        except Exception:
            pass

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    if dim != expected_dim:
        raise RuntimeError(
            f"Model dimension is {dim}, expected {expected_dim}. "
            "Pick a 1024-d model (e.g., intfloat/e5-large-v2)."
        )

    texts = build_texts(df)
    t0 = time.perf_counter()
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    embed_seconds = time.perf_counter() - t0

    if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
        raise RuntimeError(f"Unexpected embedding shape: {vectors.shape}")

    np.savez_compressed(
        cache_npz,
        vectors=vectors,
        title=np.asarray(df["title"].tolist(), dtype=object),
        description=np.asarray(df["description"].tolist(), dtype=object),
        label=np.asarray(df["label"].tolist(), dtype=object),
    )

    meta = {
        "version": CACHE_VERSION,
        "cache_key": key,
        "csv_path": str(Path(csv_path).resolve()),
        "csv_sha256": csv_sha256,
        "csv_size_mb": bytes_to_mb(os.path.getsize(csv_path)),
        "model_name": model_name,
        "expected_dim": int(expected_dim),
        "rows": int(row_count),
        "prefix_version": PREFIX_VERSION,
        "cache_key_suffix": cache_key_suffix,
        "created_unix": time.time(),
        "embed_seconds": embed_seconds,
    }
    cache_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return CacheBundle(
        vectors=vectors,
        titles=df["title"].tolist(),
        descriptions=df["description"].tolist(),
        labels=df["label"].tolist(),
        csv_size_mb=bytes_to_mb(os.path.getsize(csv_path)),
        embed_seconds=embed_seconds,
        source="computed",
    )
