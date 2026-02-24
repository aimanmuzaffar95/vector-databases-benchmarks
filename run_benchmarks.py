#!/usr/bin/env python3
"""Unified orchestration CLI for docker, insert, and benchmark workflows."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DB_ORDER = ["pgvector", "chroma", "qdrant", "weaviate", "milvus", "faiss"]


@dataclass(frozen=True)
class DBConfig:
    insert_script: str
    benchmark_script: str
    docker_compose: str | None


@dataclass(frozen=True)
class BenchmarkRecord:
    db_name: str
    run: str
    distance: str
    measured_queries: str
    recall_at_1: str
    recall_at_5: str
    recall_at_10: str
    latency_avg_ms: str
    latency_p50_ms: str
    latency_p95_ms: str


DB_CONFIG: dict[str, DBConfig] = {
    "pgvector": DBConfig(
        insert_script="pgvector/insert-data-pgvector.py",
        benchmark_script="pgvector/benchmark-pgvector.py",
        docker_compose="docker-compose-pgvector.yml",
    ),
    "chroma": DBConfig(
        insert_script="chroma/insert-data-chroma.py",
        benchmark_script="chroma/benchmark-chroma.py",
        docker_compose="docker-compose-chroma.yml",
    ),
    "qdrant": DBConfig(
        insert_script="qdrant/insert-data-qdrant.py",
        benchmark_script="qdrant/benchmark-qdrant.py",
        docker_compose="docker-compose-qdrant.yml",
    ),
    "weaviate": DBConfig(
        insert_script="weaviate/insert-data-weaviate.py",
        benchmark_script="weaviate/benchmark-weaviate.py",
        docker_compose="docker-compose-weviate.yml",
    ),
    "milvus": DBConfig(
        insert_script="milvus/insert-data-milvus.py",
        benchmark_script="milvus/benchmark-milvus.py",
        docker_compose="docker-compose-milvus.yml",
    ),
    "faiss": DBConfig(
        insert_script="faiss/insert-data-faiss.py",
        benchmark_script="faiss/benchmark-faiss.py",
        docker_compose=None,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run docker, insert, and benchmark across one or more vector DB backends."
    )
    parser.add_argument("-dbname", "--dbname", required=True, help="DB name(s): qdrant,pgvector or all")
    parser.add_argument("--insert", action="store_true", help="Run insert script(s)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark script(s)")
    parser.add_argument("--docker", action="store_true", help="Start DB container(s) with docker compose")
    parser.add_argument(
        "--insert-args",
        default="",
        help='Extra args forwarded only to insert scripts. Example: "--distance cosine --batch-size 128"',
    )
    parser.add_argument(
        "--benchmark-args",
        default="",
        help='Extra args forwarded only to benchmark scripts. Example: "--k-values 1,5 --num-queries 100"',
    )
    return parser


def parse_db_selection(dbname_raw: str) -> list[str]:
    tokens = [token.strip().lower() for token in dbname_raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("No DB names provided in --dbname.")

    if "all" in tokens:
        if len(tokens) > 1:
            raise ValueError("'all' cannot be combined with explicit DB names.")
        return list(DB_ORDER)

    deduped: list[str] = []
    unknown: list[str] = []
    for name in tokens:
        if name not in DB_CONFIG:
            unknown.append(name)
            continue
        if name not in deduped:
            deduped.append(name)

    if unknown:
        valid = ", ".join(DB_ORDER + ["all"])
        raise ValueError(f"Unknown DB name(s): {', '.join(unknown)}. Valid values: {valid}")

    return deduped


def command_to_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def has_cli_flag(args: list[str], flag: str) -> bool:
    for token in args:
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


def run_step(
    db_name: str,
    step_name: str,
    cmd: list[str],
    *,
    capture_output: bool = False,
) -> tuple[bool, str, str]:
    print(f"[{db_name}] {step_name} command: {command_to_text(cmd)}")
    if capture_output:
        result = subprocess.run(
            cmd,
            check=False,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = result.stdout or ""
        if output:
            # Keep script output visible in real time logs while still allowing post-parse.
            print(output, end="" if output.endswith("\n") else "\n")
    else:
        result = subprocess.run(cmd, check=False, cwd=ROOT)
        output = ""

    ok = result.returncode == 0
    status = "OK" if ok else f"FAIL (exit={result.returncode})"
    print(f"[{db_name}] {step_name}: {status}")
    return ok, status, output


_RECALL_RE = re.compile(r"^Recall@(\d+):\s*(.+)$")


def parse_benchmark_records(db_name: str, output: str) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    lines = output.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "Benchmark Results":
            i += 1
            continue

        run = ""
        distance = ""
        measured_queries = ""
        recall_map: dict[str, str] = {}
        latency_avg_ms = ""
        latency_p50_ms = ""
        latency_p95_ms = ""

        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if s == "Benchmark Results":
                break
            if s.startswith("Run: "):
                run = s.split(":", 1)[1].strip()
            elif s.startswith("Distance: "):
                distance = s.split(":", 1)[1].strip()
            elif s.startswith("Measured queries: "):
                measured_queries = s.split(":", 1)[1].strip()
            elif s.startswith("Recall@"):
                m = _RECALL_RE.match(s)
                if m:
                    recall_map[m.group(1)] = m.group(2).strip()
            elif s.startswith("Latency avg: "):
                latency_avg_ms = s.split(":", 1)[1].replace("ms", "").strip()
            elif s.startswith("Latency p50: "):
                latency_p50_ms = s.split(":", 1)[1].replace("ms", "").strip()
            elif s.startswith("Latency p95: "):
                latency_p95_ms = s.split(":", 1)[1].replace("ms", "").strip()
            j += 1

        if run:
            records.append(
                BenchmarkRecord(
                    db_name=db_name,
                    run=run,
                    distance=distance or "-",
                    measured_queries=measured_queries or "-",
                    recall_at_1=recall_map.get("1", "-"),
                    recall_at_5=recall_map.get("5", "-"),
                    recall_at_10=recall_map.get("10", "-"),
                    latency_avg_ms=latency_avg_ms or "-",
                    latency_p50_ms=latency_p50_ms or "-",
                    latency_p95_ms=latency_p95_ms or "-",
                )
            )

        i = j

    return records


def print_benchmark_table(records: list[BenchmarkRecord]) -> None:
    if not records:
        return

    headers = [
        "DB",
        "Run",
        "Distance",
        "Queries",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Lat avg (ms)",
        "Lat p50 (ms)",
        "Lat p95 (ms)",
    ]
    rows = [
        [
            r.db_name,
            r.run,
            r.distance,
            r.measured_queries,
            r.recall_at_1,
            r.recall_at_5,
            r.recall_at_10,
            r.latency_avg_ms,
            r.latency_p50_ms,
            r.latency_p95_ms,
        ]
        for r in records
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)

    print("\nFinal Benchmark Table")
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not (args.docker or args.insert or args.benchmark):
        parser.error("At least one action flag is required: --docker and/or --insert and/or --benchmark")

    try:
        selected_dbs = parse_db_selection(args.dbname)
    except ValueError as err:
        parser.error(str(err))

    try:
        insert_args = shlex.split(args.insert_args)
    except ValueError as err:
        parser.error(f"Invalid --insert-args value: {err}")

    try:
        benchmark_args = shlex.split(args.benchmark_args)
    except ValueError as err:
        parser.error(f"Invalid --benchmark-args value: {err}")

    if not has_cli_flag(insert_args, "--distance"):
        insert_args.extend(["--distance", "cosine"])
    if not has_cli_flag(benchmark_args, "--distance"):
        benchmark_args.extend(["--distance", "cosine"])
    if not has_cli_flag(benchmark_args, "--num-queries"):
        benchmark_args.extend(["--num-queries", "480"])

    actions: list[str] = []
    if args.docker:
        actions.append("docker")
    if args.insert:
        actions.append("insert")
    if args.benchmark:
        actions.append("benchmark")

    print("Selected DBs:", ", ".join(selected_dbs))
    print("Selected actions:", ", ".join(actions))
    print("")

    summary: list[tuple[str, str, str]] = []
    benchmark_records: list[BenchmarkRecord] = []
    any_failed = False

    for db_name in selected_dbs:
        cfg = DB_CONFIG[db_name]

        if args.docker:
            if cfg.docker_compose is None:
                msg = "SKIPPED (local backend)"
                print(f"[{db_name}] docker: {msg}")
                summary.append((db_name, "docker", msg))
            else:
                cmd = ["docker", "compose", "-f", cfg.docker_compose, "up", "-d"]
                ok, status, _ = run_step(db_name, "docker", cmd)
                summary.append((db_name, "docker", status))
                if not ok:
                    any_failed = True

        if args.insert:
            cmd = [sys.executable, cfg.insert_script, *insert_args]
            ok, status, _ = run_step(db_name, "insert", cmd)
            summary.append((db_name, "insert", status))
            if not ok:
                any_failed = True

        if args.benchmark:
            cmd = [sys.executable, cfg.benchmark_script, *benchmark_args]
            ok, status, output = run_step(db_name, "benchmark", cmd, capture_output=True)
            summary.append((db_name, "benchmark", status))
            if output:
                benchmark_records.extend(parse_benchmark_records(db_name, output))
            if not ok:
                any_failed = True

    print("\nSummary")
    for db_name, step_name, status in summary:
        print(f"- {db_name:8} | {step_name:9} | {status}")

    print_benchmark_table(benchmark_records)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
