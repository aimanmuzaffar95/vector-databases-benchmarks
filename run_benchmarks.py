#!/usr/bin/env python3
"""Unified orchestration CLI for docker, insert, and benchmark workflows."""

from __future__ import annotations

import argparse
import csv
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
    qps_script: str
    docker_compose: str | None


@dataclass(frozen=True)
class BenchmarkRecord:
    benchmark_type: str
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
    qps: str
    latency_qps_avg_ms: str
    latency_qps_p50_ms: str
    latency_qps_p95_ms: str
    latency_qps_p99_ms: str


@dataclass(frozen=True)
class InsertRecord:
    db_name: str
    rows: str
    embedding_source: str
    embedding_time_s: str
    write_time_s: str
    build_time_s: str


DB_CONFIG: dict[str, DBConfig] = {
    "pgvector": DBConfig(
        insert_script="pgvector/insert-data-pgvector.py",
        benchmark_script="pgvector/benchmark-pgvector.py",
        qps_script="pgvector/benchmark-qps-pgvector.py",
        docker_compose="docker-compose-pgvector.yml",
    ),
    "chroma": DBConfig(
        insert_script="chroma/insert-data-chroma.py",
        benchmark_script="chroma/benchmark-chroma.py",
        qps_script="chroma/benchmark-qps-chroma.py",
        docker_compose="docker-compose-chroma.yml",
    ),
    "qdrant": DBConfig(
        insert_script="qdrant/insert-data-qdrant.py",
        benchmark_script="qdrant/benchmark-qdrant.py",
        qps_script="qdrant/benchmark-qps-qdrant.py",
        docker_compose="docker-compose-qdrant.yml",
    ),
    "weaviate": DBConfig(
        insert_script="weaviate/insert-data-weaviate.py",
        benchmark_script="weaviate/benchmark-weaviate.py",
        qps_script="weaviate/benchmark-qps-weaviate.py",
        docker_compose="docker-compose-weviate.yml",
    ),
    "milvus": DBConfig(
        insert_script="milvus/insert-data-milvus.py",
        benchmark_script="milvus/benchmark-milvus.py",
        qps_script="milvus/benchmark-qps-milvus.py",
        docker_compose="docker-compose-milvus.yml",
    ),
    "faiss": DBConfig(
        insert_script="faiss/insert-data-faiss.py",
        benchmark_script="faiss/benchmark-faiss.py",
        qps_script="faiss/benchmark-qps-faiss.py",
        docker_compose=None,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run docker, insert, recall benchmark, and QPS workflows across one or more vector DB backends."
    )
    parser.add_argument("-dbname", "--dbname", required=True, help="DB name(s): qdrant,pgvector or all")
    parser.add_argument("--insert", action="store_true", help="Run insert script(s)")
    parser.add_argument("--recall", action="store_true", help="Run recall benchmark script(s)")
    parser.add_argument("--qps", action="store_true", help="Run QPS benchmark script(s)")
    parser.add_argument("--docker", action="store_true", help="Start DB container(s) with docker compose")
    parser.add_argument(
        "--insert-args",
        default="",
        help='Extra args forwarded only to insert scripts. Example: "--distance cosine --batch-size 128"',
    )
    parser.add_argument(
        "--recall-args",
        default="",
        help='Extra args forwarded only to recall benchmark scripts. Example: "--k-values 1,5 --num-queries 100"',
    )
    parser.add_argument(
        "--qps-args",
        default="",
        help='Extra args forwarded only to QPS benchmark scripts. Example: "--k 10 --seconds 20 --concurrency 8"',
    )
    parser.add_argument(
        "--benchmark-csv-path",
        default="benchmark.csv",
        help="Output path for consolidated benchmark table as CSV (default: benchmark_results.csv)",
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


def get_cli_arg_value(args: list[str], flag: str) -> str | None:
    for idx, token in enumerate(args):
        if token == flag:
            if idx + 1 < len(args):
                return args[idx + 1]
            return None
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


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


def classify_docker_error(output: str) -> str:
    text = output.lower()
    if (
        "cannot connect to the docker daemon" in text
        or "is the docker daemon running" in text
        or "error during connect" in text
    ):
        return "FAIL (Docker daemon is not running)"
    if "command not found" in text and "docker" in text:
        return "FAIL (Docker CLI not found)"
    return "FAIL (Docker command failed; see logs above)"


def is_fatal_docker_status(status: str) -> bool:
    return (
        "Docker daemon is not running" in status
        or "Docker CLI not found" in status
    )


def check_docker_container_status(db_name: str, cfg: DBConfig) -> tuple[bool, str]:
    if cfg.docker_compose is None:
        return True, "N/A (local backend)"

    cmd = ["docker", "compose", "-f", cfg.docker_compose, "ps", "--status", "running", "--services"]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        return False, "FAIL (Docker CLI not found)"

    output = result.stdout or ""
    if result.returncode != 0:
        return False, classify_docker_error(output)

    services = [line.strip() for line in output.splitlines() if line.strip()]
    if not services:
        return (
            False,
            f"FAIL (Container not running; start with: docker compose -f {cfg.docker_compose} up -d)",
        )
    return True, f"OK ({', '.join(services)} running)"


def ensure_embedding_cache(insert_args: list[str]) -> tuple[bool, str]:
    if has_cli_flag(insert_args, "--no-embedding-cache"):
        return True, "SKIPPED (no-embedding-cache)"

    csv_path = get_cli_arg_value(insert_args, "--csv-path") or "train.csv"
    model = get_cli_arg_value(insert_args, "--model") or "intfloat/e5-large-v2"
    cache_dir = get_cli_arg_value(insert_args, "--cache-dir") or ".cache/embeddings"
    force_rebuild = has_cli_flag(insert_args, "--force-rebuild-embeddings")

    cmd = [
        sys.executable,
        "shared/precompute_embeddings.py",
        "--csv-path",
        csv_path,
        "--model",
        model,
        "--expected-dim",
        "1024",
        "--cache-dir",
        cache_dir,
    ]
    if force_rebuild:
        cmd.append("--force-rebuild")

    ok, status, _ = run_step("shared", "precompute", cmd)
    return ok, status


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
        qps = ""
        recall_map: dict[str, str] = {}
        latency_avg_ms = ""
        latency_p50_ms = ""
        latency_p95_ms = ""
        latency_p99_ms = ""

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
            elif s.startswith("QPS: "):
                qps = s.split(":", 1)[1].strip()
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
            elif s.startswith("Latency p99: "):
                latency_p99_ms = s.split(":", 1)[1].replace("ms", "").strip()
            j += 1

        if run:
            is_qps_run = bool(qps)
            records.append(
                BenchmarkRecord(
                    benchmark_type="qps" if is_qps_run else "recall",
                    db_name=db_name,
                    run=run,
                    distance=distance or "-",
                    measured_queries=measured_queries or "-",
                    recall_at_1=recall_map.get("1", "-") if not is_qps_run else "-",
                    recall_at_5=recall_map.get("5", "-") if not is_qps_run else "-",
                    recall_at_10=recall_map.get("10", "-") if not is_qps_run else "-",
                    latency_avg_ms=(latency_avg_ms or "-") if not is_qps_run else "-",
                    latency_p50_ms=(latency_p50_ms or "-") if not is_qps_run else "-",
                    latency_p95_ms=(latency_p95_ms or "-") if not is_qps_run else "-",
                    qps=qps or "-",
                    latency_qps_avg_ms=(latency_avg_ms or "-") if is_qps_run else "-",
                    latency_qps_p50_ms=(latency_p50_ms or "-") if is_qps_run else "-",
                    latency_qps_p95_ms=(latency_p95_ms or "-") if is_qps_run else "-",
                    latency_qps_p99_ms=(latency_p99_ms or "-") if is_qps_run else "-",
                )
            )

        i = j

    return records


def _extract_value(line: str) -> str:
    if ":" not in line:
        return ""
    return line.split(":", 1)[1].strip()


def parse_insert_record(db_name: str, output: str) -> InsertRecord | None:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]

    rows = "-"
    embedding_source = "-"
    embedding_time_s = "-"
    write_time_s = "-"
    build_time_s = "-"

    for line in lines:
        if line.startswith("Rows inserted"):
            rows = _extract_value(line)
        elif line.startswith("Rows indexed"):
            rows = _extract_value(line)
        elif line.startswith("Embedding source"):
            embedding_source = _extract_value(line)
        elif line.startswith("Embedding time (s)"):
            embedding_time_s = _extract_value(line)
        elif line.startswith("DB write time (s)"):
            write_time_s = _extract_value(line)
        elif line.startswith("Index build time (s)"):
            build_time_s = _extract_value(line)
        elif line.startswith("FAISS build/add time (s)"):
            build_time_s = _extract_value(line)

    if rows == "-" and embedding_source == "-" and embedding_time_s == "-":
        return None

    return InsertRecord(
        db_name=db_name,
        rows=rows,
        embedding_source=embedding_source,
        embedding_time_s=embedding_time_s,
        write_time_s=write_time_s,
        build_time_s=build_time_s,
    )


def print_insert_table(records: list[InsertRecord]) -> None:
    if not records:
        return

    headers = [
        "DB",
        "Rows",
        "Embedding Source",
        "Embedding time (s)",
        "Write time (s)",
        "Build time (s)",
    ]
    rows = [
        [
            r.db_name,
            r.rows,
            r.embedding_source,
            r.embedding_time_s,
            r.write_time_s,
            r.build_time_s,
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

    print("\nFinal Insert Table")
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))


def build_benchmark_table(records: list[BenchmarkRecord]) -> tuple[list[str], list[list[str]]]:
    if not records:
        return [], []

    has_recall = any(r.benchmark_type == "recall" for r in records)
    has_qps = any(r.benchmark_type == "qps" for r in records)

    headers: list[str] = [
        "DB",
        "Run",
        "Distance",
        "Queries",
    ]
    if has_recall:
        headers.extend(
            [
                "Recall@1",
                "Recall@5",
                "Recall@10",
                "Lat avg (ms)",
                "Lat p50 (ms)",
                "Lat p95 (ms)",
            ]
        )
    if has_qps:
        headers.extend(
            [
                "QPS",
                "Lat-QPS avg (ms)",
                "Lat-QPS p50 (ms)",
                "Lat-QPS p95 (ms)",
                "Lat-QPS p99 (ms)",
            ]
        )

    rows: list[list[str]] = []
    for r in records:
        row = [
            r.db_name,
            r.run,
            r.distance,
            r.measured_queries,
        ]
        if has_recall:
            row.extend(
                [
                    r.recall_at_1,
                    r.recall_at_5,
                    r.recall_at_10,
                    r.latency_avg_ms,
                    r.latency_p50_ms,
                    r.latency_p95_ms,
                ]
            )
        if has_qps:
            row.extend(
                [
                    r.qps,
                    r.latency_qps_avg_ms,
                    r.latency_qps_p50_ms,
                    r.latency_qps_p95_ms,
                    r.latency_qps_p99_ms,
                ]
            )
        rows.append(row)

    return headers, rows


def print_benchmark_table(records: list[BenchmarkRecord]) -> None:
    headers, rows = build_benchmark_table(records)
    if not headers:
        return

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


def _build_single_benchmark_table(records: list[BenchmarkRecord], benchmark_type: str) -> tuple[list[str], list[list[str]]]:
    filtered = [r for r in records if r.benchmark_type == benchmark_type]
    if not filtered:
        return [], []

    headers = ["DB", "Run", "Distance", "Queries"]
    rows: list[list[str]] = []

    if benchmark_type == "recall":
        headers.extend(["Recall@1", "Recall@5", "Recall@10", "Lat avg (ms)", "Lat p50 (ms)", "Lat p95 (ms)"])
        for r in filtered:
            rows.append(
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
            )
    elif benchmark_type == "qps":
        headers.extend(["QPS", "Lat avg (ms)", "Lat p50 (ms)", "Lat p95 (ms)", "Lat p99 (ms)"])
        for r in filtered:
            rows.append(
                [
                    r.db_name,
                    r.run,
                    r.distance,
                    r.measured_queries,
                    r.qps,
                    r.latency_qps_avg_ms,
                    r.latency_qps_p50_ms,
                    r.latency_qps_p95_ms,
                    r.latency_qps_p99_ms,
                ]
            )
    else:
        return [], []

    return headers, rows


def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    if not headers:
        return

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)

    print(f"\n{title}")
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))


def print_benchmark_tables(records: list[BenchmarkRecord]) -> None:
    recall_headers, recall_rows = _build_single_benchmark_table(records, "recall")
    qps_headers, qps_rows = _build_single_benchmark_table(records, "qps")
    _print_table("Final Recall Benchmark Table", recall_headers, recall_rows)
    _print_table("Final QPS Benchmark Table", qps_headers, qps_rows)


def write_benchmark_csv(records: list[BenchmarkRecord], csv_path: str) -> None:
    headers, rows = build_benchmark_table(records)
    if not headers:
        return

    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Benchmark CSV written: {output_path}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not (args.docker or args.insert or args.recall or args.qps):
        parser.error("At least one action flag is required: --docker and/or --insert and/or --recall and/or --qps")

    try:
        selected_dbs = parse_db_selection(args.dbname)
    except ValueError as err:
        parser.error(str(err))

    try:
        insert_args = shlex.split(args.insert_args)
    except ValueError as err:
        parser.error(f"Invalid --insert-args value: {err}")

    try:
        recall_args = shlex.split(args.recall_args)
    except ValueError as err:
        parser.error(f"Invalid --recall-args value: {err}")
    try:
        qps_args = shlex.split(args.qps_args)
    except ValueError as err:
        parser.error(f"Invalid --qps-args value: {err}")

    if not has_cli_flag(insert_args, "--distance"):
        insert_args.extend(["--distance", "cosine"])
    if not has_cli_flag(recall_args, "--distance"):
        recall_args.extend(["--distance", "cosine"])
    if not has_cli_flag(recall_args, "--num-queries"):
        recall_args.extend(["--num-queries", "480"])
    if not has_cli_flag(qps_args, "--distance"):
        qps_args.extend(["--distance", "cosine"])

    actions: list[str] = []
    if args.docker:
        actions.append("docker")
    if args.insert:
        actions.append("insert")
    if args.recall:
        actions.append("recall")
    if args.qps:
        actions.append("qps")

    print("Selected DBs:", ", ".join(selected_dbs))
    print("Selected actions:", ", ".join(actions))
    print("")

    summary: list[tuple[str, str, str]] = []
    insert_records: list[InsertRecord] = []
    benchmark_records: list[BenchmarkRecord] = []
    any_failed = False

    for db_name in selected_dbs:
        cfg = DB_CONFIG[db_name]
        can_run_db_steps = True
        skip_reason = "SKIPPED (docker unavailable for this DB)"

        if (args.insert or args.recall or args.qps) and not args.docker:
            ok, docker_status = check_docker_container_status(db_name, cfg)
            summary.append((db_name, "docker", docker_status))
            if not ok:
                any_failed = True
                print(f"[{db_name}] docker: {docker_status}")
                if is_fatal_docker_status(docker_status):
                    break
                can_run_db_steps = False

        if args.insert and db_name == selected_dbs[0]:
            ok, status = ensure_embedding_cache(insert_args)
            summary.append(("shared", "precompute", status))
            if not ok:
                any_failed = True
                break

        if args.docker:
            if cfg.docker_compose is None:
                msg = "N/A (local backend)"
                print(f"[{db_name}] docker: {msg}")
                summary.append((db_name, "docker", msg))
            else:
                cmd = ["docker", "compose", "-f", cfg.docker_compose, "up", "-d"]
                ok, status, output = run_step(db_name, "docker", cmd, capture_output=True)
                if not ok:
                    status = classify_docker_error(output)
                summary.append((db_name, "docker", status))
                if not ok:
                    any_failed = True
                    print(f"[{db_name}] docker: {status}")
                    if is_fatal_docker_status(status):
                        break
                    can_run_db_steps = False

        if not can_run_db_steps:
            if args.insert:
                summary.append((db_name, "insert", skip_reason))
            if args.recall:
                summary.append((db_name, "recall", skip_reason))
            if args.qps:
                summary.append((db_name, "qps", skip_reason))
            continue

        if args.insert:
            cmd = [sys.executable, cfg.insert_script, *insert_args]
            ok, status, output = run_step(db_name, "insert", cmd, capture_output=True)
            summary.append((db_name, "insert", status))
            if output:
                record = parse_insert_record(db_name, output)
                if record is not None:
                    insert_records.append(record)
            if not ok:
                any_failed = True

        if args.recall:
            cmd = [sys.executable, cfg.benchmark_script, *recall_args]
            ok, status, output = run_step(db_name, "recall", cmd, capture_output=True)
            summary.append((db_name, "recall", status))
            if output:
                benchmark_records.extend(parse_benchmark_records(db_name, output))
            if not ok:
                any_failed = True

        if args.qps:
            cmd = [sys.executable, cfg.qps_script, *qps_args]
            ok, status, output = run_step(db_name, "qps", cmd, capture_output=True)
            summary.append((db_name, "qps", status))
            if output:
                benchmark_records.extend(parse_benchmark_records(db_name, output))
            if not ok:
                any_failed = True

    print("\nSummary")
    for db_name, step_name, status in summary:
        print(f"- {db_name:8} | {step_name:9} | {status}")

    print_insert_table(insert_records)
    print_benchmark_tables(benchmark_records)
    write_benchmark_csv(benchmark_records, args.benchmark_csv_path)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
