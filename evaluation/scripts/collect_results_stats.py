#!/usr/bin/env python
"""
Collect extended statistics from an evaluation results directory.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional


STAGE_BY_NUMBER = {
    "1": "add",
    "2": "search",
    "3": "answer",
    "4": "evaluate",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_stats(values: List[float], *, scale: float = 1.0) -> Optional[Dict[str, float]]:
    if not values:
        return None
    scaled = [v * scale for v in values]
    sorted_vals = sorted(scaled)
    return {
        "count": len(values),
        "min": round(float(sorted_vals[0]), 3),
        "max": round(float(sorted_vals[-1]), 3),
        "mean": round(float(mean(scaled)), 3),
        "median": round(float(median(scaled)), 3),
        "p95": round(float(sorted_vals[int(0.95 * (len(values) - 1))]), 3),
    }


def _parse_pipeline_log(path: Path) -> Dict[str, List[float]]:
    stage_durations: Dict[str, List[float]] = {}
    start_times: Dict[str, datetime] = {}

    start_re = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d{3}).*Starting Stage (?P<num>\d+): (?P<name>\w+)",
    )
    end_re = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d{3}).*Stage (?P<num>\d+) completed",
    )

    for line in path.read_text(encoding="utf-8").splitlines():
        start_match = start_re.match(line)
        if start_match:
            ts = datetime.strptime(
                f"{start_match.group('ts')},{start_match.group('ms')}",
                "%Y-%m-%d %H:%M:%S,%f",
            )
            stage_name = start_match.group("name").strip().lower()
            start_times[stage_name] = ts
            continue

        end_match = end_re.match(line)
        if end_match:
            ts = datetime.strptime(
                f"{end_match.group('ts')},{end_match.group('ms')}",
                "%Y-%m-%d %H:%M:%S,%f",
            )
            stage_name = STAGE_BY_NUMBER.get(end_match.group("num"), "unknown")
            start_ts = start_times.get(stage_name)
            if start_ts:
                duration = (ts - start_ts).total_seconds()
                stage_durations.setdefault(stage_name, []).append(duration)
            continue

    return stage_durations


def _collect_search_latency(search_results: List[dict]) -> List[float]:
    latencies: List[float] = []
    for item in search_results:
        meta = item.get("retrieval_metadata") or {}
        latency = meta.get("total_latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
    return latencies


def _collect_answer_latency(answer_results: List[dict]) -> List[float]:
    latencies: List[float] = []
    for item in answer_results:
        for key in ("response_duration_ms", "answer_duration_ms", "latency_ms"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                latencies.append(float(value))
                break
    return latencies


def _collect_search_errors(search_results: List[dict]) -> Counter:
    errors = Counter()
    for item in search_results:
        meta = item.get("retrieval_metadata") or {}
        err = meta.get("error") or item.get("error")
        if err:
            errors[str(err)] += 1
    return errors


def _collect_answer_errors(answer_results: List[dict]) -> Counter:
    errors = Counter()
    for item in answer_results:
        answer = item.get("answer") or ""
        if isinstance(answer, str) and answer.startswith("Error:"):
            errors[answer] += 1
        meta = item.get("metadata") or {}
        err = meta.get("error")
        if err:
            errors[str(err)] += 1
    return errors


def _load_list(path: Path) -> List[dict]:
    if not path.exists():
        return []
    data = _load_json(path)
    return data if isinstance(data, list) else []


def build_report(results_dir: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "results_dir": str(results_dir),
        "generated_at": datetime.now().isoformat(),
    }

    pipeline_log = results_dir / "pipeline.log"
    stage_totals: Dict[str, float] = {}
    if pipeline_log.exists():
        durations = _parse_pipeline_log(pipeline_log)
        for stage, values in durations.items():
            if values:
                stage_totals[stage] = round(float(values[-1]), 3)

    search_results = _load_list(results_dir / "search_results.json")
    answer_results = _load_list(results_dir / "answer_results.json")

    report["stage_totals_seconds"] = stage_totals

    report["add_latency_seconds"] = {
        "status": "missing",
        "reason": "per-item timing not recorded for add stage",
    }
    report["search_latency_seconds"] = _safe_stats(
        _collect_search_latency(search_results), scale=0.001
    ) or {
        "status": "missing",
        "reason": "no per-query latency found in search_results.json",
    }

    report["counts"] = {
        "search_results": len(search_results),
        "answer_results": len(answer_results),
    }
    stage_per_query_avg: Dict[str, Optional[float]] = {}
    if stage_totals:
        if search_results:
            stage_per_query_avg["search"] = round(
                stage_totals.get("search", 0.0) / len(search_results), 3
            )
        else:
            stage_per_query_avg["search"] = None

        if answer_results:
            stage_per_query_avg["answer"] = round(
                stage_totals.get("answer", 0.0) / len(answer_results), 3
            )
        else:
            stage_per_query_avg["answer"] = None

        stage_per_query_avg["add"] = None
        stage_per_query_avg["evaluate"] = None
    report["stage_per_query_avg_seconds"] = stage_per_query_avg

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect extended statistics from evaluation results."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to a results directory (contains pipeline.log, search_results.json, answer_results.json).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save JSON report.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.exists():
        raise SystemExit(f"Results dir not found: {results_dir}")

    report = build_report(results_dir)
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.write_text(output, encoding="utf-8")
        print(f"Saved report to: {out_path}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
