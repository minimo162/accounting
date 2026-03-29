"""Run representative answer eval checks and detect latency regressions."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.answer_eval import evaluate_answer_case, load_answer_eval_cases, summarize_answer_eval  # noqa: E402
from src.arag.observability import question_sha1  # noqa: E402


_LOCAL_AGENT = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="eval/answer_eval_set.jsonl")
    parser.add_argument("--backend", choices=["api", "local"], default="api")
    parser.add_argument("--api-url", default="http://localhost:8000/api/ask")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--avg-latency-threshold", type=float, default=35.0)
    parser.add_argument("--p95-latency-threshold", type=float, default=60.0)
    parser.add_argument("--avg-loops-threshold", type=float, default=5.0)
    parser.add_argument("--output")
    parser.add_argument("--fail-on-threshold", action="store_true")
    return parser.parse_args()


def normalize_api_url(api_url: str) -> str:
    trimmed = api_url.rstrip("/")
    if trimmed.endswith("/api/ask"):
        return trimmed
    return f"{trimmed}/api/ask"


def select_cases(all_cases, selected_ids: set[str], limit: int | None):
    cases = [case for case in all_cases if not selected_ids or case.case_id in selected_ids]
    if limit is not None:
        return cases[:limit]
    return cases


async def run_local_question(question: str) -> tuple[str, dict[str, Any]]:
    from src.api.main import get_agent

    global _LOCAL_AGENT
    if _LOCAL_AGENT is None:
        _LOCAL_AGENT = get_agent()
    result = await _LOCAL_AGENT.arun(question, history=[])
    metadata = dict(result)
    metadata["references"] = result.get("references", [])
    metadata["source_url_map"] = result.get("source_url_map", {})
    return result.get("answer", ""), metadata


def run_api_question(case_id: str, question: str, api_url: str, timeout: float) -> tuple[str, dict[str, Any]]:
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.post(
            api_url,
            json={"question": question, "history": []},
            headers={"X-Monitor-Case-Id": case_id},
        )
        response.raise_for_status()
        payload = response.json()
    metadata = dict(payload.get("metadata", {}))
    metadata["references"] = payload.get("references", metadata.get("references", []))
    metadata["source_url_map"] = payload.get("source_url_map", metadata.get("source_url_map", {}))
    return payload.get("answer", ""), metadata


def collect_threshold_failures(summary: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    failures: list[str] = []
    if summary.get("avg_latency_sec", 0.0) > thresholds["avg_latency_sec"]:
        failures.append(
            f"avg_latency_sec>{thresholds['avg_latency_sec']:.1f} ({summary['avg_latency_sec']:.1f})"
        )
    if summary.get("p95_latency_sec", 0.0) > thresholds["p95_latency_sec"]:
        failures.append(
            f"p95_latency_sec>{thresholds['p95_latency_sec']:.1f} ({summary['p95_latency_sec']:.1f})"
        )
    if summary.get("avg_loops", 0.0) > thresholds["avg_loops"]:
        failures.append(
            f"avg_loops>{thresholds['avg_loops']:.1f} ({summary['avg_loops']:.1f})"
        )
    return failures


async def main() -> None:
    args = parse_args()
    api_url = normalize_api_url(args.api_url) if args.backend == "api" else ""
    cases = select_cases(
        load_answer_eval_cases(Path(args.cases)),
        selected_ids=set(args.case_id),
        limit=args.limit,
    )
    if not cases:
        raise SystemExit("No monitoring cases selected.")

    thresholds = {
        "avg_latency_sec": args.avg_latency_threshold,
        "p95_latency_sec": args.p95_latency_threshold,
        "avg_loops": args.avg_loops_threshold,
    }
    results = []
    monitor_rows = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case.case_id}", file=sys.stderr)
        start = time.perf_counter()
        if args.backend == "local":
            answer, metadata = await run_local_question(case.question)
        else:
            answer, metadata = run_api_question(case.case_id, case.question, api_url, args.timeout)
        elapsed = time.perf_counter() - start
        result = evaluate_answer_case(case, answer, metadata, elapsed)
        results.append(result)
        monitor_rows.append(
            {
                "case_id": case.case_id,
                "question_sha1": question_sha1(case.question),
                "elapsed_sec": round(elapsed, 3),
                "loops": result.loops,
                "stop_reason": result.stop_reason,
                "retrieved_tokens": result.retrieved_tokens,
                "citation_count": result.citation_count,
                "cited_line_count": result.cited_line_count,
                "reference_count": result.reference_count,
                "request_id": metadata.get("request_id"),
                "query_class": metadata.get("query_class"),
                "passed": result.passed,
                "failure_reasons": list(result.failure_reasons),
            }
        )

    summary = summarize_answer_eval(results)
    threshold_failures = collect_threshold_failures(summary, thresholds)
    payload = {
        "event": "answer_eval_monitor",
        "checked_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "backend": args.backend,
        "api_url": api_url if args.backend == "api" else None,
        "thresholds": thresholds,
        "threshold_failures": threshold_failures,
        "summary": summary,
        "cases": monitor_rows,
    }

    output = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Wrote report: {out_path}", file=sys.stderr)

    print(output)

    if args.fail_on_threshold and threshold_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
