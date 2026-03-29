"""Run answer-quality and latency evaluation against local agent or HTTP API."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.answer_eval import (  # noqa: E402
    AnswerEvalResult,
    evaluate_answer_case,
    load_answer_eval_cases,
    render_answer_eval_markdown,
    summarize_answer_eval,
    summarize_answer_eval_gate,
)


_LOCAL_AGENT = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="eval/answer_eval_set.jsonl")
    parser.add_argument("--backend", choices=["api", "local"], default="api")
    parser.add_argument("--api-url", default="http://localhost:8000/api/ask")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--fail-on-fail", action="store_true")
    parser.add_argument("--gate-mode", choices=["none", "hard", "strict"], default="none")
    parser.add_argument("--override-reason", default="")
    return parser.parse_args()


async def run_local_question(question: str) -> tuple[str, dict[str, Any]]:
    from src.api.main import get_agent

    global _LOCAL_AGENT
    if _LOCAL_AGENT is None:
        _LOCAL_AGENT = get_agent()
    result = await _LOCAL_AGENT.arun(question, history=[])
    metadata = {
        "loops": result.get("loops", 0),
        "chunks_read_count": result.get("cited_reference_count", 0),
        "read_chunk_count": result.get("read_chunk_count", 0),
        "stop_reason": result.get("stop_reason", ""),
        "total_cost": result.get("total_cost", 0.0),
        "total_retrieved_tokens": result.get("total_retrieved_tokens", 0),
        "cited_reference_count": result.get("cited_reference_count", 0),
        "references": result.get("references", []),
        "source_url_map": result.get("source_url_map", {}),
        "request_id": result.get("request_id"),
        "query_class": result.get("query_class"),
    }
    return result.get("answer", ""), metadata


def normalize_api_url(api_url: str) -> str:
    trimmed = api_url.rstrip("/")
    if trimmed.endswith("/api/ask"):
        return trimmed
    return f"{trimmed}/api/ask"


def run_api_question(case_id: str, question: str, api_url: str, timeout: float) -> tuple[str, dict[str, Any]]:
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.post(
            api_url,
            json={"question": question, "history": []},
            headers={"X-Monitor-Case-Id": f"answer-eval:{case_id}"},
        )
        response.raise_for_status()
        payload = response.json()
    metadata = dict(payload.get("metadata", {}))
    metadata["references"] = payload.get("references", metadata.get("references", []))
    metadata["source_url_map"] = payload.get("source_url_map", metadata.get("source_url_map", {}))
    return payload.get("answer", ""), metadata


def select_cases(all_cases, selected_ids: set[str], limit: int | None):
    cases = [case for case in all_cases if not selected_ids or case.case_id in selected_ids]
    if limit is not None:
        return cases[:limit]
    return cases


async def main():
    args = parse_args()
    api_url = normalize_api_url(args.api_url) if args.backend == "api" else ""
    cases = select_cases(
        load_answer_eval_cases(Path(args.cases)),
        selected_ids=set(args.case_id),
        limit=args.limit,
    )
    if not cases:
        raise SystemExit("No evaluation cases selected.")

    results: list[AnswerEvalResult] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case.case_id}", file=sys.stderr)
        start = time.perf_counter()
        if args.backend == "local":
            answer, metadata = await run_local_question(case.question)
        else:
            answer, metadata = run_api_question(case.case_id, case.question, api_url, args.timeout)
        elapsed = time.perf_counter() - start
        results.append(evaluate_answer_case(case, answer, metadata, elapsed))

    summary = summarize_answer_eval(results)
    gate = summarize_answer_eval_gate(results, args.gate_mode)
    override_reason = args.override_reason.strip()
    if override_reason:
        gate["override_reason"] = override_reason
        gate["override_applied"] = not gate["passed"]
    else:
        gate["override_applied"] = False
    payload = {
        "summary": summary,
        "gate": gate,
        "results": [result.to_dict() for result in results],
    }

    if args.format == "markdown":
        output_text = render_answer_eval_markdown(summary, results, gate)
    else:
        output_text = json.dumps(payload, ensure_ascii=False, indent=2)

    markdown_text = render_answer_eval_markdown(summary, results, gate)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Wrote report: {out_path}")

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json_text, encoding="utf-8")
        print(f"Wrote JSON report: {json_path}", file=sys.stderr)

    if args.markdown_output:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown_text, encoding="utf-8")
        print(f"Wrote Markdown report: {markdown_path}", file=sys.stderr)

    print(output_text)

    if args.fail_on_fail and summary["failed"] > 0:
        raise SystemExit(1)
    if args.gate_mode != "none" and not gate["passed"] and not override_reason:
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())
