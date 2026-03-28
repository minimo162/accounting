"""Helpers for answer-quality evaluation and performance measurement."""

from dataclasses import asdict, dataclass, field
import json
import math
import re
from pathlib import Path
from typing import Any


_SPACE_RE = re.compile(r"\s+")
_CITATION_RE = re.compile(r"\[(\d+)\]")


def normalize_answer_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text.strip()).lower()


def count_visible_citations(answer: str, metadata: dict[str, Any] | None = None) -> int:
    numbered = {int(match) for match in _CITATION_RE.findall(answer)}
    if numbered:
        return len(numbered)
    if metadata:
        value = metadata.get("chunks_read_count") or metadata.get("cited_reference_count") or 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


@dataclass(frozen=True)
class AnswerEvalCase:
    case_id: str
    question: str
    must_include_all: list[str] = field(default_factory=list)
    must_include_any: list[str] = field(default_factory=list)
    must_exclude: list[str] = field(default_factory=list)
    min_citations: int = 1
    max_loops: int | None = None
    max_latency_sec: float | None = None
    max_retrieved_tokens: int | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnswerEvalCase":
        return cls(
            case_id=str(payload["case_id"]),
            question=str(payload["question"]),
            must_include_all=[str(item) for item in payload.get("must_include_all", [])],
            must_include_any=[str(item) for item in payload.get("must_include_any", [])],
            must_exclude=[str(item) for item in payload.get("must_exclude", [])],
            min_citations=int(payload.get("min_citations", 1)),
            max_loops=int(payload["max_loops"]) if payload.get("max_loops") is not None else None,
            max_latency_sec=float(payload["max_latency_sec"]) if payload.get("max_latency_sec") is not None else None,
            max_retrieved_tokens=(
                int(payload["max_retrieved_tokens"]) if payload.get("max_retrieved_tokens") is not None else None
            ),
            notes=str(payload.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerEvalResult:
    case_id: str
    question: str
    passed: bool
    elapsed_sec: float
    loops: int
    read_chunk_count: int
    retrieved_tokens: int
    citation_count: int
    missing_all: list[str] = field(default_factory=list)
    missing_any: list[str] = field(default_factory=list)
    excluded_hits: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    stop_reason: str = ""
    answer_preview: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_answer_eval_cases(path: Path) -> list[AnswerEvalCase]:
    cases: list[AnswerEvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(AnswerEvalCase.from_dict(json.loads(line)))
    return cases


def evaluate_answer_case(
    case: AnswerEvalCase,
    answer: str,
    metadata: dict[str, Any] | None,
    elapsed_sec: float,
) -> AnswerEvalResult:
    normalized = normalize_answer_text(answer)
    missing_all = [term for term in case.must_include_all if normalize_answer_text(term) not in normalized]
    any_hits = [term for term in case.must_include_any if normalize_answer_text(term) in normalized]
    missing_any = [] if (not case.must_include_any or any_hits) else list(case.must_include_any)
    excluded_hits = [term for term in case.must_exclude if normalize_answer_text(term) in normalized]

    metadata = metadata or {}
    loops = int(metadata.get("loops") or 0)
    read_chunk_count = int(metadata.get("read_chunk_count") or 0)
    retrieved_tokens = int(metadata.get("total_retrieved_tokens") or 0)
    citation_count = count_visible_citations(answer, metadata)
    stop_reason = str(metadata.get("stop_reason") or "")

    failure_reasons: list[str] = []
    if missing_all:
        failure_reasons.append(f"missing_all={missing_all}")
    if missing_any:
        failure_reasons.append(f"missing_any={missing_any}")
    if excluded_hits:
        failure_reasons.append(f"must_exclude={excluded_hits}")
    if citation_count < case.min_citations:
        failure_reasons.append(f"citations<{case.min_citations} ({citation_count})")
    if case.max_loops is not None and loops > case.max_loops:
        failure_reasons.append(f"loops>{case.max_loops} ({loops})")
    if case.max_latency_sec is not None and elapsed_sec > case.max_latency_sec:
        failure_reasons.append(f"latency>{case.max_latency_sec:.1f}s ({elapsed_sec:.1f}s)")
    if case.max_retrieved_tokens is not None and retrieved_tokens > case.max_retrieved_tokens:
        failure_reasons.append(f"retrieved_tokens>{case.max_retrieved_tokens} ({retrieved_tokens})")

    return AnswerEvalResult(
        case_id=case.case_id,
        question=case.question,
        passed=not failure_reasons,
        elapsed_sec=elapsed_sec,
        loops=loops,
        read_chunk_count=read_chunk_count,
        retrieved_tokens=retrieved_tokens,
        citation_count=citation_count,
        missing_all=missing_all,
        missing_any=missing_any,
        excluded_hits=excluded_hits,
        failure_reasons=failure_reasons,
        stop_reason=stop_reason,
        answer_preview=answer[:500],
        notes=case.notes,
    )


def summarize_answer_eval(results: list[AnswerEvalResult]) -> dict[str, Any]:
    if not results:
        return {
            "cases": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "avg_latency_sec": 0.0,
            "p95_latency_sec": 0.0,
            "avg_loops": 0.0,
            "avg_read_chunk_count": 0.0,
            "avg_retrieved_tokens": 0.0,
            "avg_citations": 0.0,
        }

    latencies = sorted(result.elapsed_sec for result in results)
    p95_index = min(len(latencies) - 1, max(0, math.ceil(len(latencies) * 0.95) - 1))
    passed = sum(1 for result in results if result.passed)
    return {
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results),
        "avg_latency_sec": sum(result.elapsed_sec for result in results) / len(results),
        "p95_latency_sec": latencies[p95_index],
        "avg_loops": sum(result.loops for result in results) / len(results),
        "avg_read_chunk_count": sum(result.read_chunk_count for result in results) / len(results),
        "avg_retrieved_tokens": sum(result.retrieved_tokens for result in results) / len(results),
        "avg_citations": sum(result.citation_count for result in results) / len(results),
    }


def render_answer_eval_markdown(summary: dict[str, Any], results: list[AnswerEvalResult]) -> str:
    lines = [
        "# Answer Eval Report",
        "",
        f"- Cases: {summary['cases']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Pass rate: {summary['pass_rate']:.0%}",
        f"- Avg latency: {summary['avg_latency_sec']:.1f}s",
        f"- P95 latency: {summary['p95_latency_sec']:.1f}s",
        f"- Avg loops: {summary['avg_loops']:.1f}",
        f"- Avg read_chunk calls: {summary['avg_read_chunk_count']:.1f}",
        f"- Avg retrieved tokens: {summary['avg_retrieved_tokens']:.0f}",
        "",
        "## Cases",
    ]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.extend(
            [
                "",
                f"### {status} {result.case_id}",
                f"- Question: {result.question}",
                f"- Latency: {result.elapsed_sec:.1f}s",
                f"- Loops: {result.loops}",
                f"- read_chunk calls: {result.read_chunk_count}",
                f"- Retrieved tokens: {result.retrieved_tokens}",
                f"- Citations: {result.citation_count}",
                f"- Stop reason: {result.stop_reason or '-'}",
            ]
        )
        if result.failure_reasons:
            lines.append(f"- Failures: {', '.join(result.failure_reasons)}")
            if result.answer_preview:
                lines.append(f"- Answer preview: {result.answer_preview}")
        if result.notes:
            lines.append(f"- Notes: {result.notes}")
    return "\n".join(lines) + "\n"
