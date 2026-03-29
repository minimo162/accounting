"""Helpers for answer-quality evaluation and performance measurement."""

from dataclasses import asdict, dataclass, field
import json
import math
import re
from pathlib import Path
from typing import Any


_SPACE_RE = re.compile(r"\s+")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_HEADING_RE = re.compile(r"^#{1,6}\s+")
_LIST_PREFIX_RE = re.compile(r"^(?:[-*+]\s+|\d+\.\s+)")
_BOLD_ONLY_RE = re.compile(r"^\*\*[^*]+\*\*$")
_HARD_FAILURE_PREFIXES = (
    "missing_all=",
    "must_exclude=",
    "citations<",
    "cited_lines<",
    "missing_inline_citations",
    "uncited_lines>",
    "reference_alignment_mismatch",
    "missing_reference_urls=",
)


def normalize_answer_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text.strip()).lower()


def extract_visible_citation_numbers(answer: str) -> set[int]:
    return {int(match) for match in _CITATION_RE.findall(answer)}


def count_visible_citations(answer: str, metadata: dict[str, Any] | None = None) -> int:
    numbered = extract_visible_citation_numbers(answer)
    if numbered:
        return len(numbered)
    if metadata:
        value = metadata.get("chunks_read_count") or metadata.get("cited_reference_count") or 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def find_uncited_lines(answer: str) -> list[str]:
    uncited: list[str] = []
    in_code_block = False

    for raw_line in answer.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if _HEADING_RE.match(stripped):
            continue
        if re.fullmatch(r"[-*_]{3,}", stripped):
            continue
        if _CITATION_RE.search(stripped):
            continue

        candidate = _LIST_PREFIX_RE.sub("", stripped)
        if _BOLD_ONLY_RE.fullmatch(candidate):
            continue

        candidate = re.sub(r"[>*_`~]", "", candidate).strip()
        if not candidate:
            continue
        uncited.append(stripped)

    return uncited


def count_cited_lines(answer: str) -> int:
    count = 0
    for raw_line in answer.splitlines():
        stripped = raw_line.strip()
        if not stripped or _HEADING_RE.match(stripped):
            continue
        if _CITATION_RE.search(stripped):
            count += 1
    return count


def evaluate_reference_alignment(answer: str, references: list[dict[str, Any]]) -> tuple[bool, int]:
    numbered = extract_visible_citation_numbers(answer)
    if not numbered:
        return len(references) == 0, 0

    max_citation_number = max(numbered)
    expected = set(range(1, max_citation_number + 1))
    return numbered == expected and len(references) == max_citation_number, max_citation_number


def find_missing_reference_urls(references: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for ref in references:
        if ref.get("url"):
            continue
        label = str(ref.get("source") or ref.get("id") or "unknown")
        missing.append(label)
    return missing


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
    min_cited_lines: int = 0
    max_uncited_lines: int = 0
    require_inline_citations: bool = True
    require_reference_alignment: bool = True
    require_reference_urls: bool = True
    allowed_stop_reasons: list[str] = field(default_factory=list)
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
            min_cited_lines=int(payload.get("min_cited_lines", 0)),
            max_uncited_lines=int(payload.get("max_uncited_lines", 0)),
            require_inline_citations=bool(payload.get("require_inline_citations", True)),
            require_reference_alignment=bool(payload.get("require_reference_alignment", True)),
            require_reference_urls=bool(payload.get("require_reference_urls", True)),
            allowed_stop_reasons=[str(item) for item in payload.get("allowed_stop_reasons", [])],
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
    cited_line_count: int
    inline_citation_count: int
    reference_count: int
    max_citation_number: int
    uncited_line_count: int
    reference_alignment_ok: bool
    missing_all: list[str] = field(default_factory=list)
    missing_any: list[str] = field(default_factory=list)
    excluded_hits: list[str] = field(default_factory=list)
    uncited_lines: list[str] = field(default_factory=list)
    missing_reference_urls: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    hard_failure_reasons: list[str] = field(default_factory=list)
    soft_failure_reasons: list[str] = field(default_factory=list)
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


def split_failure_reasons(failure_reasons: list[str]) -> tuple[list[str], list[str]]:
    hard = [reason for reason in failure_reasons if reason.startswith(_HARD_FAILURE_PREFIXES)]
    soft = [reason for reason in failure_reasons if reason not in hard]
    return hard, soft


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
    references_raw = metadata.get("references") or []
    references = [ref for ref in references_raw if isinstance(ref, dict)]
    inline_citation_numbers = extract_visible_citation_numbers(answer)
    citation_count = count_visible_citations(answer, metadata)
    cited_line_count = count_cited_lines(answer)
    inline_citation_count = len(inline_citation_numbers)
    reference_alignment_ok, max_citation_number = evaluate_reference_alignment(answer, references)
    missing_reference_urls = find_missing_reference_urls(references)
    uncited_lines = find_uncited_lines(answer)
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
    if case.min_cited_lines and cited_line_count < case.min_cited_lines:
        failure_reasons.append(f"cited_lines<{case.min_cited_lines} ({cited_line_count})")
    if case.require_inline_citations and not inline_citation_numbers:
        failure_reasons.append("missing_inline_citations")
    if case.max_loops is not None and loops > case.max_loops:
        failure_reasons.append(f"loops>{case.max_loops} ({loops})")
    if case.max_latency_sec is not None and elapsed_sec > case.max_latency_sec:
        failure_reasons.append(f"latency>{case.max_latency_sec:.1f}s ({elapsed_sec:.1f}s)")
    if case.max_retrieved_tokens is not None and retrieved_tokens > case.max_retrieved_tokens:
        failure_reasons.append(f"retrieved_tokens>{case.max_retrieved_tokens} ({retrieved_tokens})")
    if len(uncited_lines) > case.max_uncited_lines:
        failure_reasons.append(f"uncited_lines>{case.max_uncited_lines} ({len(uncited_lines)})")
    if case.require_reference_alignment and not reference_alignment_ok:
        failure_reasons.append(
            f"reference_alignment_mismatch (inline={sorted(inline_citation_numbers)}, references={len(references)})"
        )
    if case.require_reference_urls and missing_reference_urls:
        failure_reasons.append(f"missing_reference_urls={missing_reference_urls}")
    if case.allowed_stop_reasons and stop_reason not in case.allowed_stop_reasons:
        failure_reasons.append(f"stop_reason_not_allowed={stop_reason or '-'}")
    hard_failure_reasons, soft_failure_reasons = split_failure_reasons(failure_reasons)

    return AnswerEvalResult(
        case_id=case.case_id,
        question=case.question,
        passed=not failure_reasons,
        elapsed_sec=elapsed_sec,
        loops=loops,
        read_chunk_count=read_chunk_count,
        retrieved_tokens=retrieved_tokens,
        citation_count=citation_count,
        cited_line_count=cited_line_count,
        inline_citation_count=inline_citation_count,
        reference_count=len(references),
        max_citation_number=max_citation_number,
        uncited_line_count=len(uncited_lines),
        reference_alignment_ok=reference_alignment_ok,
        missing_all=missing_all,
        missing_any=missing_any,
        excluded_hits=excluded_hits,
        uncited_lines=uncited_lines,
        missing_reference_urls=missing_reference_urls,
        failure_reasons=failure_reasons,
        hard_failure_reasons=hard_failure_reasons,
        soft_failure_reasons=soft_failure_reasons,
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
            "avg_cited_lines": 0.0,
            "avg_uncited_lines": 0.0,
            "reference_alignment_failures": 0,
            "missing_reference_url_failures": 0,
            "insufficient_detail_failures": 0,
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
        "avg_cited_lines": sum(result.cited_line_count for result in results) / len(results),
        "avg_uncited_lines": sum(result.uncited_line_count for result in results) / len(results),
        "reference_alignment_failures": sum(1 for result in results if not result.reference_alignment_ok),
        "missing_reference_url_failures": sum(1 for result in results if result.missing_reference_urls),
        "insufficient_detail_failures": sum(
            1 for result in results if any(reason.startswith("cited_lines<") for reason in result.failure_reasons)
        ),
    }


def summarize_answer_eval_gate(results: list[AnswerEvalResult], gate_mode: str = "none") -> dict[str, Any]:
    gate_mode = gate_mode.lower()
    if gate_mode not in {"none", "hard", "strict"}:
        raise ValueError(f"Unsupported gate mode: {gate_mode}")

    hard_failed_results = [result for result in results if result.hard_failure_reasons]
    soft_only_failed_results = [
        result for result in results if result.failure_reasons and not result.hard_failure_reasons
    ]
    if gate_mode == "strict":
        blocking_results = [result for result in results if result.failure_reasons]
    elif gate_mode == "hard":
        blocking_results = hard_failed_results
    else:
        blocking_results = []

    return {
        "gate_mode": gate_mode,
        "passed": len(blocking_results) == 0,
        "blocking_failed": len(blocking_results),
        "hard_failed": len(hard_failed_results),
        "soft_failed": len(soft_only_failed_results),
        "blocking_case_ids": [result.case_id for result in blocking_results],
        "hard_failure_case_ids": [result.case_id for result in hard_failed_results],
        "soft_failure_case_ids": [result.case_id for result in soft_only_failed_results],
    }


def render_answer_eval_markdown(
    summary: dict[str, Any],
    results: list[AnswerEvalResult],
    gate: dict[str, Any] | None = None,
) -> str:
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
        f"- Avg cited lines: {summary['avg_cited_lines']:.1f}",
        f"- Avg uncited lines: {summary['avg_uncited_lines']:.1f}",
        f"- Reference alignment failures: {summary['reference_alignment_failures']}",
        f"- Missing reference URL failures: {summary['missing_reference_url_failures']}",
        f"- Insufficient detail failures: {summary['insufficient_detail_failures']}",
    ]
    if gate is not None:
        lines.extend(
            [
                f"- Gate mode: {gate['gate_mode']}",
                f"- Gate passed: {'yes' if gate['passed'] else 'no'}",
                f"- Blocking failures: {gate['blocking_failed']}",
                f"- Hard-fail cases: {gate['hard_failed']}",
                f"- Soft-fail cases: {gate['soft_failed']}",
            ]
        )
        override_reason = str(gate.get("override_reason") or "").strip()
        if override_reason:
            lines.append(f"- Override reason: {override_reason}")
        if gate.get("blocking_case_ids"):
            lines.append(f"- Blocking case IDs: {', '.join(gate['blocking_case_ids'])}")
    lines.extend(["", "## Cases"])
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
                f"- Cited lines: {result.cited_line_count}",
                f"- Inline citations: {result.inline_citation_count}",
                f"- References: {result.reference_count}",
                f"- Max citation number: {result.max_citation_number}",
                f"- Uncited lines: {result.uncited_line_count}",
                f"- Reference alignment: {'ok' if result.reference_alignment_ok else 'mismatch'}",
                f"- Stop reason: {result.stop_reason or '-'}",
            ]
        )
        if result.failure_reasons:
            lines.append(f"- Failures: {', '.join(result.failure_reasons)}")
            if result.hard_failure_reasons:
                lines.append(f"- Hard failures: {', '.join(result.hard_failure_reasons)}")
            if result.soft_failure_reasons:
                lines.append(f"- Soft failures: {', '.join(result.soft_failure_reasons)}")
            if result.uncited_lines:
                lines.append(f"- Uncited lines detail: {' | '.join(result.uncited_lines[:5])}")
            if result.missing_reference_urls:
                lines.append(f"- Missing reference URLs: {', '.join(result.missing_reference_urls)}")
            if result.answer_preview:
                lines.append(f"- Answer preview: {result.answer_preview}")
        if result.notes:
            lines.append(f"- Notes: {result.notes}")
    return "\n".join(lines) + "\n"
