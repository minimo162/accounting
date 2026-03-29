import tempfile
import unittest
from pathlib import Path

from src.arag.answer_eval import (
    AnswerEvalCase,
    AnswerEvalResult,
    count_visible_citations,
    evaluate_answer_case,
    extract_visible_citation_numbers,
    find_uncited_lines,
    load_answer_eval_cases,
    render_answer_eval_markdown,
    summarize_answer_eval,
)


class AnswerEvalTests(unittest.TestCase):
    def test_extract_visible_citation_numbers_returns_unique_numbers(self):
        answer = "借手です。[1] 貸手です。[2][2]"

        self.assertEqual(extract_visible_citation_numbers(answer), {1, 2})

    def test_count_visible_citations_prefers_inline_markers(self):
        answer = "借手は使用権資産を計上します。[1] 経過措置もあります。[2]"

        self.assertEqual(count_visible_citations(answer, {"chunks_read_count": 9}), 2)

    def test_count_visible_citations_falls_back_to_metadata(self):
        answer = "借手は使用権資産を計上します。"

        self.assertEqual(count_visible_citations(answer, {"chunks_read_count": 3}), 3)

    def test_load_answer_eval_cases_reads_jsonl(self):
        payload = (
            '{"case_id":"lease","question":"Q","must_include_all":["借手"],"min_citations":2,'
            '"max_loops":4,"max_latency_sec":45,"max_retrieved_tokens":9000,'
            '"max_uncited_lines":0,"allowed_stop_reasons":["natural"]}\n'
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.jsonl"
            path.write_text(payload, encoding="utf-8")

            cases = load_answer_eval_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "lease")
        self.assertEqual(cases[0].must_include_all, ["借手"])
        self.assertEqual(cases[0].max_retrieved_tokens, 9000)
        self.assertEqual(cases[0].max_uncited_lines, 0)
        self.assertEqual(cases[0].allowed_stop_reasons, ["natural"])

    def test_find_uncited_lines_ignores_headings_and_flags_plain_lines(self):
        answer = "## 見出し\n借手です。\n- 使用権資産を計上します。[1]\n"

        self.assertEqual(find_uncited_lines(answer), ["借手です。"])

    def test_evaluate_answer_case_collects_failures(self):
        case = AnswerEvalCase(
            case_id="lease",
            question="Q",
            must_include_all=["借手", "貸手"],
            must_include_any=["経過措置", "適用時期"],
            must_exclude=["IFRS"],
            min_citations=2,
            max_loops=4,
            max_latency_sec=30.0,
            max_retrieved_tokens=8000,
            allowed_stop_reasons=["natural", "retrieval_budget"],
        )

        result = evaluate_answer_case(
            case,
            "## リース\n借手の処理を説明します。[1] IFRS にも触れます。\n結論です。",
            {
                "loops": 5,
                "read_chunk_count": 3,
                "total_retrieved_tokens": 9001,
                "stop_reason": "max_loops",
                "references": [{"id": "a:p1", "source": "企業会計基準第13号"}],
            },
            elapsed_sec=31.5,
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.read_chunk_count, 3)
        self.assertEqual(result.reference_count, 1)
        self.assertEqual(result.inline_citation_count, 1)
        self.assertEqual(result.uncited_line_count, 1)
        self.assertIn("missing_all=['貸手']", result.failure_reasons)
        self.assertIn("missing_any=['経過措置', '適用時期']", result.failure_reasons)
        self.assertIn("must_exclude=['IFRS']", result.failure_reasons)
        self.assertIn("citations<2 (1)", result.failure_reasons)
        self.assertIn("loops>4 (5)", result.failure_reasons)
        self.assertIn("latency>30.0s (31.5s)", result.failure_reasons)
        self.assertIn("retrieved_tokens>8000 (9001)", result.failure_reasons)
        self.assertIn("uncited_lines>0 (1)", result.failure_reasons)
        self.assertIn("missing_reference_urls=['企業会計基準第13号']", result.failure_reasons)
        self.assertIn("stop_reason_not_allowed=max_loops", result.failure_reasons)

    def test_evaluate_answer_case_detects_reference_alignment_mismatch(self):
        case = AnswerEvalCase(case_id="lease", question="Q")

        result = evaluate_answer_case(
            case,
            "借手です。[1][3]",
            {
                "references": [
                    {"id": "a:p1", "source": "A", "url": "https://example.com/a"},
                    {"id": "a:p2", "source": "B", "url": "https://example.com/b"},
                ],
                "stop_reason": "natural",
            },
            elapsed_sec=1.0,
        )

        self.assertFalse(result.reference_alignment_ok)
        self.assertEqual(result.max_citation_number, 3)
        self.assertIn("reference_alignment_mismatch (inline=[1, 3], references=2)", result.failure_reasons)

    def test_summarize_answer_eval_aggregates_metrics(self):
        results = [
            AnswerEvalResult(
                case_id="a",
                question="Q1",
                passed=True,
                elapsed_sec=10.0,
                loops=3,
                read_chunk_count=2,
                retrieved_tokens=1000,
                citation_count=2,
                inline_citation_count=2,
                reference_count=2,
                max_citation_number=2,
                uncited_line_count=0,
                reference_alignment_ok=True,
            ),
            AnswerEvalResult(
                case_id="b",
                question="Q2",
                passed=False,
                elapsed_sec=20.0,
                loops=5,
                read_chunk_count=4,
                retrieved_tokens=3000,
                citation_count=1,
                inline_citation_count=1,
                reference_count=1,
                max_citation_number=1,
                uncited_line_count=1,
                reference_alignment_ok=False,
                missing_reference_urls=["B"],
                failure_reasons=["missing_all=['借手']"],
            ),
        ]

        summary = summarize_answer_eval(results)

        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertEqual(summary["avg_latency_sec"], 15.0)
        self.assertEqual(summary["p95_latency_sec"], 20.0)
        self.assertEqual(summary["avg_loops"], 4.0)
        self.assertEqual(summary["avg_read_chunk_count"], 3.0)
        self.assertEqual(summary["avg_retrieved_tokens"], 2000.0)
        self.assertEqual(summary["avg_citations"], 1.5)
        self.assertEqual(summary["avg_uncited_lines"], 0.5)
        self.assertEqual(summary["reference_alignment_failures"], 1)
        self.assertEqual(summary["missing_reference_url_failures"], 1)

    def test_render_answer_eval_markdown_includes_failures_and_preview(self):
        summary = {
            "cases": 1,
            "passed": 0,
            "failed": 1,
            "pass_rate": 0.0,
            "avg_latency_sec": 12.3,
            "p95_latency_sec": 12.3,
            "avg_loops": 4.0,
            "avg_read_chunk_count": 2.0,
            "avg_retrieved_tokens": 5000.0,
            "avg_citations": 1.0,
            "avg_uncited_lines": 1.0,
            "reference_alignment_failures": 1,
            "missing_reference_url_failures": 1,
        }
        results = [
            AnswerEvalResult(
                case_id="lease",
                question="Q",
                passed=False,
                elapsed_sec=12.3,
                loops=4,
                read_chunk_count=2,
                retrieved_tokens=5000,
                citation_count=1,
                inline_citation_count=1,
                reference_count=1,
                max_citation_number=1,
                uncited_line_count=1,
                reference_alignment_ok=False,
                uncited_lines=["結論です。"],
                missing_reference_urls=["企業会計基準第13号"],
                failure_reasons=["missing_all=['貸手']"],
                answer_preview="借手の説明だけです。[1]",
            )
        ]

        report = render_answer_eval_markdown(summary, results)

        self.assertIn("# Answer Eval Report", report)
        self.assertIn("Avg read_chunk calls: 2.0", report)
        self.assertIn("Avg uncited lines: 1.0", report)
        self.assertIn("FAIL lease", report)
        self.assertIn("Reference alignment: mismatch", report)
        self.assertIn("Answer preview: 借手の説明だけです。[1]", report)


if __name__ == "__main__":
    unittest.main()
