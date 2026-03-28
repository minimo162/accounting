import unittest

from src.arag.agent import Agent
from src.arag.context import AgentContext


class AgentReferenceTests(unittest.TestCase):
    def make_agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent.chunk_map = {
            "std13.pdf:p4": {
                "id": "std13.pdf:p4",
                "parent_id": "std13.pdf:p4",
                "text": "親チャンク 第10項",
                "source": "企業会計基準第13号",
                "file": "std13.pdf",
                "pdf_page": 4,
            },
            "std13.pdf:c12": {
                "id": "std13.pdf:c12",
                "parent_id": "std13.pdf:p4",
                "text": "第10項 使用権資産を計上する。",
                "source": "企業会計基準第13号 > 第10項",
                "file": "std13.pdf",
                "pdf_page": 5,
            },
            "std20.pdf:c2": {
                "id": "std20.pdf:c2",
                "parent_id": "std20.pdf:p1",
                "text": "第2項 追加の注記が必要である。",
                "source": "実務対応報告第20号 > 第2項",
                "file": "std20.pdf",
                "pdf_page": 1,
            },
        }
        agent.pdf_sources = {
            "std13.pdf": "https://example.com/std13.pdf",
            "std20.pdf": "https://example.com/std20.pdf",
        }
        return agent

    def test_finalize_answer_numbers_chunk_ids_in_citation_order(self):
        agent = self.make_agent()
        context = AgentContext()
        context.read_chunk_ids.add("std13.pdf:p4")

        answer, refs, source_url_map = agent._finalize_answer(
            (
                "## 結論\n"
                "使用権資産を計上します（企業会計基準第13号 第10項）[std13.pdf:c12]。\n"
                "追加注記も必要です（実務対応報告第20号 第2項）[std20.pdf:c2]。"
            ),
            context,
        )

        self.assertNotIn("std13.pdf:c12", answer)
        self.assertNotIn("std20.pdf:c2", answer)
        self.assertNotIn("企業会計基準第13号 第10項", answer)
        self.assertNotIn("実務対応報告第20号 第2項", answer)
        self.assertIn("[1]", answer)
        self.assertIn("[2]", answer)
        self.assertEqual([ref["id"] for ref in refs], ["std13.pdf:c12", "std20.pdf:c2"])
        self.assertEqual(refs[0]["url"], "https://example.com/std13.pdf#page=5")
        self.assertEqual(refs[1]["url"], "https://example.com/std20.pdf")
        self.assertEqual(source_url_map["企業会計基準第13号"], "https://example.com/std13.pdf#page=5")
        self.assertEqual(source_url_map["実務対応報告第20号"], "https://example.com/std20.pdf")
        self.assertEqual(source_url_map["第13号"], "https://example.com/std13.pdf#page=5")
        self.assertEqual(source_url_map["第20号"], "https://example.com/std20.pdf")

    def test_finalize_answer_keeps_cited_child_when_parent_was_read(self):
        agent = self.make_agent()
        context = AgentContext()
        context.read_chunk_ids.add("std13.pdf:p4")

        answer, refs, _ = agent._finalize_answer(
            "第10項を参照してください（企業会計基準第13号 第10項）[std13.pdf:c12][std13.pdf:p4]。",
            context,
        )

        self.assertEqual(answer.count("[1]"), 1)
        self.assertEqual(answer.count("[2]"), 0)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["id"], "std13.pdf:c12")
        self.assertEqual(refs[0]["url"], "https://example.com/std13.pdf#page=5")

    def test_finalize_answer_prefers_child_reference_when_parent_and_child_are_both_cited(self):
        agent = self.make_agent()
        context = AgentContext()

        answer, refs, _ = agent._finalize_answer(
            "使用権資産を計上します[std13.pdf:p4][std13.pdf:c12]。",
            context,
        )

        self.assertEqual(answer, "使用権資産を計上します[1]。")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["id"], "std13.pdf:c12")
        self.assertEqual(refs[0]["url"], "https://example.com/std13.pdf#page=5")

    def test_finalize_answer_shows_only_cited_references_when_other_chunks_were_read(self):
        agent = self.make_agent()
        context = AgentContext()
        context.read_chunk_ids.update({"std13.pdf:p4", "std20.pdf:c2"})

        answer, refs, _ = agent._finalize_answer(
            "使用権資産を計上します[std13.pdf:c12]。",
            context,
        )

        self.assertEqual(answer, "使用権資産を計上します[1]。")
        self.assertEqual([ref["id"] for ref in refs], ["std13.pdf:c12"])

    def test_sanitize_answer_strips_citation_parentheticals_and_page_labels(self):
        agent = self.make_agent()

        answer, cited_ids = agent._sanitize_answer(
            "使用権資産を計上します（企業会計基準第13号 第10項 p60）[std13.pdf:c12]。",
            number_refs=True,
        )

        self.assertEqual(answer, "使用権資産を計上します[1]。")
        self.assertEqual(cited_ids, ["std13.pdf:c12"])

    def test_sanitize_answer_keeps_non_citation_parentheses(self):
        agent = self.make_agent()

        answer = agent._sanitize_answer(
            "借手（連結子会社を含む）は使用権資産を計上します。"
        )

        self.assertEqual(answer, "借手（連結子会社を含む）は使用権資産を計上します。")

    def test_sanitize_answer_drops_uncited_summary_lines(self):
        agent = self.make_agent()

        answer, cited_ids = agent._sanitize_answer(
            (
                "## 結論\n"
                "- 借手は使用権資産を計上します[std13.pdf:c12]。\n"
                "以上が借手側の会計処理の概要です。"
            ),
            number_refs=True,
        )

        self.assertEqual(answer, "## 結論\n- 借手は使用権資産を計上します[1]。")
        self.assertEqual(cited_ids, ["std13.pdf:c12"])

    def test_sanitize_answer_drops_empty_heading_sections(self):
        agent = self.make_agent()

        answer, cited_ids = agent._sanitize_answer(
            (
                "## 結論\n"
                "- 借手は使用権資産を計上します[std13.pdf:c12]。\n"
                "## 補足\n"
                "以上が概要です。"
            ),
            number_refs=True,
        )

        self.assertEqual(answer, "## 結論\n- 借手は使用権資産を計上します[1]。")
        self.assertEqual(cited_ids, ["std13.pdf:c12"])

    def test_count_cited_references_prefers_numbered_markers(self):
        count = Agent._count_cited_references(
            "償却します[1]。追加注記も必要です[2]。同じ根拠を再掲します[1]。"
        )

        self.assertEqual(count, 2)

    def test_build_answer_summary_separates_citation_count_and_read_count(self):
        agent = self.make_agent()
        context = AgentContext()
        context.read_chunk_ids.update({"std13.pdf:p4", "std20.pdf:c2"})

        summary = agent._build_answer_summary("償却します[1]。追加注記も必要です[2]。", context)

        self.assertEqual(summary["chunks_read_count"], 2)
        self.assertEqual(summary["read_chunk_count"], 2)
        self.assertEqual(set(summary["chunks_read_ids"]), {"std13.pdf:p4", "std20.pdf:c2"})


if __name__ == "__main__":
    unittest.main()
