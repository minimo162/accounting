import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import src.api.main as api_main


class FakeAgent:
    async def arun(self, question, history):
        return {
            "answer": "## 結論\n借手は使用権資産を計上します[1]。",
            "loops": 2,
            "stop_reason": "natural",
            "total_cost": 0.0,
            "total_retrieved_tokens": 321,
            "cited_reference_count": 1,
            "read_chunk_count": 1,
            "references": [
                {
                    "id": "std13.pdf:c12",
                    "source": "企業会計基準第13号 > 第10項",
                    "text": "第10項 使用権資産を計上する。",
                    "url": "https://example.test/std13.pdf#page=5",
                    "display_number": 1,
                }
            ],
            "source_url_map": {
                "企業会計基準第13号": "https://example.test/std13.pdf#page=5",
                "第13号": "https://example.test/std13.pdf#page=5",
            },
        }

    async def arun_stream(self, question, history):
        yield {"type": "status", "data": "調査中..."}
        yield {"type": "answer_delta", "data": "## 結論\n借手は使用権資産を計上します"}
        yield {"type": "answer_delta", "data": "[1]。"}
        yield {
            "type": "reference",
            "data": {
                "id": "std13.pdf:c12",
                "source": "企業会計基準第13号 > 第10項",
                "text": "第10項 使用権資産を計上する。",
                "url": "https://example.test/std13.pdf#page=5",
                "display_number": 1,
            },
        }
        yield {
            "type": "done",
            "data": {
                "loops": 2,
                "stop_reason": "natural",
                "chunks_read_count": 1,
                "read_chunk_count": 1,
                "total_cost": 0.0,
                "total_retrieved_tokens": 321,
                "source_url_map": {
                    "企業会計基準第13号": "https://example.test/std13.pdf#page=5",
                    "第13号": "https://example.test/std13.pdf#page=5",
                },
            },
        }


class ApiReferenceContractTests(unittest.TestCase):
    def test_ask_returns_references_and_source_url_map(self):
        with patch("src.api.main.get_agent", return_value=FakeAgent()):
            with TestClient(api_main.app) as client:
                response = client.post("/api/ask", json={"question": "Q", "history": []})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "## 結論\n借手は使用権資産を計上します[1]。")
        self.assertEqual(payload["metadata"]["chunks_read_count"], 1)
        self.assertEqual(payload["metadata"]["read_chunk_count"], 1)
        self.assertEqual(payload["references"][0]["display_number"], 1)
        self.assertEqual(payload["references"][0]["url"], "https://example.test/std13.pdf#page=5")
        self.assertEqual(
            payload["source_url_map"]["企業会計基準第13号"],
            "https://example.test/std13.pdf#page=5",
        )

    def test_ask_stream_emits_reference_events_and_done_metadata(self):
        with patch("src.api.main.get_agent", return_value=FakeAgent()):
            with TestClient(api_main.app) as client:
                with client.stream("POST", "/api/ask/stream", json={"question": "Q", "history": []}) as response:
                    self.assertEqual(response.status_code, 200)
                    body = "".join(response.iter_text())

        events = []
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            events.append(json.loads(line[6:]))

        self.assertEqual(events[0]["type"], "status")
        self.assertEqual(events[1]["type"], "answer_delta")
        self.assertEqual(events[2]["type"], "answer_delta")
        self.assertEqual(events[3]["type"], "reference")
        self.assertEqual(events[3]["data"]["display_number"], 1)
        self.assertEqual(events[3]["data"]["url"], "https://example.test/std13.pdf#page=5")
        self.assertEqual(events[4]["type"], "done")
        self.assertEqual(events[4]["data"]["source_url_map"]["第13号"], "https://example.test/std13.pdf#page=5")
        self.assertEqual(events[4]["data"]["read_chunk_count"], 1)


if __name__ == "__main__":
    unittest.main()
