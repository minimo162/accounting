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
            "query_class": "simple",
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
                    "doc_type": "企業会計基準",
                    "doc_title": "企業会計基準第13号",
                    "section_title": "第10項",
                    "section_label": "第10項",
                    "page_label": "p.5",
                    "display_number": 1,
                }
            ],
            "source_url_map": {
                "企業会計基準第13号": "https://example.test/std13.pdf#page=5",
                "第13号": "https://example.test/std13.pdf#page=5",
            },
            "evidence_coverage": {
                "slots": ["借手"],
                "covered_slots": ["借手"],
                "uncovered_slots": [],
                "coverage_ratio": 1.0,
            },
            "uncertainty": {
                "present": False,
                "insufficient_points": [],
                "covered_points": ["借手"],
                "coverage_ratio": 1.0,
                "note": "",
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
                "doc_type": "企業会計基準",
                "doc_title": "企業会計基準第13号",
                "section_title": "第10項",
                "section_label": "第10項",
                "page_label": "p.5",
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
                "evidence_coverage": {
                    "slots": ["借手"],
                    "covered_slots": ["借手"],
                    "uncovered_slots": [],
                    "coverage_ratio": 1.0,
                },
                "uncertainty": {
                    "present": False,
                    "insufficient_points": [],
                    "covered_points": ["借手"],
                    "coverage_ratio": 1.0,
                    "note": "",
                },
                "source_url_map": {
                    "企業会計基準第13号": "https://example.test/std13.pdf#page=5",
                    "第13号": "https://example.test/std13.pdf#page=5",
                },
            },
        }


class ApiReferenceContractTests(unittest.TestCase):
    @staticmethod
    def _structured_events(log_output):
        events = []
        for entry in log_output:
            _, _, message = entry.partition("src.api.main:")
            message = message.strip()
            if not message.startswith("{"):
                continue
            payload = json.loads(message)
            if "event" in payload:
                events.append(payload)
        return events

    def test_ask_returns_references_and_source_url_map(self):
        with patch("src.api.main.get_agent", return_value=FakeAgent()):
            with self.assertLogs("src.api.main", level="INFO") as logs:
                with TestClient(api_main.app) as client:
                    response = client.post("/api/ask", json={"question": "Q", "history": []})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "## 結論\n借手は使用権資産を計上します[1]。")
        self.assertEqual(payload["metadata"]["request_id"], response.headers["X-Request-ID"])
        self.assertEqual(payload["metadata"]["chunks_read_count"], 1)
        self.assertEqual(payload["metadata"]["read_chunk_count"], 1)
        self.assertFalse(payload["metadata"]["uncertainty"]["present"])
        self.assertEqual(payload["metadata"]["evidence_coverage"]["covered_slots"], ["借手"])
        self.assertEqual(payload["references"][0]["display_number"], 1)
        self.assertEqual(payload["references"][0]["url"], "https://example.test/std13.pdf#page=5")
        self.assertEqual(payload["references"][0]["doc_type"], "企業会計基準")
        self.assertEqual(payload["references"][0]["doc_title"], "企業会計基準第13号")
        self.assertEqual(payload["references"][0]["section_label"], "第10項")
        self.assertEqual(
            payload["source_url_map"]["企業会計基準第13号"],
            "https://example.test/std13.pdf#page=5",
        )
        events = self._structured_events(logs.output)
        completion = next(event for event in events if event["event"] == "api_request_complete")
        self.assertEqual(completion["path"], "/api/ask")
        self.assertEqual(completion["request_id"], response.headers["X-Request-ID"])
        self.assertEqual(completion["loops"], 2)
        self.assertEqual(completion["retrieved_tokens"], 321)
        self.assertEqual(completion["reference_count"], 1)

    def test_ask_stream_emits_reference_events_and_done_metadata(self):
        with patch("src.api.main.get_agent", return_value=FakeAgent()):
            with self.assertLogs("src.api.main", level="INFO") as logs:
                with TestClient(api_main.app) as client:
                    with client.stream("POST", "/api/ask/stream", json={"question": "Q", "history": []}) as response:
                        self.assertEqual(response.status_code, 200)
                        request_id = response.headers["X-Request-ID"]
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
        self.assertFalse(events[4]["data"]["uncertainty"]["present"])
        self.assertEqual(events[4]["data"]["evidence_coverage"]["covered_slots"], ["借手"])
        self.assertEqual(events[4]["data"]["request_id"], request_id)
        structured = self._structured_events(logs.output)
        completion = next(event for event in structured if event["event"] == "api_request_complete")
        self.assertEqual(completion["path"], "/api/ask/stream")
        self.assertTrue(completion["stream"])
        self.assertEqual(completion["request_id"], request_id)

    def test_ask_logs_zero_reference_answers(self):
        class ZeroReferenceAgent(FakeAgent):
            async def arun(self, question, history):
                result = await super().arun(question, history)
                result["cited_reference_count"] = 0
                result["references"] = []
                return result

        with patch("src.api.main.get_agent", return_value=ZeroReferenceAgent()):
            with self.assertLogs("src.api.main", level="INFO") as logs:
                with TestClient(api_main.app) as client:
                    response = client.post("/api/ask", json={"question": "Q", "history": []})

        self.assertEqual(response.status_code, 200)
        events = self._structured_events(logs.output)
        completion = next(event for event in events if event["event"] == "api_request_complete")
        self.assertTrue(completion["zero_reference"])

    def test_ask_logs_error_events_on_timeout(self):
        class FailingAgent:
            async def arun(self, question, history):
                raise TimeoutError("llm timeout")

        with patch("src.api.main.get_agent", return_value=FailingAgent()):
            with self.assertLogs("src.api.main", level="ERROR") as logs:
                with TestClient(api_main.app, raise_server_exceptions=False) as client:
                    response = client.post("/api/ask", json={"question": "Q", "history": []})

        self.assertEqual(response.status_code, 500)
        events = self._structured_events(logs.output)
        error = next(event for event in events if event["event"] == "api_request_error")
        self.assertEqual(error["error_type"], "TimeoutError")
        self.assertEqual(error["path"], "/api/ask")

    def test_ui_event_endpoint_logs_reference_open(self):
        with patch("src.api.main.get_agent", return_value=FakeAgent()):
            with self.assertLogs("src.api.main", level="INFO") as logs:
                with TestClient(api_main.app) as client:
                    response = client.post(
                        "/api/ui-event",
                        json={
                            "event": "reference_opened",
                            "request_id": "req-test",
                            "reference_number": 2,
                            "reference_id": "std13.pdf:c12",
                            "reference_count": 3,
                            "uncertainty_present": False,
                            "developer_mode": True,
                        },
                    )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        events = self._structured_events(logs.output)
        ui_event = next(event for event in events if event["event"] == "ui_event")
        self.assertEqual(ui_event["event_name"], "reference_opened")
        self.assertEqual(ui_event["request_id"], "req-test")
        self.assertEqual(ui_event["reference_number"], 2)
        self.assertEqual(ui_event["reference_id"], "std13.pdf:c12")


if __name__ == "__main__":
    unittest.main()
