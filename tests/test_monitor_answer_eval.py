import unittest

from scripts.monitor_answer_eval import collect_threshold_failures


class MonitorAnswerEvalTests(unittest.TestCase):
    def test_collect_threshold_failures_returns_all_regressions(self):
        summary = {
            "avg_latency_sec": 41.2,
            "p95_latency_sec": 65.0,
            "avg_loops": 6.1,
        }
        thresholds = {
            "avg_latency_sec": 35.0,
            "p95_latency_sec": 60.0,
            "avg_loops": 5.0,
        }

        failures = collect_threshold_failures(summary, thresholds)

        self.assertEqual(
            failures,
            [
                "avg_latency_sec>35.0 (41.2)",
                "p95_latency_sec>60.0 (65.0)",
                "avg_loops>5.0 (6.1)",
            ],
        )

    def test_collect_threshold_failures_returns_empty_when_within_budget(self):
        summary = {
            "avg_latency_sec": 30.0,
            "p95_latency_sec": 58.0,
            "avg_loops": 4.8,
        }
        thresholds = {
            "avg_latency_sec": 35.0,
            "p95_latency_sec": 60.0,
            "avg_loops": 5.0,
        }

        failures = collect_threshold_failures(summary, thresholds)

        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
