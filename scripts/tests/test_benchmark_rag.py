import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).resolve().parents[1] / "benchmark_rag.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_thinking_is_not_answer_latency_and_comments_are_ignored(self):
        ticks = iter([10.01, 12.0, 12.2, 12.5])
        result = benchmark.consume_sse([
            b": heartbeat\r\n", b"\r\n",
            b'data: {"type":"thinking","stage":"agentic_plan","details":{"placeholder":true}}\r\n', b"\r\n",
            b'data: {"type":"response","content":"  "}\n', b"\n",
            b'data: {"type":"response","content":"answer"}\n', b"\n",
            b'data: {"type":"done"}\n', b"\n",
        ], lambda: next(ticks), 10)
        self.assertAlmostEqual(result["first_event_ms"], 10)
        self.assertAlmostEqual(result["first_answer_ms"], 2200)
        self.assertAlmostEqual(result["done_ms"], 2500)
        self.assertTrue(result["done"])

    def test_multiline_data_and_unicode_sources(self):
        result = benchmark.consume_sse([
            'data: {"type":"sources",\n', 'data: "sources":[{"title":"资料.pdf","page":2}]}\n', '\n',
            'data: {"type":"response","content":"林乔 [1]"}\n', '\n',
        ], lambda: 1, 0)
        self.assertEqual(result["sources"], [{"title": "资料.pdf", "page": 2}])
        self.assertEqual(result["answer"], "林乔 [1]")
        self.assertFalse(result["done"])

    def test_broken_final_frame_is_not_counted_as_done(self):
        result = benchmark.consume_sse([b'data: {"type":"done"}\n'], lambda: 1, 0)
        self.assertFalse(result["done"])
        self.assertIsNone(result["first_answer_ms"])

    def test_error_frames_are_counted_without_exposing_payload(self):
        result = benchmark.consume_sse([b'data: {"type":"error","content":"private trace"}\n', b'\n'], lambda: 1, 0)
        self.assertEqual(result["error_count"], 1)
        self.assertNotIn("private trace", str(result))

    def test_nearest_rank_percentiles_keep_sample_count(self):
        self.assertEqual(benchmark.summary([100, None, 20, 30]), {"n": 3, "min": 20, "p50": 30, "p95": 100, "max": 100})
        self.assertIsNone(benchmark.summary([]))


if __name__ == "__main__":
    unittest.main()
