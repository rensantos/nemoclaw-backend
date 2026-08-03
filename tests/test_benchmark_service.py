import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from services.benchmark import BenchmarkError, BenchmarkService
from services.gpu import GPUInfo


class FakeModelManager:
    def selected_model_id(self):
        return "tiny"

    def current_model(self):
        return {
            "id": "tiny",
            "name": "Tiny",
            "path": "Tiny/Tiny",
            "engine": "transformers",
        }


class FakeGPUManager:
    def __init__(self, snapshots=None):
        self.snapshots = snapshots or []
        self.calls = 0

    def detect_gpus(self):
        if not self.snapshots:
            return []
        index = min(self.calls, len(self.snapshots) - 1)
        self.calls += 1
        return [self.snapshots[index]]


class FakeBenchmarkService(BenchmarkService):
    def __init__(self, results, gpu_manager=None):
        config = SimpleNamespace(
            backend=SimpleNamespace(host="127.0.0.1", port=8000, gpu="0"),
            model=SimpleNamespace(id="tiny", temperature_default=0.7),
        )
        super().__init__(
            config=config,
            model_manager=FakeModelManager(),
            gpu_manager=gpu_manager or FakeGPUManager(),
        )
        self.results = list(results)

    def _run_chat_once(self, prompt, max_tokens):
        return self.results.pop(0)


def gpu_snapshot(used, free):
    return GPUInfo(
        index="0",
        name="RTX A4000",
        memory_total_mib=16384,
        memory_used_mib=used,
        memory_free_mib=free,
        temperature_c=45,
        utilization_percent=12,
        driver_version="535.0",
    )


class BenchmarkServiceTests(unittest.TestCase):
    def test_latency_reports_average_min_and_max(self):
        service = FakeBenchmarkService([
            {"latency_seconds": 1.0, "completion_tokens": 10},
            {"latency_seconds": 2.0, "completion_tokens": 10},
            {"latency_seconds": 3.0, "completion_tokens": 10},
        ])

        result = service.latency("hello", max_tokens=8, runs=3)

        self.assertEqual(result["benchmark"], "latency")
        self.assertEqual(result["model"], "tiny")
        self.assertEqual(result["average_seconds"], 2.0)
        self.assertEqual(result["min_seconds"], 1.0)
        self.assertEqual(result["max_seconds"], 3.0)

    def test_throughput_reports_request_and_token_rates(self):
        service = FakeBenchmarkService([
            {"latency_seconds": 1.0, "completion_tokens": 10},
            {"latency_seconds": 1.0, "completion_tokens": 14},
        ])

        with mock.patch("services.benchmark.time.perf_counter", side_effect=[10.0, 14.0]):
            result = service.throughput("hello", max_tokens=8, runs=2)

        self.assertEqual(result["benchmark"], "throughput")
        self.assertEqual(result["elapsed_seconds"], 4.0)
        self.assertEqual(result["requests_per_second"], 0.5)
        self.assertEqual(result["tokens_per_second"], 6.0)

    def test_vram_reports_before_peak_and_after(self):
        gpu_manager = FakeGPUManager([
            gpu_snapshot(1000, 15384),
            gpu_snapshot(1400, 14984),
            gpu_snapshot(1200, 15184),
        ])
        service = FakeBenchmarkService([
            {"latency_seconds": 1.0, "completion_tokens": 10},
        ], gpu_manager=gpu_manager)

        result = service.vram("hello", max_tokens=8, runs=1)

        self.assertEqual(result["benchmark"], "vram")
        self.assertEqual(result["vram_before"]["memory_used_mib"], 1000)
        self.assertEqual(result["vram_peak"]["memory_used_mib"], 1400)
        self.assertEqual(result["vram_after"]["memory_used_mib"], 1200)

    def test_concurrency_is_accepted_but_reported_as_sequential(self):
        service = FakeBenchmarkService([
            {"latency_seconds": 1.0, "completion_tokens": 10},
        ])

        result = service.latency("hello", max_tokens=8, runs=1, concurrency=3)

        self.assertEqual(result["concurrency"], 3)
        self.assertIn("currently runs requests sequentially", result["concurrency_note"])

    def test_invalid_options_raise_value_error(self):
        service = FakeBenchmarkService([])

        with self.assertRaises(ValueError):
            service.latency("hello", max_tokens=0)


if __name__ == "__main__":
    unittest.main()


def _sse(events):
    """Encodes chunk payloads as the SSE body the backend produces."""
    body = "".join("data: {}\n\n".format(json.dumps(e)) for e in events)
    body += "data: [DONE]\n\n"

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    return _Response(body.encode("utf-8"))


def _chunk(delta=None, finish_reason=None, usage=None):
    body = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "qwen3:30b",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }
    if usage:
        body["usage"] = usage
    return body


class FirstTokenLatencyTests(unittest.TestCase):
    """Measured over a real stream now that streaming exists. Reasoning
    deltas are excluded: timing them would flatter the number while saying
    nothing about when the user sees an answer."""

    def _service(self):
        config = SimpleNamespace(
            backend=SimpleNamespace(host="127.0.0.1", port=8000, gpu="0"),
            model=SimpleNamespace(id="tiny", temperature_default=0.7),
        )
        return BenchmarkService(
            config=config,
            model_manager=FakeModelManager(),
            gpu_manager=FakeGPUManager(),
        )

    def _run(self, events, runs=1):
        service = self._service()
        with mock.patch(
            "services.benchmark.urllib.request.urlopen",
            side_effect=lambda *a, **k: _sse(events),
        ):
            return service.first_token_latency("hi", 32, runs=runs, concurrency=1)

    def test_reports_a_real_measurement(self):
        events = [
            _chunk({"role": "assistant"}),
            _chunk({"content": "Lis"}),
            _chunk({"content": "bon"}),
            _chunk({}, finish_reason="stop",
                   usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}),
        ]
        result = self._run(events)

        self.assertTrue(result["available"])
        self.assertIsNotNone(result["average_seconds"])
        self.assertEqual(result["results"][0]["total_tokens"], 7)
        self.assertEqual(result["results"][0]["finish_reason"], "stop")

    def test_reasoning_does_not_count_as_the_first_token(self):
        events = [
            _chunk({"role": "assistant"}),
            _chunk({"reasoning": "thinking hard"}),
            _chunk({"content": "Lisbon"}),
            _chunk({}, finish_reason="stop", usage={}),
        ]
        result = self._run(events)

        run = result["results"][0]
        self.assertIsNotNone(run["first_reasoning_seconds"])
        self.assertIsNotNone(run["first_token_seconds"])
        # Reasoning arrived first, so the answer must be no earlier than it.
        self.assertGreaterEqual(
            run["first_token_seconds"], run["first_reasoning_seconds"]
        )

    def test_reports_unavailable_when_only_reasoning_is_emitted(self):
        """Counting a reasoning-only stream as zero would be a fabricated
        number; say so instead."""
        events = [
            _chunk({"role": "assistant"}),
            _chunk({"reasoning": "thinking forever"}),
            _chunk({}, finish_reason="stop", usage={}),
        ]
        result = self._run(events)

        self.assertFalse(result["available"])
        self.assertIn("only reasoning", result["message"])

    def test_requests_a_stream(self):
        service = self._service()
        with mock.patch(
            "services.benchmark.urllib.request.urlopen",
            side_effect=lambda *a, **k: _sse([_chunk({"content": "x"})]),
        ) as urlopen:
            service.first_token_latency("hi", 32, runs=1, concurrency=1)

        sent = json.loads(urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertTrue(sent["stream"])

    def test_surfaces_a_mid_stream_error(self):
        service = self._service()
        events = [_chunk({"content": "partial"}), {"error": {"message": "daemon died"}}]
        with mock.patch(
            "services.benchmark.urllib.request.urlopen",
            side_effect=lambda *a, **k: _sse(events),
        ):
            with self.assertRaisesRegex(BenchmarkError, "daemon died"):
                service.first_token_latency("hi", 32, runs=1, concurrency=1)
