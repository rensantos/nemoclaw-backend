import json
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from config import config as default_config
from services.gpu import GPUManager
from services.model import ModelManager


class BenchmarkError(Exception):
    """Raised when a benchmark cannot reach or parse the backend."""


class BenchmarkService:
    """Owns benchmark execution against the OpenAI-compatible HTTP API."""

    def __init__(
        self,
        config=default_config,
        model_manager: Optional[ModelManager] = None,
        gpu_manager: Optional[GPUManager] = None,
        timeout_seconds: int = 120,
    ):
        self.config = config
        self.model_manager = model_manager or ModelManager()
        self.gpu_manager = gpu_manager or GPUManager(config)
        self.timeout_seconds = timeout_seconds

    def latency(
        self,
        prompt: str,
        max_tokens: int,
        runs: int = 3,
        concurrency: int = 1,
    ) -> Dict[str, object]:
        self._validate_options(max_tokens, runs, concurrency)
        results = self._run_chat_benchmark(prompt, max_tokens, runs)
        latencies = [result["latency_seconds"] for result in results]

        return self._base_result("latency", prompt, max_tokens, runs, concurrency, results, {
            "average_seconds": self._average(latencies),
            "min_seconds": min(latencies),
            "max_seconds": max(latencies),
        })

    def throughput(
        self,
        prompt: str,
        max_tokens: int,
        runs: int = 3,
        concurrency: int = 1,
    ) -> Dict[str, object]:
        self._validate_options(max_tokens, runs, concurrency)
        started = time.perf_counter()
        results = self._run_chat_benchmark(prompt, max_tokens, runs)
        elapsed = time.perf_counter() - started
        completion_tokens = self._sum_known(results, "completion_tokens")

        metrics = {
            "elapsed_seconds": elapsed,
            "requests_per_second": runs / elapsed if elapsed > 0 else None,
            "tokens_per_second": (
                completion_tokens / elapsed
                if completion_tokens is not None and elapsed > 0
                else None
            ),
        }

        return self._base_result("throughput", prompt, max_tokens, runs, concurrency, results, metrics)

    def vram(
        self,
        prompt: str,
        max_tokens: int,
        runs: int = 3,
        concurrency: int = 1,
    ) -> Dict[str, object]:
        self._validate_options(max_tokens, runs, concurrency)
        before = self._selected_gpu_snapshot()
        snapshots = [before]
        results = []

        for _ in range(runs):
            results.append(self._run_chat_once(prompt, max_tokens))
            snapshots.append(self._selected_gpu_snapshot())

        after = self._selected_gpu_snapshot()
        snapshots.append(after)

        metrics = {
            "vram_before": before,
            "vram_peak": self._peak_snapshot(snapshots),
            "vram_after": after,
        }

        return self._base_result("vram", prompt, max_tokens, runs, concurrency, results, metrics)

    def first_token_latency(
        self,
        prompt: str,
        max_tokens: int,
        runs: int = 3,
        concurrency: int = 1,
    ) -> Dict[str, object]:
        self._validate_options(max_tokens, runs, concurrency)
        result = self._base_result(
            "first-token-latency",
            prompt,
            max_tokens,
            runs,
            concurrency,
            [],
            {
                "available": False,
                "message": (
                    "First-token latency is unavailable because streaming is "
                    "not implemented by this backend yet."
                ),
            },
        )
        return result

    def _run_chat_benchmark(self, prompt: str, max_tokens: int, runs: int) -> List[Dict[str, object]]:
        return [self._run_chat_once(prompt, max_tokens) for _ in range(runs)]

    def _run_chat_once(self, prompt: str, max_tokens: int) -> Dict[str, object]:
        started = time.perf_counter()
        response = self._post_chat_completion(prompt, max_tokens)
        elapsed = time.perf_counter() - started
        usage = response.get("usage") or {}

        return {
            "latency_seconds": elapsed,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "finish_reason": self._finish_reason(response),
        }

    def _post_chat_completion(self, prompt: str, max_tokens: int) -> Dict[str, object]:
        payload = {
            "model": self._current_model_id(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.config.model.temperature_default,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._chat_url(),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BenchmarkError("Backend returned HTTP {}: {}".format(exc.code, detail))
        except urllib.error.URLError as exc:
            raise BenchmarkError("Backend is unavailable: {}".format(exc))

        try:
            decoded = json.loads(raw)
        except ValueError:
            raise BenchmarkError("Backend returned invalid JSON")

        if not isinstance(decoded, dict):
            raise BenchmarkError("Backend returned an unexpected response")
        return decoded

    def _base_result(
        self,
        name: str,
        prompt: str,
        max_tokens: int,
        runs: int,
        concurrency: int,
        results: List[Dict[str, object]],
        metrics: Dict[str, object],
    ) -> Dict[str, object]:
        result = {
            "benchmark": name,
            "model": self._current_model_id(),
            "model_info": self._current_model_info(),
            "gpu": self.config.backend.gpu,
            "endpoint": self._chat_url(),
            "prompt": prompt,
            "max_tokens": max_tokens,
            "runs": runs,
            "concurrency": concurrency,
            "results": results,
        }
        result.update(metrics)
        if concurrency != 1:
            result["concurrency_note"] = (
                "Concurrency is accepted for CLI/API stability, but this "
                "implementation currently runs requests sequentially."
            )
        return result

    def _selected_gpu_snapshot(self) -> Dict[str, object]:
        selected_index = str(self.config.backend.gpu)
        selected = None
        for gpu in self.gpu_manager.detect_gpus():
            if str(gpu.index) == selected_index:
                selected = gpu
                break

        if selected is None:
            return {
                "gpu": selected_index,
                "memory_total_mib": None,
                "memory_used_mib": None,
                "memory_free_mib": None,
                "temperature_c": None,
                "utilization_percent": None,
            }

        return {
            "gpu": selected.index,
            "name": selected.name,
            "memory_total_mib": selected.memory_total_mib,
            "memory_used_mib": selected.memory_used_mib,
            "memory_free_mib": selected.memory_free_mib,
            "temperature_c": selected.temperature_c,
            "utilization_percent": selected.utilization_percent,
        }

    def _peak_snapshot(self, snapshots: List[Dict[str, object]]) -> Dict[str, object]:
        known = [
            snapshot for snapshot in snapshots
            if snapshot.get("memory_used_mib") is not None
        ]
        if not known:
            return snapshots[-1]
        return max(known, key=lambda snapshot: snapshot["memory_used_mib"])

    def _current_model_id(self) -> str:
        try:
            return self.model_manager.selected_model_id()
        except Exception:
            return self.config.model.id

    def _current_model_info(self):
        try:
            return self.model_manager.current_model()
        except Exception:
            return {"id": self._current_model_id(), "engine": "transformers"}

    def _chat_url(self) -> str:
        host = self.config.backend.host
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        return "http://{}:{}/v1/chat/completions".format(host, self.config.backend.port)

    def _validate_options(self, max_tokens: int, runs: int, concurrency: int) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if runs < 1:
            raise ValueError("runs must be at least 1")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")

    def _average(self, values: List[float]) -> float:
        return sum(values) / len(values)

    def _sum_known(self, results: List[Dict[str, object]], key: str):
        values = [result.get(key) for result in results]
        if any(value is None for value in values):
            return None
        return sum(values)

    def _finish_reason(self, response: Dict[str, object]):
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        return first.get("finish_reason")
