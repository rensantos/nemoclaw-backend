import io
import json
import sys
import types
import unittest
import urllib.error
from unittest import mock

from config import BackendConfig, Config, ModelConfig
from engines.base import EngineUnavailableError, ModelNotFoundError
from services.gpu import GPUInfo
from services.inference import InferenceService, _build_engine
from services.lifecycle import LifecycleState


def _make_config(
    engine,
    ollama_host="http://127.0.0.1:11434",
    model_id="test-model",
    think_default=None,
):
    return Config(
        backend=BackendConfig(
            host="127.0.0.1",
            port=8000,
            gpu="0",
            engine=engine,
            ollama_host=ollama_host,
            instance="test-instance",
        ),
        model=ModelConfig(
            id=model_id,
            max_tokens_default=16,
            temperature_default=0.1,
            quantization="none",
            revision="",
            think_default=think_default,
        ),
    )


def _fake_urlopen_response(payload: dict):
    body = json.dumps(payload).encode("utf-8")

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    return _Response(body)


class EngineFactoryTests(unittest.TestCase):
    def test_transformers_selected_for_transformers(self):
        fake_module = types.ModuleType("engines.transformers_engine")

        class FakeTransformersEngine:
            def __init__(self, config):
                self.config = config

        fake_module.TransformersEngine = FakeTransformersEngine

        with mock.patch.dict(sys.modules, {"engines.transformers_engine": fake_module}):
            engine = _build_engine(_make_config("transformers"))

        self.assertIsInstance(engine, FakeTransformersEngine)
        self.assertEqual(engine.config.backend.engine, "transformers")

    def test_ollama_selected_for_ollama(self):
        from engines.ollama_engine import OllamaEngine

        engine = _build_engine(_make_config("ollama"))

        self.assertIsInstance(engine, OllamaEngine)

    def test_unknown_engine_name_raises(self):
        with self.assertRaises(ValueError):
            _build_engine(_make_config("bogus"))


class OllamaEngineReadPathTests(unittest.TestCase):
    """docs/ollama-engine-design.md Increment 2: health(), list_models(),
    load_model() against a (mocked) live Ollama daemon."""

    def _engine(self, model_id="test-model"):
        from engines.ollama_engine import OllamaEngine

        return OllamaEngine(_make_config("ollama", model_id=model_id))

    def _mock_urlopen(self, payload=None, raise_url_error=False):
        if raise_url_error:
            return mock.patch(
                "engines.ollama_engine.urllib.request.urlopen",
                side_effect=urllib.error.URLError("connection refused"),
            )
        # A fresh response per call: each `with urlopen(...)` closes the one
        # it got, so a single reused return_value breaks any engine method
        # that makes more than one request (e.g. switch_model).
        return mock.patch(
            "engines.ollama_engine.urllib.request.urlopen",
            side_effect=lambda *args, **kwargs: _fake_urlopen_response(payload),
        )

    def test_constructs_without_error(self):
        engine = self._engine()

        self.assertIsNotNone(engine)
        self.assertEqual(engine.base_url, "http://127.0.0.1:11434")

    def test_load_model_succeeds_when_tag_present(self):
        engine = self._engine(model_id="qwen3:1.7b")
        payload = {"models": [{"name": "qwen3:1.7b"}, {"name": "qwen3:4b"}]}

        with self._mock_urlopen(payload):
            engine.load_model()  # must not raise

    def test_load_model_raises_clear_error_when_tag_missing(self):
        engine = self._engine(model_id="llama3:8b")
        payload = {"models": [{"name": "qwen3:1.7b"}]}

        with self._mock_urlopen(payload):
            with self.assertRaisesRegex(RuntimeError, "ollama pull llama3:8b"):
                engine.load_model()

    def test_load_model_raises_engine_unavailable_when_daemon_unreachable(self):
        engine = self._engine()

        with self._mock_urlopen(raise_url_error=True):
            with self.assertRaises(EngineUnavailableError):
                engine.load_model()

    def test_health_reports_model_and_gpu_fields_when_daemon_reachable(self):
        engine = self._engine(model_id="qwen3:1.7b")
        payload = {"models": [{"name": "qwen3:1.7b"}]}

        with self._mock_urlopen(payload), mock.patch.object(
            engine, "_gpu_snapshot", return_value=(False, None)
        ):
            health = engine.health()

        self.assertEqual(
            health, {"model": "qwen3:1.7b", "cuda": False, "gpu": None}
        )

    def test_health_raises_engine_unavailable_with_partial_health_when_daemon_down(self):
        engine = self._engine(model_id="qwen3:1.7b")

        with self._mock_urlopen(raise_url_error=True), mock.patch.object(
            engine, "_gpu_snapshot", return_value=(False, None)
        ):
            with self.assertRaises(EngineUnavailableError) as ctx:
                engine.health()

        self.assertEqual(
            ctx.exception.partial_health,
            {"model": "qwen3:1.7b", "cuda": False, "gpu": None},
        )

    def test_list_models_returns_only_the_configured_tag(self):
        engine = self._engine(model_id="qwen3:1.7b")
        payload = {
            "models": [{"name": "qwen3:1.7b"}, {"name": "nomic-embed-text:latest"}]
        }

        with self._mock_urlopen(payload):
            result = engine.list_models()

        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["data"][0]["id"], "qwen3:1.7b")
        self.assertEqual(result["data"][0]["owned_by"], "ollama")

    def test_unload_model_sends_keep_alive_zero(self):
        engine = self._engine(model_id="qwen3:1.7b")

        with self._mock_urlopen({"done": True}) as mocked_urlopen:
            engine.unload_model()  # must not raise

        request = mocked_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent, {"model": "qwen3:1.7b", "keep_alive": 0})

    def test_unload_model_raises_engine_unavailable_when_daemon_unreachable(self):
        engine = self._engine()

        with self._mock_urlopen(raise_url_error=True):
            with self.assertRaises(EngineUnavailableError):
                engine.unload_model()

    def test_supports_runtime_lifecycle(self):
        self.assertTrue(self._engine().supports_runtime_lifecycle)

    def test_load_model_with_explicit_id_repoints_the_engine(self):
        engine = self._engine(model_id="qwen3:1.7b")
        payload = {"models": [{"name": "qwen3:1.7b"}, {"name": "qwen3:4b"}]}

        with self._mock_urlopen(payload):
            engine.load_model("qwen3:4b")

        self.assertEqual(engine.model_id, "qwen3:4b")

    def test_load_model_leaves_model_id_untouched_when_target_missing(self):
        engine = self._engine(model_id="qwen3:1.7b")
        payload = {"models": [{"name": "qwen3:1.7b"}]}

        with self._mock_urlopen(payload):
            with self.assertRaisesRegex(RuntimeError, "ollama pull llama3:8b"):
                engine.load_model("llama3:8b")

        self.assertEqual(engine.model_id, "qwen3:1.7b")

    def test_switch_model_verifies_target_before_unloading_the_old_one(self):
        engine = self._engine(model_id="qwen3:1.7b")
        payload = {"models": [{"name": "qwen3:1.7b"}]}

        with self._mock_urlopen(payload) as mocked_urlopen:
            with self.assertRaisesRegex(RuntimeError, "ollama pull llama3:8b"):
                engine.switch_model("llama3:8b")

        # Only the tag check ran; the old model was never released, so it
        # keeps serving.
        self.assertEqual(engine.model_id, "qwen3:1.7b")
        called_urls = [call[0][0].full_url for call in mocked_urlopen.call_args_list]
        self.assertEqual(called_urls, ["http://127.0.0.1:11434/api/tags"])

    def test_switch_model_unloads_old_then_repoints(self):
        engine = self._engine(model_id="qwen3:1.7b")
        payload = {"models": [{"name": "qwen3:1.7b"}, {"name": "qwen3:4b"}]}

        with self._mock_urlopen(payload) as mocked_urlopen:
            engine.switch_model("qwen3:4b")

        self.assertEqual(engine.model_id, "qwen3:4b")
        requests = mocked_urlopen.call_args_list
        self.assertEqual(requests[0][0][0].full_url, "http://127.0.0.1:11434/api/tags")
        unload = requests[1][0][0]
        self.assertEqual(unload.full_url, "http://127.0.0.1:11434/api/generate")
        # The old tag is the one evicted, not the incoming one.
        self.assertEqual(
            json.loads(unload.data.decode("utf-8")),
            {"model": "qwen3:1.7b", "keep_alive": 0},
        )


def _message(role, content):
    return types.SimpleNamespace(role=role, content=content)


class OllamaEngineChatTests(unittest.TestCase):
    """docs/ollama-engine-design.md Increment 3: chat()/generate_text()
    against a (mocked) live Ollama daemon."""

    def _engine(self, model_id="qwen3:1.7b", think_default=None):
        from engines.ollama_engine import OllamaEngine

        return OllamaEngine(
            _make_config("ollama", model_id=model_id, think_default=think_default)
        )

    def _mock_urlopen(self, payload=None, raise_url_error=False):
        if raise_url_error:
            return mock.patch(
                "engines.ollama_engine.urllib.request.urlopen",
                side_effect=urllib.error.URLError("connection refused"),
            )
        # A fresh response per call: each `with urlopen(...)` closes the one
        # it got, so a single reused return_value breaks any engine method
        # that makes more than one request (e.g. switch_model).
        return mock.patch(
            "engines.ollama_engine.urllib.request.urlopen",
            side_effect=lambda *args, **kwargs: _fake_urlopen_response(payload),
        )

    def test_chat_returns_content_and_token_usage(self):
        engine = self._engine()
        payload = {
            "message": {"role": "assistant", "content": "hello there"},
            "prompt_eval_count": 5,
            "eval_count": 3,
        }

        with self._mock_urlopen(payload):
            result = engine.chat([_message("user", "hi")], None, None)

        self.assertEqual(result["content"], "hello there")
        self.assertEqual(result["prompt_tokens"], 5)
        self.assertEqual(result["completion_tokens"], 3)
        self.assertEqual(result["total_tokens"], 8)

    def test_chat_sends_expected_payload(self):
        engine = self._engine()
        payload = {
            "message": {"content": "hi"},
            "prompt_eval_count": 1,
            "eval_count": 1,
        }

        with self._mock_urlopen(payload) as mocked_urlopen:
            engine.chat([_message("user", "hi")], 64, 0.3)

        request = mocked_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["model"], "qwen3:1.7b")
        self.assertEqual(sent["messages"], [{"role": "user", "content": "hi"}])
        self.assertFalse(sent["stream"])
        self.assertEqual(sent["options"], {"temperature": 0.3, "num_predict": 64})

    def test_chat_omits_think_by_default(self):
        engine = self._engine()
        payload = {"message": {"content": "hi"}, "prompt_eval_count": 1, "eval_count": 1}

        with self._mock_urlopen(payload) as mocked_urlopen:
            engine.chat([_message("user", "hi")], None, None)

        sent = json.loads(mocked_urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertNotIn("think", sent)

    def test_chat_request_think_overrides_config_default(self):
        engine = self._engine(think_default=True)
        payload = {"message": {"content": "hi"}, "prompt_eval_count": 1, "eval_count": 1}

        with self._mock_urlopen(payload) as mocked_urlopen:
            engine.chat([_message("user", "hi")], None, None, think=False)

        sent = json.loads(mocked_urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(sent["think"], False)

    def test_chat_falls_back_to_config_think_default(self):
        engine = self._engine(think_default=False)
        payload = {"message": {"content": "hi"}, "prompt_eval_count": 1, "eval_count": 1}

        with self._mock_urlopen(payload) as mocked_urlopen:
            engine.chat([_message("user", "hi")], None, None)

        sent = json.loads(mocked_urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(sent["think"], False)

    def test_chat_reports_zero_and_does_not_raise_when_usage_missing(self):
        engine = self._engine()
        payload = {"message": {"content": "hi"}}

        with self._mock_urlopen(payload):
            result = engine.chat([_message("user", "hi")], None, None)

        self.assertEqual(result["prompt_tokens"], 0)
        self.assertEqual(result["completion_tokens"], 0)
        self.assertEqual(result["total_tokens"], 0)

    def test_chat_succeeds_when_requested_model_matches_servable_model(self):
        engine = self._engine()
        payload = {"message": {"content": "hi"}, "prompt_eval_count": 1, "eval_count": 1}

        with self._mock_urlopen(payload):
            engine.chat([_message("user", "hi")], None, None, requested_model="qwen3:1.7b")

    def test_chat_raises_model_not_found_when_requested_model_mismatches(self):
        engine = self._engine()

        with self.assertRaises(ModelNotFoundError) as ctx:
            engine.chat([_message("user", "hi")], None, None, requested_model="llama3:8b")

        self.assertEqual(ctx.exception.requested_model, "llama3:8b")
        self.assertEqual(ctx.exception.servable_model, "qwen3:1.7b")

    def test_chat_does_not_call_daemon_when_requested_model_mismatches(self):
        engine = self._engine()

        with self._mock_urlopen() as mocked_urlopen:
            with self.assertRaises(ModelNotFoundError):
                engine.chat([_message("user", "hi")], None, None, requested_model="llama3:8b")

        mocked_urlopen.assert_not_called()

    def test_chat_stream_raises_model_not_found_without_iterating(self):
        """Found live: chat_stream() was a generator function, so this
        guard did not run until the first delta was pulled - long after
        FastAPI had sent HTTP 200. The client saw the connection die with
        no [DONE] and no in-band error instead of a 404. The rejection has
        to happen on the call itself, before anything is consumed."""
        engine = self._engine()

        with self._mock_urlopen() as mocked_urlopen:
            with self.assertRaises(ModelNotFoundError) as ctx:
                engine.chat_stream(
                    [_message("user", "hi")], None, None, requested_model="llama3:8b"
                )

        self.assertEqual(ctx.exception.requested_model, "llama3:8b")
        mocked_urlopen.assert_not_called()

    def test_chat_raises_engine_unavailable_when_daemon_unreachable(self):
        engine = self._engine()

        with self._mock_urlopen(raise_url_error=True):
            with self.assertRaises(EngineUnavailableError):
                engine.chat([_message("user", "hi")], None, None)

    def test_generate_text_returns_completion_only(self):
        engine = self._engine()
        payload = {"response": "generated text"}

        with self._mock_urlopen(payload):
            result = engine.generate_text("prompt", 32, 0.5)

        self.assertEqual(
            result,
            {"model": "qwen3:1.7b", "response": "generated text", "reasoning": None},
        )

    def test_generate_text_sends_expected_payload(self):
        engine = self._engine()
        payload = {"response": "ok"}

        with self._mock_urlopen(payload) as mocked_urlopen:
            engine.generate_text("hello", 20, 0.9)

        request = mocked_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["prompt"], "hello")
        self.assertEqual(sent["options"], {"temperature": 0.9, "num_predict": 20})

    def test_generate_text_omits_think_by_default(self):
        engine = self._engine()
        payload = {"response": "ok"}

        with self._mock_urlopen(payload) as mocked_urlopen:
            engine.generate_text("hello", 20, 0.9)

        sent = json.loads(mocked_urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertNotIn("think", sent)

    def test_generate_text_request_think_overrides_config_default(self):
        engine = self._engine(think_default=True)
        payload = {"response": "ok"}

        with self._mock_urlopen(payload) as mocked_urlopen:
            engine.generate_text("hello", 20, 0.9, think=False)

        sent = json.loads(mocked_urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(sent["think"], False)

    def test_generate_text_raises_engine_unavailable_when_daemon_unreachable(self):
        engine = self._engine()

        with self._mock_urlopen(raise_url_error=True):
            with self.assertRaises(EngineUnavailableError):
                engine.generate_text("prompt", 32, 0.5)


class OllamaEngineInferenceServiceIntegrationTests(unittest.TestCase):
    """InferenceService wiring: startup validation and health degradation
    for a real (mocked) OllamaEngine, not the FakeEngine used elsewhere."""

    def _mock_urlopen(self, payload=None, raise_url_error=False):
        if raise_url_error:
            return mock.patch(
                "engines.ollama_engine.urllib.request.urlopen",
                side_effect=urllib.error.URLError("connection refused"),
            )
        # A fresh response per call: each `with urlopen(...)` closes the one
        # it got, so a single reused return_value breaks any engine method
        # that makes more than one request (e.g. switch_model).
        return mock.patch(
            "engines.ollama_engine.urllib.request.urlopen",
            side_effect=lambda *args, **kwargs: _fake_urlopen_response(payload),
        )

    def test_inference_service_constructs_when_tag_present_at_startup(self):
        from engines.ollama_engine import OllamaEngine

        engine = OllamaEngine(_make_config("ollama", model_id="qwen3:1.7b"))
        payload = {"models": [{"name": "qwen3:1.7b"}]}

        with self._mock_urlopen(payload):
            service = InferenceService(engine)

        self.assertEqual(service.lifecycle_state, LifecycleState.READY)

    def test_inference_service_fails_startup_when_tag_missing(self):
        from engines.ollama_engine import OllamaEngine

        engine = OllamaEngine(_make_config("ollama", model_id="missing-model"))
        payload = {"models": [{"name": "qwen3:1.7b"}]}

        with self._mock_urlopen(payload):
            with self.assertRaises(RuntimeError):
                InferenceService(engine)

    def test_health_transitions_lifecycle_to_degraded_when_daemon_goes_down(self):
        from engines.ollama_engine import OllamaEngine

        engine = OllamaEngine(_make_config("ollama", model_id="qwen3:1.7b"))
        payload = {"models": [{"name": "qwen3:1.7b"}]}

        with self._mock_urlopen(payload):
            service = InferenceService(engine)

        with self._mock_urlopen(raise_url_error=True), mock.patch.object(
            engine, "_gpu_snapshot", return_value=(False, None)
        ):
            health = service.health()

        self.assertEqual(service.lifecycle_state, LifecycleState.DEGRADED)
        self.assertEqual(health["status"], "degraded")
        self.assertEqual(health["lifecycle_state"], "degraded")
        self.assertEqual(health["model"], "qwen3:1.7b")


if __name__ == "__main__":
    unittest.main()


class OllamaDaemonDiscoveryTests(unittest.TestCase):
    """`pgrep -f "ollama serve"` also matches the shell the daemon was
    launched from, which would double-report every GPU finding."""

    def _pgrep(self, stdout):
        return mock.patch(
            "engines.ollama_engine.subprocess.run",
            return_value=types.SimpleNamespace(stdout=stdout, stderr=""),
        )

    def _cmdlines(self, mapping):
        def fake_open(path, *args, **kwargs):
            pid = int(str(path).split("/")[2])
            return mock.mock_open(read_data=mapping[pid])()

        return mock.patch("builtins.open", side_effect=fake_open)

    def test_excludes_the_shell_wrapper_that_merely_mentions_ollama(self):
        from engines.ollama_engine import find_daemon_pids

        cmdlines = {
            23823: b"bash\0-c\0cd ~/ollama && ./bin/ollama serve\0",
            23825: b"./bin/ollama\0serve\0",
        }
        with self._pgrep("23823\n23825\n"), self._cmdlines(cmdlines):
            self.assertEqual(find_daemon_pids(), [23825])

    def test_keeps_pid_when_cmdline_is_unreadable(self):
        from engines.ollama_engine import find_daemon_pids

        with self._pgrep("999\n"), mock.patch("builtins.open", side_effect=OSError):
            self.assertEqual(find_daemon_pids(), [999])

    def test_returns_empty_when_pgrep_is_unavailable(self):
        from engines.ollama_engine import find_daemon_pids

        with mock.patch(
            "engines.ollama_engine.subprocess.run", side_effect=OSError
        ):
            self.assertEqual(find_daemon_pids(), [])

    def test_runtime_pids_include_the_model_runner_child(self):
        """nvidia-smi attributes VRAM to the `ollama runner` child, not to
        `ollama serve`, so ownership checks need the whole tree."""
        from engines import ollama_engine

        with mock.patch.object(
            ollama_engine, "find_daemon_pids", return_value=[23825]
        ), mock.patch.object(
            ollama_engine,
            "_children_by_parent",
            return_value={23825: [17181], 17181: [17999]},
        ):
            self.assertEqual(
                ollama_engine.find_runtime_pids(), [17181, 17999, 23825]
            )

    def test_runtime_pids_empty_when_no_daemon(self):
        from engines import ollama_engine

        with mock.patch.object(ollama_engine, "find_daemon_pids", return_value=[]):
            self.assertEqual(ollama_engine.find_runtime_pids(), [])

    def test_engine_exposes_runtime_pids(self):
        from engines.ollama_engine import OllamaEngine
        from engines import ollama_engine

        engine = OllamaEngine(_make_config("ollama"))
        with mock.patch.object(
            ollama_engine, "find_runtime_pids", return_value=[1, 2]
        ):
            self.assertEqual(engine.runtime_pids(), [1, 2])


class OllamaVRAMEstimationTests(unittest.TestCase):
    def _engine(self, model_id="qwen3:30b"):
        from engines.ollama_engine import OllamaEngine

        return OllamaEngine(_make_config("ollama", model_id=model_id))

    def _tags(self, payload):
        return mock.patch(
            "engines.ollama_engine.urllib.request.urlopen",
            side_effect=lambda *a, **k: _fake_urlopen_response(payload),
        )

    def test_reads_disk_size_for_the_configured_tag(self):
        engine = self._engine()
        payload = {"models": [
            {"name": "qwen3:30b", "size": 18556699314},
            {"name": "other:1b", "size": 1000},
        ]}

        with self._tags(payload):
            self.assertEqual(engine.model_disk_size_mib(), 17697)

    def test_estimate_scales_disk_size_by_the_overhead_factor(self):
        from engines.ollama_engine import VRAM_OVERHEAD_FACTOR

        engine = self._engine()
        payload = {"models": [{"name": "qwen3:30b", "size": 18556699314}]}

        with self._tags(payload):
            estimate = engine.estimated_vram_mib()

        self.assertEqual(estimate, int(17697 * VRAM_OVERHEAD_FACTOR))
        # Must cover what Ollama actually reported on UBI (25.2 GiB).
        self.assertGreaterEqual(estimate, int(25.2 * 1024))

    def test_returns_none_for_an_unknown_tag(self):
        engine = self._engine()
        with self._tags({"models": [{"name": "other:1b", "size": 1000}]}):
            self.assertIsNone(engine.model_disk_size_mib())
            self.assertIsNone(engine.estimated_vram_mib())


class OllamaDaemonRestartTests(unittest.TestCase):
    def test_launch_spec_reads_argv_and_env_from_proc(self):
        from engines import ollama_engine

        contents = {
            "/proc/23825/cmdline": b"./bin/ollama\0serve\0",
            "/proc/23825/environ": b"CUDA_VISIBLE_DEVICES=0,1,2,3\0OLLAMA_HOST=127.0.0.1:11434\0",
        }

        def fake_open(path, *args, **kwargs):
            return mock.mock_open(read_data=contents[str(path)])()

        with mock.patch("builtins.open", side_effect=fake_open):
            argv, env = ollama_engine.daemon_launch_spec(23825)

        self.assertEqual(argv, ["./bin/ollama", "serve"])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(env["OLLAMA_HOST"], "127.0.0.1:11434")

    def test_launch_spec_returns_none_when_unreadable(self):
        from engines import ollama_engine

        with mock.patch("builtins.open", side_effect=OSError):
            self.assertIsNone(ollama_engine.daemon_launch_spec(23825))

    def test_restart_overrides_only_cuda_visible_devices(self):
        from engines import ollama_engine

        spec = (["./bin/ollama", "serve"], {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "OLLAMA_MODELS": "/home/d3894/ollama/models",
            "PWD": "/home/d3894/ollama",
        })

        with mock.patch.object(ollama_engine, "daemon_launch_spec", return_value=spec), \
                mock.patch.object(ollama_engine.os, "kill", side_effect=[None, OSError]), \
                mock.patch.object(ollama_engine.time, "sleep"), \
                mock.patch.object(ollama_engine.subprocess, "Popen") as popen:
            popen.return_value = types.SimpleNamespace(pid=99999)
            new_pid = ollama_engine.restart_daemon_pinned(23825, ["2", "3"])

        self.assertEqual(new_pid, 99999)
        _, kwargs = popen.call_args
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2,3")
        # Everything else about how it was launched is preserved.
        self.assertEqual(kwargs["env"]["OLLAMA_MODELS"], "/home/d3894/ollama/models")
        self.assertEqual(kwargs["cwd"], "/home/d3894/ollama")

    def test_restart_refuses_when_launch_command_is_unknown(self):
        from engines import ollama_engine

        with mock.patch.object(ollama_engine, "daemon_launch_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "refusing to restart it blind"):
                ollama_engine.restart_daemon_pinned(23825, ["2"])

    def test_restart_gives_up_if_the_daemon_will_not_exit(self):
        from engines import ollama_engine

        spec = (["./bin/ollama", "serve"], {})
        with mock.patch.object(ollama_engine, "daemon_launch_spec", return_value=spec), \
                mock.patch.object(ollama_engine.os, "kill", return_value=None), \
                mock.patch.object(ollama_engine.time, "sleep"), \
                mock.patch.object(ollama_engine.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "did not exit after SIGTERM"):
                ollama_engine.restart_daemon_pinned(23825, ["2"])

        popen.assert_not_called()

    def test_inherits_the_daemons_existing_log_destination(self):
        """Shell redirection means the log path is in neither argv nor
        environ; without reading fd/1 a restart silently loses serve.log."""
        from engines import ollama_engine

        spec = (["./bin/ollama", "serve"], {})
        opened = {}

        def fake_open(path, mode="r", *args, **kwargs):
            opened["path"] = path
            return mock.mock_open()()

        with mock.patch.object(ollama_engine, "daemon_launch_spec", return_value=spec), \
                mock.patch.object(
                    ollama_engine.os, "readlink",
                    return_value="/home/d3894/ollama/serve.log",
                ), \
                mock.patch.object(ollama_engine.os, "kill", side_effect=[None, OSError]), \
                mock.patch.object(ollama_engine.time, "sleep"), \
                mock.patch("builtins.open", side_effect=fake_open), \
                mock.patch.object(ollama_engine.subprocess, "Popen") as popen:
            popen.return_value = types.SimpleNamespace(pid=99999)
            ollama_engine.restart_daemon_pinned(23825, ["2", "3"])

        self.assertEqual(opened["path"], "/home/d3894/ollama/serve.log")

    def test_ignores_a_dev_null_log_destination(self):
        from engines import ollama_engine

        with mock.patch.object(ollama_engine.os, "readlink", return_value="/dev/null"):
            self.assertIsNone(ollama_engine.daemon_log_path(23825))

    def test_log_path_none_when_fd_unreadable(self):
        from engines import ollama_engine

        with mock.patch.object(ollama_engine.os, "readlink", side_effect=OSError):
            self.assertIsNone(ollama_engine.daemon_log_path(23825))


class OllamaVRAMFitWarningTests(unittest.TestCase):
    """The frontend can switch model at any time, but the daemon's GPU set
    only changes on restart - so a switch can outgrow its pin."""

    def _engine(self, model_id="qwen3:30b"):
        from engines.ollama_engine import OllamaEngine

        return OllamaEngine(_make_config("ollama", model_id=model_id))

    def test_warns_when_model_exceeds_visible_vram(self):
        engine = self._engine()
        with mock.patch.object(engine, "estimated_vram_mib", return_value=40000), \
                mock.patch.object(engine, "visible_vram_mib", return_value=32234):
            warning = engine.vram_warning_for("big:70b")

        self.assertIn("big:70b", warning)
        self.assertIn("gpu pin-free", warning)

    def test_silent_when_the_model_fits(self):
        engine = self._engine()
        with mock.patch.object(engine, "estimated_vram_mib", return_value=26545), \
                mock.patch.object(engine, "visible_vram_mib", return_value=32234):
            self.assertIsNone(engine.vram_warning_for("qwen3:30b"))

    def test_silent_when_either_side_is_unknown(self):
        engine = self._engine()
        with mock.patch.object(engine, "estimated_vram_mib", return_value=None), \
                mock.patch.object(engine, "visible_vram_mib", return_value=32234):
            self.assertIsNone(engine.vram_warning_for("x"))
        with mock.patch.object(engine, "estimated_vram_mib", return_value=1), \
                mock.patch.object(engine, "visible_vram_mib", return_value=None):
            self.assertIsNone(engine.vram_warning_for("x"))

    def test_visible_vram_sums_only_the_daemons_own_gpus(self):
        from engines import ollama_engine

        engine = self._engine()
        gpus = [
            GPUInfo(index=str(i), name="RTX A4000", memory_total_mib=16117,
                    memory_used_mib=3, memory_free_mib=16114, temperature_c=40,
                    utilization_percent=0, driver_version="470.86")
            for i in range(4)
        ]
        with mock.patch.object(ollama_engine, "find_daemon_pids", return_value=[23825]), \
                mock.patch.object(
                    engine.gpu_manager, "visible_gpu_indexes_for_process",
                    return_value=["0", "1"],
                ), \
                mock.patch.object(engine.gpu_manager, "detect_gpus", return_value=gpus):
            self.assertEqual(engine.visible_vram_mib(), 32234)

    def test_pulled_model_sizes_covers_every_tag(self):
        engine = self._engine()
        payload = {"models": [
            {"name": "qwen3:30b", "size": 18556699314},
            {"name": "small:1b", "size": 1073741824},
        ]}
        with mock.patch(
            "engines.ollama_engine.urllib.request.urlopen",
            side_effect=lambda *a, **k: _fake_urlopen_response(payload),
        ):
            sizes = engine.pulled_model_sizes()

        self.assertEqual(sorted(sizes), ["qwen3:30b", "small:1b"])
        self.assertGreater(sizes["qwen3:30b"], sizes["small:1b"])


class OllamaModelRuntimeInfoTests(unittest.TestCase):
    """Runtime facts backing the /v1/models picker: is each catalogued tag
    actually pulled, and does it fit the GPUs this daemon can reach."""

    def _engine(self):
        from engines.ollama_engine import OllamaEngine

        return OllamaEngine(_make_config("ollama", model_id="qwen3:30b"))

    def test_flags_pulled_size_and_fit(self):
        engine = self._engine()
        with mock.patch.object(
            engine, "pulled_model_sizes",
            return_value={"qwen3:30b": 26545, "qwen3:1.7b": 1944},
        ), mock.patch.object(engine, "visible_vram_mib", return_value=16117):
            info = engine.model_runtime_info(
                ["qwen3:30b", "qwen3:1.7b", "gemma3:4b"]
            )

        self.assertEqual(
            info["qwen3:30b"], {"pulled": True, "size_mib": 26545, "fits": False}
        )
        self.assertEqual(
            info["qwen3:1.7b"], {"pulled": True, "size_mib": 1944, "fits": True}
        )
        # Catalogued but not pulled: listed, with nothing invented.
        self.assertEqual(info["gemma3:4b"], {"pulled": False})

    def test_omits_fit_when_visible_vram_is_unknown(self):
        engine = self._engine()
        with mock.patch.object(
            engine, "pulled_model_sizes", return_value={"qwen3:30b": 26545}
        ), mock.patch.object(engine, "visible_vram_mib", return_value=None):
            info = engine.model_runtime_info(["qwen3:30b"])

        self.assertNotIn("fits", info["qwen3:30b"])
        self.assertEqual(info["qwen3:30b"]["size_mib"], 26545)


class ReasoningSplitTests(unittest.TestCase):
    """Three live-observed behaviours; one marker-based rule covers all,
    and is a no-op for models that do no reasoning."""

    def _split(self, content, thinking=None):
        from engines.ollama_engine import _split_reasoning

        return _split_reasoning(content, thinking)

    def test_uses_ollamas_own_thinking_field_when_present(self):
        """Dense qwen3 with think enabled: Ollama already separated it and
        this engine used to discard the field."""
        content, reasoning = self._split("Lisbon", "Let me recall...")

        self.assertEqual(content, "Lisbon")
        self.assertEqual(reasoning, "Let me recall...")

    def test_splits_reasoning_leaked_inline_before_the_closing_marker(self):
        """qwen3:30b MoE leaks it into content; the opening tag is consumed
        by the prompt, so only </think> is present."""
        raw = "Hmm, the user wants one word. Alright, Lisbon.\n</think>\n\nLisbon"

        content, reasoning = self._split(raw)

        self.assertEqual(content, "Lisbon")
        self.assertIn("Hmm, the user wants one word", reasoning)
        self.assertNotIn("</think>", content)
        self.assertNotIn("</think>", reasoning)

    def test_handles_a_matched_tag_pair(self):
        content, reasoning = self._split("<think>weighing it up</think>\n\nLisbon")

        self.assertEqual(content, "Lisbon")
        self.assertEqual(reasoning, "weighing it up")

    def test_is_a_no_op_for_models_that_do_not_reason(self):
        """llama3.2 / gemma3 / mistral emit no markers at all."""
        content, reasoning = self._split("Lisbon")

        self.assertEqual(content, "Lisbon")
        self.assertIsNone(reasoning)

    def test_splits_on_the_last_marker_when_prose_contains_one(self):
        raw = "I considered </think> as a token. Done.\n</think>\n\nAnswer"

        content, reasoning = self._split(raw)

        self.assertEqual(content, "Answer")
        self.assertTrue(reasoning.endswith("Done."))

    def test_empty_reasoning_reports_none_rather_than_a_blank_string(self):
        content, reasoning = self._split("</think>Lisbon")

        self.assertEqual(content, "Lisbon")
        self.assertIsNone(reasoning)

    def test_chat_returns_the_split_content_and_reasoning(self):
        from engines.ollama_engine import OllamaEngine

        engine = OllamaEngine(_make_config("ollama", model_id="qwen3:30b"))
        payload = {
            "message": {
                "role": "assistant",
                "content": "thinking out loud\n</think>\n\nLisbon",
            },
            "prompt_eval_count": 20,
            "eval_count": 146,
        }
        with mock.patch(
            "engines.ollama_engine.urllib.request.urlopen",
            side_effect=lambda *a, **k: _fake_urlopen_response(payload),
        ):
            result = engine.chat([_message("user", "hi")], 300, 0.7)

        self.assertEqual(result["content"], "Lisbon")
        self.assertEqual(result["reasoning"], "thinking out loud")

    def test_generate_text_splits_too(self):
        from engines.ollama_engine import OllamaEngine

        engine = OllamaEngine(_make_config("ollama", model_id="qwen3:30b"))
        payload = {"response": "pondering\n</think>\n\nLisbon"}
        with mock.patch(
            "engines.ollama_engine.urllib.request.urlopen",
            side_effect=lambda *a, **k: _fake_urlopen_response(payload),
        ):
            result = engine.generate_text("hi", 300, 0.7)

        self.assertEqual(result["response"], "Lisbon")
        self.assertEqual(result["reasoning"], "pondering")


def _ndjson_response(lines):
    body = b"".join((json.dumps(o) + "\n").encode("utf-8") for o in lines)

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    return _Response(body)


class OllamaStreamingTests(unittest.TestCase):
    """Whether a model reasons is decided from /api/show capabilities
    before any token is emitted - a stream cannot retract what it sent."""

    def _engine(self, model_id="qwen3:30b"):
        from engines.ollama_engine import OllamaEngine

        return OllamaEngine(_make_config("ollama", model_id=model_id))

    def _stream(self, engine, chunks, capabilities):
        with mock.patch.object(
            engine, "model_capabilities", return_value=capabilities
        ), mock.patch(
            "engines.ollama_engine.urllib.request.urlopen",
            side_effect=lambda *a, **k: _ndjson_response(chunks),
        ):
            return list(engine.chat_stream([_message("user", "hi")], 50, 0.7))

    def test_supports_streaming(self):
        self.assertTrue(self._engine().supports_streaming)

    def test_non_reasoning_model_streams_every_token_immediately(self):
        chunks = [
            {"message": {"content": "Lis"}, "done": False},
            {"message": {"content": "bon"}, "done": False},
            {"message": {"content": ""}, "done": True,
             "prompt_eval_count": 5, "eval_count": 2},
        ]
        deltas = self._stream(self._engine(), chunks, capabilities=["completion"])

        self.assertEqual(
            [d["content"] for d in deltas if "content" in d], ["Lis", "bon"]
        )
        self.assertEqual(deltas[-1]["usage"]["total_tokens"], 7)

    def test_separate_thinking_deltas_are_reasoning(self):
        chunks = [
            {"message": {"thinking": "weighing"}, "done": False},
            {"message": {"content": "Lisbon"}, "done": False},
            {"message": {"content": ""}, "done": True,
             "prompt_eval_count": 5, "eval_count": 2},
        ]
        deltas = self._stream(self._engine(), chunks, capabilities=["thinking"])

        self.assertEqual([d for d in deltas if "reasoning" in d][0]["reasoning"], "weighing")
        self.assertEqual([d for d in deltas if "content" in d][0]["content"], "Lisbon")

    def test_inline_leak_is_split_at_the_marker(self):
        """qwen3:30b: reasoning arrives as content tokens, ending </think>."""
        chunks = [
            {"message": {"content": "Hmm, "}, "done": False},
            {"message": {"content": "one word."}, "done": False},
            {"message": {"content": "\n</think>\n\nLis"}, "done": False},
            {"message": {"content": "bon"}, "done": False},
            {"message": {"content": ""}, "done": True,
             "prompt_eval_count": 5, "eval_count": 4},
        ]
        deltas = self._stream(self._engine(), chunks, capabilities=["thinking"])

        reasoning = "".join(d["reasoning"] for d in deltas if "reasoning" in d)
        content = "".join(d["content"] for d in deltas if "content" in d)
        self.assertEqual(reasoning, "Hmm, one word.")
        self.assertEqual(content, "Lisbon")
        # The answer after the marker still streams incrementally.
        self.assertEqual(
            [d["content"] for d in deltas if "content" in d], ["Lis", "bon"]
        )

    def test_reasoning_capable_model_that_does_not_reason_is_not_mislabelled(self):
        """think:false honoured: no marker ever arrives, so the buffered
        text is the answer - emitting it as reasoning would hide it."""
        chunks = [
            {"message": {"content": "Por"}, "done": False},
            {"message": {"content": "to"}, "done": False},
            {"message": {"content": ""}, "done": True,
             "prompt_eval_count": 5, "eval_count": 2},
        ]
        deltas = self._stream(self._engine(), chunks, capabilities=["thinking"])

        self.assertEqual(
            "".join(d["content"] for d in deltas if "content" in d), "Porto"
        )
        self.assertEqual([d for d in deltas if "reasoning" in d], [])

    def test_rejects_a_mismatched_requested_model_before_streaming(self):
        from engines.base import ModelNotFoundError

        engine = self._engine()
        with self.assertRaises(ModelNotFoundError):
            list(engine.chat_stream([_message("user", "hi")], 50, 0.7,
                                    requested_model="other:7b"))

    def test_model_capabilities_returns_empty_when_daemon_unreachable(self):
        engine = self._engine()
        with mock.patch(
            "engines.ollama_engine.urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            self.assertEqual(engine.model_capabilities(), [])

    def test_leading_newlines_after_the_marker_are_trimmed_across_chunks(self):
        """The blank line after </think> often arrives in a later chunk
        than the marker, so trimming only the marker's chunk leaks it."""
        chunks = [
            {"message": {"content": "thinking"}, "done": False},
            {"message": {"content": "\n</think>"}, "done": False},
            {"message": {"content": "\n\n"}, "done": False},
            {"message": {"content": "Lisbon"}, "done": False},
            {"message": {"content": ""}, "done": True,
             "prompt_eval_count": 5, "eval_count": 2},
        ]
        deltas = self._stream(self._engine(), chunks, capabilities=["thinking"])

        self.assertEqual(
            "".join(d["content"] for d in deltas if "content" in d), "Lisbon"
        )


class OllamaEngineEmbeddingTests(unittest.TestCase):
    """embed() must not go through the requested-model check chat() uses:
    an embedding model is a different model from the loaded chat model."""

    def _engine(self, model_id="qwen3:30b"):
        from engines.ollama_engine import OllamaEngine

        return OllamaEngine(_make_config("ollama", model_id=model_id))

    def test_uses_the_batch_endpoint_and_keeps_input_order(self):
        engine = self._engine()
        seen = {}

        def fake_post(path, payload):
            seen["path"], seen["payload"] = path, payload
            return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

        with mock.patch.object(engine, "_post", side_effect=fake_post):
            vectors = engine.embed(["first", "second"], "nomic-embed-text")

        self.assertEqual(seen["path"], "/api/embed")
        self.assertEqual(seen["payload"]["model"], "nomic-embed-text")
        self.assertEqual(seen["payload"]["input"], ["first", "second"])
        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])

    def test_embedding_model_is_not_checked_against_the_loaded_model(self):
        engine = self._engine(model_id="qwen3:30b")

        with mock.patch.object(engine, "_post", return_value={"embeddings": [[1.0]]}):
            engine.embed(["x"], "nomic-embed-text")  # must not raise ModelNotFoundError

        self.assertEqual(engine.model_id, "qwen3:30b", "must not switch the loaded model")

    def test_falls_back_to_the_legacy_single_embedding_endpoint(self):
        """Older daemons expose /api/embeddings ({"prompt"} -> {"embedding"})
        and 404 the batch route; the frontend hits the same split."""
        engine = self._engine()
        paths = []

        def fake_post(path, payload):
            paths.append(path)
            if path == "/api/embed":
                raise EngineUnavailableError("404 from an older daemon")
            return {"embedding": [0.7, 0.8]}

        with mock.patch.object(engine, "_post", side_effect=fake_post):
            vectors = engine.embed(["only"], "nomic-embed-text")

        self.assertEqual(paths, ["/api/embed", "/api/embeddings"])
        self.assertEqual(vectors, [[0.7, 0.8]])

    def test_an_unpulled_embedding_model_is_model_not_found_not_unavailable(self):
        """Ollama answers 404 "try pulling it first" from a perfectly healthy
        daemon. _request() collapses every HTTP error into
        EngineUnavailableError, which would surface as 503 and send the
        caller debugging their daemon instead of pulling the model."""
        engine = self._engine()

        def fake_post(path, payload):
            raise EngineUnavailableError("HTTP Error 404: Not Found")

        with mock.patch.object(engine, "_post", side_effect=fake_post):
            with mock.patch.object(engine, "_get_tags", return_value={"models": [{"name": "qwen3:30b"}]}):
                with self.assertRaises(ModelNotFoundError) as caught:
                    engine.embed(["x"], "nomic-embed-text")

        self.assertEqual(caught.exception.requested_model, "nomic-embed-text")

    def test_a_genuinely_down_daemon_still_reports_unavailable(self):
        """The counterpart: when the tag lookup fails too, the daemon really
        is down and 503 must stand rather than being relabelled a 404."""
        engine = self._engine()

        with mock.patch.object(engine, "_post", side_effect=EngineUnavailableError("refused")):
            with mock.patch.object(engine, "_get_tags", side_effect=EngineUnavailableError("refused")):
                with self.assertRaises(EngineUnavailableError):
                    engine.embed(["x"], "nomic-embed-text")

    def test_empty_input_makes_no_request(self):
        engine = self._engine()

        with mock.patch.object(engine, "_post") as post:
            self.assertEqual(engine.embed([], "nomic-embed-text"), [])

        post.assert_not_called()

    def test_a_short_batch_response_falls_back_rather_than_misaligning(self):
        """Fewer vectors than inputs would silently pair text with the wrong
        vector, so it must not be accepted."""
        engine = self._engine()

        def fake_post(path, payload):
            if path == "/api/embed":
                return {"embeddings": [[0.1]]}  # 1 vector for 2 inputs
            return {"embedding": [0.9]}

        with mock.patch.object(engine, "_post", side_effect=fake_post):
            vectors = engine.embed(["a", "b"], "nomic-embed-text")

        self.assertEqual(vectors, [[0.9], [0.9]])
