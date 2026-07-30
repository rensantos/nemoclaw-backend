import io
import json
import sys
import types
import unittest
import urllib.error
from unittest import mock

from config import BackendConfig, Config, ModelConfig
from engines.base import EngineUnavailableError, ModelNotFoundError
from services.inference import InferenceService, _build_engine
from services.lifecycle import LifecycleState


def _make_config(engine, ollama_host="http://127.0.0.1:11434", model_id="test-model"):
    return Config(
        backend=BackendConfig(
            host="127.0.0.1",
            port=8000,
            gpu="0",
            engine=engine,
            ollama_host=ollama_host,
        ),
        model=ModelConfig(id=model_id, max_tokens_default=16, temperature_default=0.1),
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
        return mock.patch(
            "engines.ollama_engine.urllib.request.urlopen",
            return_value=_fake_urlopen_response(payload),
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


def _message(role, content):
    return types.SimpleNamespace(role=role, content=content)


class OllamaEngineChatTests(unittest.TestCase):
    """docs/ollama-engine-design.md Increment 3: chat()/generate_text()
    against a (mocked) live Ollama daemon."""

    def _engine(self, model_id="qwen3:1.7b"):
        from engines.ollama_engine import OllamaEngine

        return OllamaEngine(_make_config("ollama", model_id=model_id))

    def _mock_urlopen(self, payload=None, raise_url_error=False):
        if raise_url_error:
            return mock.patch(
                "engines.ollama_engine.urllib.request.urlopen",
                side_effect=urllib.error.URLError("connection refused"),
            )
        return mock.patch(
            "engines.ollama_engine.urllib.request.urlopen",
            return_value=_fake_urlopen_response(payload),
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

        self.assertEqual(result, {"model": "qwen3:1.7b", "response": "generated text"})

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
        return mock.patch(
            "engines.ollama_engine.urllib.request.urlopen",
            return_value=_fake_urlopen_response(payload),
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
