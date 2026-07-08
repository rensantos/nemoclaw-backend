import sys
import types
import unittest
from unittest import mock

from config import BackendConfig, Config, ModelConfig
from services.inference import InferenceService, _build_engine


def _make_config(engine):
    return Config(
        backend=BackendConfig(host="127.0.0.1", port=8000, gpu="0", engine=engine),
        model=ModelConfig(id="test-model", max_tokens_default=16, temperature_default=0.1),
    )


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


class OllamaEngineStubTests(unittest.TestCase):
    def test_constructs_without_error(self):
        from engines.ollama_engine import OllamaEngine

        engine = OllamaEngine(_make_config("ollama"))

        self.assertIsNotNone(engine)

    def test_load_model_is_a_safe_no_op(self):
        from engines.ollama_engine import OllamaEngine

        engine = OllamaEngine(_make_config("ollama"))

        engine.load_model()  # must not raise

    def test_other_methods_raise_not_implemented(self):
        from engines.ollama_engine import OllamaEngine

        engine = OllamaEngine(_make_config("ollama"))

        with self.assertRaises(NotImplementedError):
            engine.unload_model()
        with self.assertRaises(NotImplementedError):
            engine.health()
        with self.assertRaises(NotImplementedError):
            engine.list_models()
        with self.assertRaises(NotImplementedError):
            engine.chat([], None, None)
        with self.assertRaises(NotImplementedError):
            engine.generate_text("prompt", 8, 0.5)

    def test_inference_service_constructs_with_ollama_engine(self):
        # InferenceService.__init__ eagerly calls load_model(); this must
        # not raise for the Increment 1 stub, so `engine: ollama` can start
        # up cleanly instead of crashing the process at construction time.
        from engines.ollama_engine import OllamaEngine

        service = InferenceService(OllamaEngine(_make_config("ollama")))

        self.assertIsNotNone(service)

    def test_inference_service_health_fails_cleanly_not_a_hang(self):
        from engines.ollama_engine import OllamaEngine

        service = InferenceService(OllamaEngine(_make_config("ollama")))

        with self.assertRaises(NotImplementedError):
            service.health()


if __name__ == "__main__":
    unittest.main()
