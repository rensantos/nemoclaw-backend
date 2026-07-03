import unittest
from pathlib import Path

from services.inference import InferenceService


class FakeEngine:
    def __init__(self):
        self.loaded = False
        self.calls = []

    def load_model(self):
        self.loaded = True
        self.calls.append("load_model")

    def unload_model(self):
        self.loaded = False
        self.calls.append("unload_model")

    def health(self):
        self.calls.append("health")
        return {"status": "ok"}

    def list_models(self):
        self.calls.append("list_models")
        return {"object": "list", "data": []}

    def chat(self, messages, max_tokens, temperature):
        self.calls.append(("chat", messages, max_tokens, temperature))
        return {
            "content": "hello",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }

    def generate_text(self, prompt, max_new_tokens, temperature):
        self.calls.append(("generate_text", prompt, max_new_tokens, temperature))
        return {"model": "fake", "response": prompt}


class InferenceServiceTests(unittest.TestCase):
    def test_service_loads_engine_once_on_init(self):
        engine = FakeEngine()

        InferenceService(engine)

        self.assertTrue(engine.loaded)
        self.assertEqual(engine.calls, ["load_model"])

    def test_service_delegates_health_and_models(self):
        engine = FakeEngine()
        service = InferenceService(engine)

        self.assertEqual(service.health(), {"status": "ok"})
        self.assertEqual(service.list_models(), {"object": "list", "data": []})
        self.assertIn("health", engine.calls)
        self.assertIn("list_models", engine.calls)

    def test_service_delegates_chat(self):
        engine = FakeEngine()
        service = InferenceService(engine)

        result = service.chat(["message"], 32, 0.5)

        self.assertEqual(result["content"], "hello")
        self.assertIn(("chat", ["message"], 32, 0.5), engine.calls)

    def test_service_delegates_generate_text(self):
        engine = FakeEngine()
        service = InferenceService(engine)

        result = service.generate_text("prompt", 12, 0.7)

        self.assertEqual(result["response"], "prompt")
        self.assertIn(("generate_text", "prompt", 12, 0.7), engine.calls)


class ApiBoundaryTests(unittest.TestCase):
    def test_api_layer_does_not_import_transformers_or_torch(self):
        api_source = Path("api.py").read_text(encoding="utf-8")

        self.assertNotIn("transformers", api_source)
        self.assertNotIn("import torch", api_source)


if __name__ == "__main__":
    unittest.main()
