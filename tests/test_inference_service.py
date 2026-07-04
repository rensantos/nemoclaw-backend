import unittest
from pathlib import Path

from services.inference import InferenceService
from services.lifecycle import LifecycleState


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

        health = service.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(service.list_models(), {"object": "list", "data": []})
        self.assertIn("health", engine.calls)
        self.assertIn("list_models", engine.calls)

    def test_default_lifecycle_state_is_ready(self):
        service = InferenceService(FakeEngine())

        self.assertEqual(service.lifecycle_state, LifecycleState.READY)

    def test_health_includes_lifecycle_state(self):
        service = InferenceService(FakeEngine())

        self.assertEqual(service.health()["lifecycle_state"], "ready")

    def test_health_preserves_existing_engine_fields(self):
        class RichFakeEngine(FakeEngine):
            def health(self):
                self.calls.append("health")
                return {
                    "status": "ok",
                    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "cuda": True,
                    "gpu": "RTX A4000",
                }

        service = InferenceService(RichFakeEngine())

        health = service.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["model"], "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.assertTrue(health["cuda"])
        self.assertEqual(health["gpu"], "RTX A4000")
        self.assertEqual(health["lifecycle_state"], "ready")

    def test_lifecycle_stub_response_shape(self):
        service = InferenceService(FakeEngine())

        response = service.lifecycle_stub_response()

        self.assertEqual(response, {
            "error": "not_implemented",
            "detail": "Model lifecycle operations are not implemented yet.",
            "lifecycle_state": "ready",
        })

    def test_lifecycle_stub_response_does_not_change_lifecycle_state(self):
        service = InferenceService(FakeEngine())

        service.lifecycle_stub_response()
        service.lifecycle_stub_response()

        self.assertEqual(service.lifecycle_state, LifecycleState.READY)
        self.assertEqual(service.health()["lifecycle_state"], "ready")

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

    def test_v1_endpoints_are_unchanged(self):
        api_source = Path("api.py").read_text(encoding="utf-8")

        self.assertIn('@router.get("/health")', api_source)
        self.assertIn('@router.get("/v1/models")', api_source)
        self.assertIn('@router.post("/v1/chat/completions")', api_source)

    def test_admin_lifecycle_endpoints_are_declared_as_501_stubs(self):
        api_source = Path("api.py").read_text(encoding="utf-8")

        for path in (
            "/admin/model/load",
            "/admin/model/unload",
            "/admin/model/switch",
        ):
            self.assertIn(
                '@router.post("{}", status_code=501)'.format(path), api_source
            )

        self.assertEqual(api_source.count("lifecycle_stub_response()"), 3)
        self.assertEqual(api_source.count("JSONResponse(status_code=501"), 3)


if __name__ == "__main__":
    unittest.main()
