"""Behavioural tests for the HTTP layer.

These issue real requests through FastAPI's TestClient and assert on
status codes and bodies. They replace the source-text greps that used to
stand in for API coverage - those passed whether or not routing actually
worked, and broke when a private helper was renamed.

They are possible at all because api.py now builds its InferenceService
lazily behind a dependency, so importing the module no longer loads a
model. Skips cleanly where fastapi/httpx are absent, as the deployment
host is the only machine guaranteed to have them.
"""

import json
import unittest
from unittest import mock

try:
    from fastapi.testclient import TestClient

    import api
    from app import app

    API_TESTABLE = True
except (ImportError, RuntimeError):  # pragma: no cover - host dependent
    API_TESTABLE = False

from services.lifecycle import (
    EmbeddingsNotSupportedError,
    LifecycleConflictError,
    LifecycleState,
    LifecycleUnavailableError,
    StreamingNotSupportedError,
)

SKIP_REASON = "fastapi/httpx not installed on this machine"


class FakeService:
    """Stands in for InferenceService at the HTTP boundary."""

    def __init__(self):
        self.lifecycle_state = LifecycleState.READY
        self.chat_result = {
            "content": "Lisbon",
            "reasoning": None,
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        }
        self.chat_error = None
        self.stream_error = None
        self.deltas = [
            {"content": "Lis"},
            {"content": "bon"},
            {"usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
        ]
        self.lifecycle_error = None
        self.embed_result = [[0.1, 0.2, 0.3]]
        self.embed_error = None
        self.calls = []

    def health(self):
        return {
            "model": "qwen3:30b", "cuda": True, "gpu": None, "status": "ok",
            "lifecycle_state": self.lifecycle_state.value,
            "loaded_model": "qwen3:30b", "target_model": None,
            "instance": "test-host",
        }

    def list_models(self):
        return {"object": "list", "data": [{"id": "qwen3:30b", "object": "model",
                                            "created": 0, "owned_by": "ollama"}]}

    def chat(self, messages, max_tokens, temperature, requested_model=None, think=None,
             num_ctx=None):
        self.calls.append(("chat", requested_model, think, num_ctx))
        if self.chat_error:
            raise self.chat_error
        return self.chat_result

    def chat_stream(self, messages, max_tokens, temperature, requested_model=None,
                    think=None, num_ctx=None):
        self.calls.append(("chat_stream", requested_model, think, num_ctx))
        if self.stream_error:
            raise self.stream_error
        return iter(self.deltas)

    def generate_text(self, prompt, max_new_tokens, temperature, think=None):
        return {"model": "qwen3:30b", "response": prompt, "reasoning": None}

    def embed(self, texts, model):
        self.calls.append(("embed", model, tuple(texts)))
        if self.embed_error:
            raise self.embed_error
        return self.embed_result

    def _lifecycle(self, name, model_id=None):
        self.calls.append((name, model_id))
        if self.lifecycle_error:
            raise self.lifecycle_error
        return {"status": "ok", "lifecycle_state": "ready",
                "loaded_model": model_id, "previous_model": "qwen3:30b",
                "elapsed_seconds": 0.01, "persisted": False}

    def load_model(self, model_id, persist=False):
        return self._lifecycle("load", model_id)

    def unload_model(self):
        return self._lifecycle("unload")

    def switch_model(self, model_id, persist=False):
        return self._lifecycle("switch", model_id)

    def pull_status(self):
        self.calls.append(("pull_status", None))
        return {"active": True, "model_id": "deepseek-r1:14b", "status": "pulling",
                "completed_bytes": 450, "total_bytes": 900, "percent": 50.0,
                "layers": 1, "started_at": 1.0, "updated_at": 2.0,
                "finished_at": None, "error": None}


@unittest.skipUnless(API_TESTABLE, SKIP_REASON)
class ApiRequestTests(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        api.set_inference_service(self.service)
        self.client = TestClient(app)

    def tearDown(self):
        api.set_inference_service(None)

    # --- the seam itself -------------------------------------------------

    def test_importing_api_does_not_build_a_service(self):
        """The whole point of the dependency: import must be side-effect
        free, or the API layer cannot be tested without a GPU."""
        api.set_inference_service(None)
        with mock.patch.object(api, "create_inference_service") as factory:
            self.assertFalse(factory.called)

    # --- health / models -------------------------------------------------

    def test_health_returns_lifecycle_fields(self):
        body = self.client.get("/health").json()

        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["lifecycle_state"], "ready")
        self.assertEqual(body["loaded_model"], "qwen3:30b")

    def test_models_lists_entries(self):
        response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "list")

    # --- chat completions ------------------------------------------------

    def test_chat_completion_shape(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "Lisbon")
        self.assertEqual(body["usage"]["total_tokens"], 4)

    def test_num_ctx_reaches_the_service(self):
        """The whole point of the field. Until 2026-08-12 there was no way
        for a caller to set a context window at all - `options` is not part
        of the OpenAI chat-completions schema - so every request ran at
        Ollama's 4096 default and long prompts were silently truncated."""
        self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "num_ctx": 65536},
        )

        chat_calls = [c for c in self.service.calls if c[0] == "chat"]
        self.assertEqual(chat_calls[-1][3], 65536)

    def test_num_ctx_is_optional_and_defaults_to_not_sending_one(self):
        """Callers that do not ask must be unaffected, so Ollama keeps its
        own default rather than this becoming a new implicit policy."""
        self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

        chat_calls = [c for c in self.service.calls if c[0] == "chat"]
        self.assertIsNone(chat_calls[-1][3])

    def test_num_ctx_reaches_the_service_when_streaming_too(self):
        """The streaming path is a separate call site and was the one more
        likely to be forgotten."""
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True, "num_ctx": 32768},
        ) as response:
            list(response.iter_lines())

        stream_calls = [c for c in self.service.calls if c[0] == "chat_stream"]
        self.assertEqual(stream_calls[-1][3], 32768)

    def test_reasoning_is_omitted_when_absent_and_present_when_not(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertNotIn("reasoning", response.json()["choices"][0]["message"])

        self.service.chat_result = dict(self.service.chat_result, reasoning="thinking")
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(
            response.json()["choices"][0]["message"]["reasoning"], "thinking"
        )

    def test_empty_messages_is_rejected(self):
        response = self.client.post("/v1/chat/completions", json={"messages": []})

        self.assertEqual(response.status_code, 400)

    def test_unknown_model_returns_the_openai_error_shape(self):
        from engines.base import ModelNotFoundError

        self.service.chat_error = ModelNotFoundError("other:7b", "qwen3:30b")

        response = self.client.post(
            "/v1/chat/completions",
            json={"model": "other:7b", "messages": [{"role": "user", "content": "hi"}]},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "model_not_found")

    def test_unreachable_engine_returns_503(self):
        from engines.base import EngineUnavailableError

        self.service.chat_error = EngineUnavailableError("daemon down")

        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

        self.assertEqual(response.status_code, 503)

    def test_mid_transition_request_returns_503(self):
        self.service.chat_error = LifecycleUnavailableError(LifecycleState.SWITCHING)

        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

        self.assertEqual(response.status_code, 503)

    # --- streaming -------------------------------------------------------

    def _sse_events(self, response):
        events = []
        for line in response.text.splitlines():
            if line.startswith("data: "):
                body = line[len("data: "):]
                events.append(body if body == "[DONE]" else json.loads(body))
        return events

    def test_streaming_returns_sse_chunks_ending_with_done(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))

        events = self._sse_events(response)
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(events[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(events[0]["object"], "chat.completion.chunk")

        content = "".join(
            e["choices"][0]["delta"].get("content", "")
            for e in events[1:-1]
        )
        self.assertEqual(content, "Lisbon")

        final = events[-2]
        self.assertEqual(final["choices"][0]["finish_reason"], "stop")
        self.assertEqual(final["usage"]["total_tokens"], 5)

    def test_streaming_emits_reasoning_deltas_separately(self):
        self.service.deltas = [
            {"reasoning": "weighing"},
            {"content": "Lisbon"},
            {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        ]

        events = self._sse_events(self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        ))
        deltas = [e["choices"][0]["delta"] for e in events[:-1]]

        self.assertIn({"reasoning": "weighing"}, deltas)
        self.assertIn({"content": "Lisbon"}, deltas)

    def test_streaming_against_a_non_streaming_engine_returns_400(self):
        """Must be a real status code, which is only possible because the
        service rejects eagerly rather than from inside the generator."""
        self.service.stream_error = StreamingNotSupportedError("TransformersEngine")

        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("TransformersEngine", response.json()["detail"])

    def test_streaming_with_an_unservable_model_returns_404(self):
        """The streaming path must return the same model_not_found shape as
        the non-streaming one. It previously returned 200 and then died
        mid-stream, because the engine's guard ran on first iteration."""
        from engines.base import ModelNotFoundError

        self.service.stream_error = ModelNotFoundError("qwen3:32b", "qwen3:30b")

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "qwen3:32b",
                "stream": True,
            },
        )

        self.assertEqual(response.status_code, 404)
        error = response.json()["error"]
        self.assertEqual(error["code"], "model_not_found")
        self.assertIn("qwen3:32b", error["message"])

    def test_streaming_while_not_ready_returns_503(self):
        self.service.stream_error = LifecycleUnavailableError(LifecycleState.UNLOADED)

        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

        self.assertEqual(response.status_code, 503)

    # --- admin lifecycle -------------------------------------------------

    def test_admin_switch_returns_the_result_body(self):
        response = self.client.post(
            "/admin/model/switch", json={"model_id": "qwen3:1.7b"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["loaded_model"], "qwen3:1.7b")
        self.assertIn(("switch", "qwen3:1.7b"), self.service.calls)

    def test_admin_unload_needs_no_body(self):
        self.assertEqual(self.client.post("/admin/model/unload").status_code, 200)

    def test_admin_load_without_a_body_is_a_422(self):
        self.assertEqual(self.client.post("/admin/model/load").status_code, 422)

    def test_unconfigured_model_is_404(self):
        self.service.lifecycle_error = ValueError("Model is not configured: nope")

        response = self.client.post("/admin/model/switch", json={"model_id": "nope"})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"], "model_not_configured")

    def test_lifecycle_conflict_is_409(self):
        self.service.lifecycle_error = LifecycleConflictError("already loaded")

        response = self.client.post("/admin/model/load", json={"model_id": "x"})

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"], "lifecycle_conflict")

    def test_unpulled_model_is_409_model_unavailable(self):
        from engines.base import ModelUnavailableError

        self.service.lifecycle_error = ModelUnavailableError("tag not present")

        response = self.client.post("/admin/model/switch", json={"model_id": "x"})

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"], "model_unavailable")

    def test_pull_status_is_readable_as_a_plain_get(self):
        """The POST that starts a download does not answer until it ends,
        so this is the only way a caller can see one running."""
        response = self.client.get("/admin/model/pull/status")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["active"])
        self.assertEqual(body["model_id"], "deepseek-r1:14b")
        self.assertEqual(body["percent"], 50.0)

    def test_engine_without_lifecycle_support_is_501(self):
        from engines.base import LifecycleNotSupportedError

        self.service.lifecycle_error = LifecycleNotSupportedError("TransformersEngine")

        response = self.client.post("/admin/model/switch", json={"model_id": "x"})

        self.assertEqual(response.status_code, 501)
        self.assertEqual(response.json()["error"], "lifecycle_not_supported")

    def test_error_bodies_carry_the_lifecycle_state(self):
        self.service.lifecycle_error = LifecycleConflictError("nope")
        self.service.lifecycle_state = LifecycleState.DEGRADED

        response = self.client.post("/admin/model/load", json={"model_id": "x"})

        self.assertEqual(response.json()["lifecycle_state"], "degraded")

    def test_health_names_which_instance_answered(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertIn("instance", response.json())

    def test_embeddings_returns_one_vector_per_input_in_order(self):
        self.service.embed_result = [[0.1, 0.2], [0.3, 0.4]]

        response = self.client.post(
            "/v1/embeddings", json={"model": "nomic-embed-text", "input": ["a", "b"]}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["model"], "nomic-embed-text")
        self.assertEqual([row["index"] for row in body["data"]], [0, 1])
        self.assertEqual([row["embedding"] for row in body["data"]], [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(self.service.calls[-1], ("embed", "nomic-embed-text", ("a", "b")))

    def test_embeddings_accepts_a_bare_string_input(self):
        response = self.client.post(
            "/v1/embeddings", json={"model": "nomic-embed-text", "input": "hello"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.calls[-1], ("embed", "nomic-embed-text", ("hello",)))

    def test_embeddings_does_not_default_to_the_loaded_chat_model(self):
        """model is required: embedding text with the chat model would return
        vectors that silently fail to match a real vectorstore."""
        response = self.client.post("/v1/embeddings", json={"input": "hello"})

        self.assertEqual(response.status_code, 422)

    def test_embeddings_rejects_empty_input(self):
        for payload in ({"model": "m", "input": ""}, {"model": "m", "input": []},
                        {"model": "m", "input": ["ok", "   "]}):
            with self.subTest(payload=payload):
                response = self.client.post("/v1/embeddings", json=payload)
                self.assertEqual(response.status_code, 400)

    def test_embeddings_on_an_engine_that_cannot_embed_is_501(self):
        self.service.embed_error = EmbeddingsNotSupportedError("TransformersEngine")

        response = self.client.post(
            "/v1/embeddings", json={"model": "nomic-embed-text", "input": "hi"}
        )

        self.assertEqual(response.status_code, 501)
        self.assertIn("TransformersEngine", response.json()["detail"])

    def test_embeddings_when_the_engine_is_unreachable_is_503(self):
        from engines.base import EngineUnavailableError

        self.service.embed_error = EngineUnavailableError("daemon down")

        response = self.client.post(
            "/v1/embeddings", json={"model": "nomic-embed-text", "input": "hi"}
        )

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
