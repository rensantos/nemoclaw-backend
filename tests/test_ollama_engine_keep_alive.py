"""keep_alive handling in OllamaEngine's request payloads.

Until 2026-08-19 this engine never sent keep_alive to Ollama on any
request, so every call ran on Ollama's own compiled-in 5-minute default
regardless of what a caller's frontend intended - measured live,
nemoclaw-research-assistant's OLLAMA_KEEP_ALIVE=10m never reached Ollama
at all: its OpenAI-compatible client path drops the field before it
leaves the frontend process, and this engine never re-added one of its
own. On a node that alternates between a chat model and an embedding
model (claim-similarity scoring), an idle gap past that default is
exactly what evicts one to reload the other - a real reload was observed
within ~15 minutes of a model's last use on a real run.
"""

import types
import unittest
from unittest import mock

from config import BackendConfig, Config, ModelConfig
from engines.ollama_engine import OllamaEngine, _KEEP_ALIVE


def _message(role="user", content="hi"):
    return types.SimpleNamespace(role=role, content=content)


def _make_config(model_id="test-model"):
    return Config(
        backend=BackendConfig(
            host="127.0.0.1",
            port=8000,
            gpu="0",
            engine="ollama",
            ollama_host="http://127.0.0.1:11434",
            instance="test-instance",
        ),
        model=ModelConfig(
            id=model_id,
            max_tokens_default=16,
            temperature_default=0.1,
            quantization="none",
            revision="",
            think_default=None,
        ),
    )


class KeepAliveReachesEveryOllamaCallTests(unittest.TestCase):
    """_KEEP_ALIVE defaults to "30m" - well past a normal gap between
    /deepweb sources - and every payload this engine sends must carry it,
    not just some of them, since a single un-covered call site is enough
    for Ollama to fall back to its own 5-minute default on that path."""

    def _engine(self):
        return OllamaEngine(_make_config())

    def test_chat_sends_keep_alive(self):
        engine = self._engine()
        with mock.patch.object(engine, "_post", return_value={"message": {"content": "ok"}}) as fake_post:
            engine.chat([_message()], None, None)
        payload = fake_post.call_args[0][1]
        self.assertEqual(payload["keep_alive"], _KEEP_ALIVE)

    def test_chat_stream_sends_keep_alive(self):
        engine = self._engine()
        with mock.patch.object(engine, "_post_stream", return_value=iter([])) as fake_stream, \
             mock.patch.object(engine, "model_capabilities", return_value=[]):
            list(engine.chat_stream([_message()], None, None))
        payload = fake_stream.call_args[0][1]
        self.assertEqual(payload["keep_alive"], _KEEP_ALIVE)

    def test_generate_text_sends_keep_alive(self):
        engine = self._engine()
        with mock.patch.object(engine, "_post", return_value={"response": "ok"}) as fake_post:
            engine.generate_text("prompt", 16, 0.1)
        payload = fake_post.call_args[0][1]
        self.assertEqual(payload["keep_alive"], _KEEP_ALIVE)

    def test_embed_sends_keep_alive_on_the_batch_route(self):
        engine = self._engine()
        with mock.patch.object(engine, "_post", return_value={"embeddings": [[0.1]]}) as fake_post:
            engine.embed(["text"], "nomic-embed-text")
        payload = fake_post.call_args[0][1]
        self.assertEqual(payload["keep_alive"], _KEEP_ALIVE)

    def test_embed_sends_keep_alive_on_the_legacy_fallback_route(self):
        """The batch route 404s on an older daemon and this falls through
        to one request per text - that path must not lose the field."""
        from engines.base import EngineUnavailableError

        engine = self._engine()
        calls = []

        def fake_post(path, payload):
            calls.append((path, payload))
            if path == "/api/embed":
                raise EngineUnavailableError("404 from an older daemon")
            return {"embedding": [0.1]}

        with mock.patch.object(engine, "_post", side_effect=fake_post):
            engine.embed(["text"], "nomic-embed-text")

        legacy_calls = [payload for path, payload in calls if path == "/api/embeddings"]
        self.assertEqual(len(legacy_calls), 1)
        self.assertEqual(legacy_calls[0]["keep_alive"], _KEEP_ALIVE)


if __name__ == "__main__":
    unittest.main()
