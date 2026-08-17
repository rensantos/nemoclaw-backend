"""servable_context_length: what THIS machine can hold, not what the model supports.

The two are different numbers and conflating them degraded nodes
repeatedly. qwen3:30b declares a 262,144 context and needs 41.1 GiB of KV
cache to use it, on a box with ~32 GiB of RAM; the load fails, Ollama
starts returning invalid JSON, and the backend goes `degraded` - a sticky
state needing a manual restart. That happened on 2026-08-12, again on
2026-08-15, and a third time locally on 2026-08-17 with a different model.

The calculation belongs in the backend because the backend is the
component standing on the hardware. Every machine runs its own backend
beside its frontend, so a local answer is available either way - but the
UBI server runs a backend with NO frontend, and only this side can answer
for it. One implementation, every node.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.ollama_engine import OllamaEngine  # noqa: E402

GB = 1024 ** 3

# qwen3:30b-a3b, read from Ollama's /api/show on 2026-08-17.
QWEN3_MOE_INFO = {
    "qwen3moe.block_count": 48,
    "qwen3moe.attention.head_count_kv": 4,
    "qwen3moe.attention.key_length": 128,
    "qwen3moe.attention.value_length": 128,
    "qwen3moe.context_length": 262144,
}


def engine_with(info, tags_size, resident_size, available_bytes):
    """An engine whose daemon and host are stubbed, so the arithmetic is
    exercised without a live Ollama or a particular machine."""
    engine = object.__new__(OllamaEngine)
    engine.model_id = "qwen3:30b-a3b"
    engine.base_url = "http://127.0.0.1:11434"
    engine._post = lambda path, payload: {"model_info": info}
    engine._get_tags = lambda: {"models": [{"name": "qwen3:30b-a3b", "size": tags_size}]}
    engine._get_running = lambda: (
        {"models": [{"name": "qwen3:30b-a3b", "size": resident_size}]} if resident_size else {"models": []}
    )
    return engine


class ServableContextTests(unittest.TestCase):
    def _servable(self, available, tags_size=int(18.6 * GB), resident=0, info=None):
        engine = engine_with(info or QWEN3_MOE_INFO, tags_size, resident, available)
        with mock.patch("engines.ollama_engine._available_memory_bytes", return_value=available):
            return engine.servable_context_length()

    def test_it_refuses_the_window_that_degraded_the_backend(self):
        """131,072 on ~30 GB free is the exact run that broke this machine
        on 2026-08-17: 12.9 GB of cache on top of an 18.6 GB model."""
        self.assertLess(self._servable(int(30 * GB)), 131072)

    def test_it_still_allows_the_window_that_worked(self):
        """96,156 tokens ran in 407s on the same machine. A calculation so
        conservative that it forbids that would throttle the node instead
        of protecting it - the other half of the failure."""
        self.assertGreater(self._servable(int(30 * GB)), 90000)

    def test_resident_weights_are_reclaimable(self):
        """A loaded model is reloaded at the new context, not loaded a
        second time, so its memory is available. Counting it as gone
        understates the window badly - measured on the frontend
        prototype, it turned a real ~90k into 16k."""
        without = self._servable(int(12 * GB), resident=0)
        with_resident = self._servable(int(12 * GB), resident=int(18.6 * GB))
        self.assertGreater(with_resident, without)

    def test_a_smaller_model_gets_a_bigger_window(self):
        """The property no configured number can express: /model changes
        the model from a chat message, and the safe window moves with it."""
        big_model = self._servable(int(30 * GB), tags_size=int(18.6 * GB))
        small_model = self._servable(int(30 * GB), tags_size=int(7.6 * GB))
        self.assertGreater(small_model, big_model)

    def test_less_free_memory_gives_a_smaller_window(self):
        """The other half: somebody else's job on a shared machine."""
        self.assertGreater(self._servable(int(30 * GB)), self._servable(int(20 * GB)))

    def test_headroom_is_actually_reserved(self):
        """Activations, the runtime and everything else on the machine
        live outside the model+cache arithmetic. Without the reserve a
        window fits on paper and degrades the backend in practice - which
        is how the number was learned."""
        import engines.ollama_engine as oe

        with mock.patch.object(oe, "_CONTEXT_MEMORY_HEADROOM_BYTES", 0):
            without = self._servable(int(30 * GB))
        with_reserve = self._servable(int(30 * GB))
        self.assertLess(with_reserve, without)

    def test_it_never_exceeds_the_model_maximum(self):
        info = dict(QWEN3_MOE_INFO, **{"qwen3moe.context_length": 8192})
        self.assertLessEqual(self._servable(int(200 * GB), info=info), 8192)

    def test_unknown_architecture_reports_zero_rather_than_guessing(self):
        """0 means "could not tell" and makes the caller fall back.
        Inventing a number here crashes a node."""
        self.assertEqual(self._servable(int(30 * GB), info={"something.else": 1}), 0)

    def test_no_free_memory_reports_zero_not_a_negative_window(self):
        self.assertEqual(self._servable(1024), 0)

    def test_an_unreachable_daemon_reports_zero(self):
        engine = engine_with(QWEN3_MOE_INFO, int(18.6 * GB), 0, int(30 * GB))

        def boom(*_args, **_kwargs):
            raise RuntimeError("daemon down")

        engine._post = boom
        with mock.patch("engines.ollama_engine._available_memory_bytes", return_value=int(30 * GB)):
            # The engine catches only EngineUnavailableError; anything else
            # is a real bug and must not be silently reported as 0.
            with self.assertRaises(RuntimeError):
                engine.servable_context_length()

    def test_the_base_engine_reports_zero(self):
        """An engine with no way to measure its host answers truthfully
        rather than guessing, and callers already handle 0."""
        from engines.base import InferenceEngine

        self.assertEqual(InferenceEngine.servable_context_length(object()), 0)


if __name__ == "__main__":
    unittest.main()
