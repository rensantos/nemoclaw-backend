"""num_ctx handling in OllamaEngine's request payload.

Until 2026-08-12 this backend never sent num_ctx to Ollama, so every
request it served ran at Ollama's default context of 4096 tokens whatever
the caller assembled. Measured against a ~40k-token prompt: this backend
reported prompt_tokens=4096, while the same prompt sent straight to Ollama
with options.num_ctx processed 26,046. A caller could not work around it -
`options` is not part of the OpenAI chat-completions schema.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.ollama_engine import OllamaEngine  # noqa: E402


class ApplyNumCtxTests(unittest.TestCase):
    """_apply_num_ctx does not touch self, so it is exercised directly
    rather than standing up an engine with a config and a live daemon."""

    def _payload(self, num_ctx):
        payload = {"options": {"temperature": 0.2, "num_predict": 128}}
        OllamaEngine._apply_num_ctx(None, payload, num_ctx)
        return payload

    def test_it_goes_inside_options_where_ollama_reads_it(self):
        """Unlike `think`, which is top-level. Putting it at the top level
        would be silently ignored - the exact failure mode being fixed."""
        payload = self._payload(65536)
        self.assertEqual(payload["options"]["num_ctx"], 65536)
        self.assertNotIn("num_ctx", {k: v for k, v in payload.items() if k != "options"})

    def test_existing_options_are_preserved(self):
        payload = self._payload(8192)
        self.assertEqual(payload["options"]["temperature"], 0.2)
        self.assertEqual(payload["options"]["num_predict"], 128)

    def test_none_omits_the_key_entirely(self):
        """So a caller that does not ask keeps Ollama's own default and
        nothing changes for them."""
        self.assertNotIn("num_ctx", self._payload(None)["options"])

    def test_non_positive_values_are_ignored_rather_than_forwarded(self):
        """0 or negative is a caller bug; Ollama's behaviour for them is
        not something to inherit silently."""
        for bad in (0, -1):
            self.assertNotIn("num_ctx", self._payload(bad)["options"], bad)

    def test_it_creates_options_if_absent(self):
        payload = {}
        OllamaEngine._apply_num_ctx(None, payload, 4096)
        self.assertEqual(payload["options"]["num_ctx"], 4096)


if __name__ == "__main__":
    unittest.main()
