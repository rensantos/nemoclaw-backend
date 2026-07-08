"""Skeleton OllamaEngine (docs/ollama-engine-design.md Increment 1).

This increment only makes the "ollama" engine selectable and constructible
through the engine factory in services/inference.py, so that path is real
and testable end to end. No Ollama HTTP calls, daemon logic, or new
dependencies exist yet; those arrive with the later increments in
docs/ollama-engine-design.md Section 6.
"""

from typing import List, Optional

from engines.base import InferenceEngine


_NOT_IMPLEMENTED = (
    "OllamaEngine.{method}() is not implemented yet (Increment 1 only wires "
    "engine selection). See docs/ollama-engine-design.md Section 6 for the "
    "increment that implements it."
)


class OllamaEngine(InferenceEngine):
    """Placeholder InferenceEngine for Ollama; every real method is
    unimplemented until later increments land."""

    def __init__(self, config):
        self.config = config

    def load_model(self) -> None:
        """No-op in Increment 1.

        InferenceService.__init__ calls load_model() eagerly at
        construction time. Real tag-presence validation against the Ollama
        daemon (docs/ollama-engine-design.md Section 4) arrives in
        Increment 2; a no-op here lets `engine: ollama` start up cleanly
        instead of crashing the process before any request is served.
        """

    def unload_model(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED.format(method="unload_model"))

    def health(self):
        raise NotImplementedError(_NOT_IMPLEMENTED.format(method="health"))

    def list_models(self):
        raise NotImplementedError(_NOT_IMPLEMENTED.format(method="list_models"))

    def chat(self, messages: List, max_tokens: Optional[int], temperature: Optional[float]):
        raise NotImplementedError(_NOT_IMPLEMENTED.format(method="chat"))

    def generate_text(self, prompt: str, max_new_tokens: int, temperature: float):
        raise NotImplementedError(_NOT_IMPLEMENTED.format(method="generate_text"))
