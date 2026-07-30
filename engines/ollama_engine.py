"""OllamaEngine (docs/ollama-engine-design.md Increment 2: read paths).

Implements load_model()/health()/list_models() against a live Ollama
daemon's GET /api/tags. No pulling: load_model() only confirms the
configured tag is already present locally (docs/ollama-engine-design.md
Section 5, Non-goals). chat()/generate_text()/unload_model() remain
unimplemented until Increments 3-4.
"""

import json
import urllib.error
import urllib.request
from typing import List, Optional

from engines.base import EngineUnavailableError, InferenceEngine
from services.gpu import GPUManager


_NOT_IMPLEMENTED = (
    "OllamaEngine.{method}() is not implemented yet. See "
    "docs/ollama-engine-design.md Section 6 for the increment that "
    "implements it."
)

_TAGS_TIMEOUT_SECONDS = 5


class OllamaEngine(InferenceEngine):
    """InferenceEngine backed by a live Ollama daemon."""

    def __init__(self, config):
        self.config = config
        self.model_id = config.model.id
        self.base_url = config.backend.ollama_host.rstrip("/")
        self.gpu_manager = GPUManager(config)

    def load_model(self) -> None:
        """Validates the configured tag is present locally. Never pulls."""
        tags = self._tag_names(self._get_tags())
        if self.model_id not in tags:
            raise RuntimeError(
                "Ollama tag '{}' is not present on the daemon at {}. Run "
                "'ollama pull {}' on the machine hosting Ollama, then "
                "restart the backend.".format(
                    self.model_id, self.base_url, self.model_id
                )
            )

    def unload_model(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED.format(method="unload_model"))

    def health(self):
        cuda, gpu = self._gpu_snapshot()
        try:
            self._get_tags()
        except EngineUnavailableError as exc:
            exc.partial_health = {"model": self.model_id, "cuda": cuda, "gpu": gpu}
            raise
        return {"model": self.model_id, "cuda": cuda, "gpu": gpu}

    def list_models(self):
        import time

        self._get_tags()
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama",
                }
            ],
        }

    def chat(self, messages: List, max_tokens: Optional[int], temperature: Optional[float]):
        raise NotImplementedError(_NOT_IMPLEMENTED.format(method="chat"))

    def generate_text(self, prompt: str, max_new_tokens: int, temperature: float):
        raise NotImplementedError(_NOT_IMPLEMENTED.format(method="generate_text"))

    def _get_tags(self) -> dict:
        request = urllib.request.Request("{}/api/tags".format(self.base_url))

        try:
            with urllib.request.urlopen(
                request, timeout=_TAGS_TIMEOUT_SECONDS
            ) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise EngineUnavailableError(
                "Ollama daemon is unreachable at {}: {}".format(self.base_url, exc)
            )

        try:
            decoded = json.loads(raw)
        except ValueError:
            raise EngineUnavailableError(
                "Ollama daemon at {} returned invalid JSON from "
                "/api/tags".format(self.base_url)
            )

        if not isinstance(decoded, dict):
            raise EngineUnavailableError(
                "Ollama daemon at {} returned an unexpected /api/tags "
                "response shape".format(self.base_url)
            )
        return decoded

    def _tag_names(self, tags_response: dict) -> List[str]:
        models = tags_response.get("models") or []
        return [model.get("name") for model in models if isinstance(model, dict)]

    def _gpu_snapshot(self):
        cuda = self.gpu_manager.current().cuda_available
        gpu = self.gpu_manager.gpu_name()
        return cuda, gpu
