"""OllamaEngine (docs/ollama-engine-design.md Increments 2-3).

Increment 2 (read paths): load_model()/health()/list_models() against a
live Ollama daemon's GET /api/tags. No pulling: load_model() only confirms
the configured tag is already present locally (Section 5, Non-goals).

Increment 3: chat()/generate_text() against POST /api/chat and
POST /api/generate, including Section 1's model-resolution decision
(reject a mismatched requested model, never silently substitute) and
Section 4's token-usage mapping. unload_model() remains unimplemented
until Increment 4.
"""

import json
import logging
import time
import urllib.error
import urllib.request
from typing import List, Optional

from engines.base import EngineUnavailableError, InferenceEngine, ModelNotFoundError
from services.gpu import GPUManager


_NOT_IMPLEMENTED = (
    "OllamaEngine.{method}() is not implemented yet. See "
    "docs/ollama-engine-design.md Section 6 for the increment that "
    "implements it."
)

_TAGS_TIMEOUT_SECONDS = 5

# Generation calls can legitimately take much longer than a reachability
# probe. Distinct, generous fixed timeout; making this operator-configurable
# is deferred (docs/ollama-engine-design.md Section 7, "Timeout behavior" -
# resolved here for Increment 3 as a fixed value, not full configurability).
_GENERATE_TIMEOUT_SECONDS = 120

_logger = logging.getLogger(__name__)


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

    def chat(
        self,
        messages: List,
        max_tokens: Optional[int],
        temperature: Optional[float],
        requested_model: Optional[str] = None,
    ):
        self._check_requested_model(requested_model)
        max_new_tokens = (
            self.config.max_tokens_default if max_tokens is None else max_tokens
        )
        temp = self.config.temperature_default if temperature is None else temperature

        payload = {
            "model": self.model_id,
            "messages": self._message_dicts(messages),
            "stream": False,
            "options": {"temperature": temp, "num_predict": max_new_tokens},
        }
        response = self._post("/api/chat", payload)
        content = (response.get("message") or {}).get("content", "")
        prompt_tokens, completion_tokens = self._usage(response)

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def generate_text(self, prompt: str, max_new_tokens: int, temperature: float):
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_new_tokens},
        }
        response = self._post("/api/generate", payload)
        return {"model": self.model_id, "response": response.get("response", "")}

    def _check_requested_model(self, requested_model: Optional[str]) -> None:
        if requested_model is not None and requested_model != self.model_id:
            raise ModelNotFoundError(requested_model, self.model_id)

    def _usage(self, response: dict):
        prompt_tokens = response.get("prompt_eval_count")
        completion_tokens = response.get("eval_count")
        missing = []
        if prompt_tokens is None:
            missing.append("prompt_eval_count")
            prompt_tokens = 0
        if completion_tokens is None:
            missing.append("eval_count")
            completion_tokens = 0
        if missing:
            _logger.warning(
                "Ollama response for model '%s' is missing token usage "
                "field(s) %s; reporting 0",
                self.model_id,
                ", ".join(missing),
            )
        return prompt_tokens, completion_tokens

    def _message_dicts(self, messages: List) -> List[dict]:
        return [{"role": message.role, "content": message.content} for message in messages]

    def _get_tags(self) -> dict:
        request = urllib.request.Request("{}/api/tags".format(self.base_url))
        return self._request(request, _TAGS_TIMEOUT_SECONDS, "/api/tags")

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "{}{}".format(self.base_url, path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request(request, _GENERATE_TIMEOUT_SECONDS, path)

    def _request(self, request: urllib.request.Request, timeout: int, path: str) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise EngineUnavailableError(
                "Ollama daemon is unreachable at {}: {}".format(self.base_url, exc)
            )

        try:
            decoded = json.loads(raw)
        except ValueError:
            raise EngineUnavailableError(
                "Ollama daemon at {} returned invalid JSON from {}".format(
                    self.base_url, path
                )
            )

        if not isinstance(decoded, dict):
            raise EngineUnavailableError(
                "Ollama daemon at {} returned an unexpected response shape "
                "from {}".format(self.base_url, path)
            )
        return decoded

    def _tag_names(self, tags_response: dict) -> List[str]:
        models = tags_response.get("models") or []
        return [model.get("name") for model in models if isinstance(model, dict)]

    def _gpu_snapshot(self):
        cuda = self.gpu_manager.current().cuda_available
        gpu = self.gpu_manager.gpu_name()
        return cuda, gpu
