"""OllamaEngine (docs/ollama-engine-design.md Increments 2-4).

Increment 2 (read paths): load_model()/health()/list_models() against a
live Ollama daemon's GET /api/tags. No pulling: load_model() only confirms
the configured tag is already present locally (Section 5, Non-goals).

Increment 3: chat()/generate_text() against POST /api/chat and
POST /api/generate, including Section 1's model-resolution decision
(reject a mismatched requested model, never silently substitute) and
Section 4's token-usage mapping.

Increment 4: unload_model(), the keep_alive: 0 mapping.

Phase 5 Increment 3 (docs/model-lifecycle-design.md) wired load/unload/
switch to the live /admin/model/* endpoints for this engine, since the
daemon owns the CUDA context; model_id is mutable from that point on.
"""

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import List, Optional

from engines.base import (
    EngineUnavailableError,
    InferenceEngine,
    ModelNotFoundError,
    ModelUnavailableError,
)
from services.gpu import GPUManager


_TAGS_TIMEOUT_SECONDS = 5

# Generation calls can legitimately take much longer than a reachability
# probe. Distinct, generous fixed timeout; making this operator-configurable
# is deferred (docs/ollama-engine-design.md Section 7, "Timeout behavior" -
# resolved here for Increment 3 as a fixed value, not full configurability).
_GENERATE_TIMEOUT_SECONDS = 120

_logger = logging.getLogger(__name__)


def find_daemon_pids() -> List[int]:
    """PIDs of `ollama serve` processes running on this machine.

    Engine-specific knowledge (how to locate this engine's runtime), kept
    here rather than in GPUManager or the CLI. Returns an empty list when
    the daemon is remote, absent, or pgrep is unavailable - callers must
    treat "no pids" as "cannot inspect", not "nothing running".
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ollama serve"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    pids = []
    for line in result.stdout.split():
        try:
            pid = int(line)
        except ValueError:
            continue
        if _is_ollama_executable(pid):
            pids.append(pid)
    return pids


def find_runtime_pids() -> List[int]:
    """Every PID that may hold GPU memory on our behalf: the `ollama
    serve` daemon plus its descendants.

    The daemon does not hold the model itself - it spawns an `ollama
    runner` child per loaded model, and nvidia-smi attributes the VRAM to
    that child. Matching only the daemon PID would classify our own model
    as another user's job (confirmed live: serve was 23825 while the
    memory belonged to runner 17181).
    """
    daemons = find_daemon_pids()
    if not daemons:
        return []

    children_by_parent = _children_by_parent()
    pids, queue = set(daemons), list(daemons)
    while queue:
        for child in children_by_parent.get(queue.pop(), ()):
            if child not in pids:
                pids.add(child)
                queue.append(child)
    return sorted(pids)


def _children_by_parent():
    """Maps parent PID -> child PIDs by reading /proc/<pid>/stat."""
    children = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return children

    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open("/proc/{}/stat".format(entry), "r") as stat_file:
                fields = stat_file.read().rsplit(")", 1)[-1].split()
            parent = int(fields[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(parent, []).append(int(entry))
    return children


def _is_ollama_executable(pid: int) -> bool:
    """Whether pid is the ollama binary itself, not a shell that merely
    mentions it. `pgrep -f "ollama serve"` also matches the `bash -c ...`
    wrapper the daemon was launched from, which would double-report every
    finding and read as two separate daemons.
    """
    try:
        with open("/proc/{}/cmdline".format(pid), "rb") as cmdline_file:
            argv0 = cmdline_file.read().split(b"\0")[0].decode("utf-8", "replace")
    except (OSError, IndexError):
        # Can't tell - keep it rather than silently dropping a real daemon.
        return True
    return argv0.rsplit("/", 1)[-1] == "ollama"


class OllamaEngine(InferenceEngine):
    """InferenceEngine backed by a live Ollama daemon."""

    # The daemon owns the CUDA context, so a runtime swap here is a
    # tag-presence check plus a pointer change - none of the in-process
    # allocator risk that makes TransformersEngine refuse.
    supports_runtime_lifecycle = True

    def __init__(self, config):
        self.config = config
        self.model_id = config.model.id
        self.base_url = config.backend.ollama_host.rstrip("/")
        self.gpu_manager = GPUManager(config)

    def runtime_pids(self) -> List[int]:
        """The daemon and its model-runner children (see
        find_runtime_pids); their VRAM is ours and reclaimable."""
        return find_runtime_pids()

    def load_model(self, model_id: Optional[str] = None) -> None:
        """Validates the target tag is present locally. Never pulls."""
        target = self.model_id if model_id is None else model_id
        self._require_tag(target)
        self.model_id = target

    def switch_model(self, model_id: str) -> None:
        """Verifies the target before releasing the current model, so a
        missing target leaves the old one serving (the design doc's
        rollback goal, essentially free for a daemon-owned runtime).
        """
        self._require_tag(model_id)
        self.unload_model()
        self.model_id = model_id

    def _require_tag(self, model_id: str) -> None:
        if model_id not in self._tag_names(self._get_tags()):
            raise ModelUnavailableError(
                "Ollama tag '{}' is not present on the daemon at {}. Run "
                "'ollama pull {}' on the machine hosting Ollama, then try "
                "again.".format(model_id, self.base_url, model_id)
            )

    def unload_model(self) -> None:
        """Best-effort: asks the daemon to evict the model from memory.

        Does not stop or supervise the daemon process itself, and eviction
        is not guaranteed to be synchronous from the daemon's side (Section
        4: "the daemon process itself is not stopped or supervised by the
        backend — only the model's residency in daemon memory is
        affected").
        """
        self._post("/api/generate", {"model": self.model_id, "keep_alive": 0})

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
        think: Optional[bool] = None,
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
        self._apply_think(payload, think)
        response = self._post("/api/chat", payload)
        content = (response.get("message") or {}).get("content", "")
        prompt_tokens, completion_tokens = self._usage(response)

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        think: Optional[bool] = None,
    ):
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_new_tokens},
        }
        self._apply_think(payload, think)
        response = self._post("/api/generate", payload)
        return {"model": self.model_id, "response": response.get("response", "")}

    def _check_requested_model(self, requested_model: Optional[str]) -> None:
        if requested_model is not None and requested_model != self.model_id:
            raise ModelNotFoundError(requested_model, self.model_id)

    def _apply_think(self, payload: dict, think: Optional[bool]) -> None:
        """Sets payload["think"] from the request override or
        config.model.think_default, in that priority order. Ollama's
        `think` is a top-level request field (not inside "options"). If
        both resolve to None, the key is omitted entirely so Ollama's own
        default behavior (reasoning-capable models think unless told
        otherwise) is unchanged - this is an opt-in knob, not a new
        default.
        """
        effective_think = (
            self.config.model.think_default if think is None else think
        )
        if effective_think is not None:
            payload["think"] = effective_think

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
