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
import signal
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
# Prompt processing scales with context length, so this is not a "should
# never take longer" bound - it is a cap on how much context is usable at
# all. Measured 2026-08-12: a filled 32768-token prompt took 84s on UBI
# and a filled 65536 took longer than 120s, so the old hardcoded 120
# silently capped both machines well below what their hardware could
# serve. Both ceilings recorded that day as memory limits were this
# timeout: Ollama's own log showed the 65536 model loading fully across
# four GPUs (58.9 GiB of 62.4 available) and then being cut off with
# "context canceled" at exactly 120s.
#
# Overridable so a slow machine or a very large context is a config
# change rather than a code change.
_GENERATE_TIMEOUT_SECONDS = int(os.environ.get("NEMOCLAW_GENERATE_TIMEOUT_SECONDS", "1800"))

# How long Ollama keeps a model resident after this engine's last request
# to it. Sent on every chat/generate/embed call because nothing here ever
# sent one before - Ollama then applied its own compiled-in 5-minute
# default regardless of what a caller's frontend intended. Found live
# 2026-08-19: the calling frontend (nemoclaw-research-assistant) sets
# OLLAMA_KEEP_ALIVE=10m expecting it to reach Ollama, but its own
# OpenAI-compatible client path drops that field before it ever leaves
# the frontend process, and this engine never re-added one of its own -
# so every request actually ran on Ollama's 5-minute default. On a node
# that alternates between a chat model and an embedding model (this
# project's claim-similarity scoring), an idle gap over that default is
# exactly what evicts one to reload the other - a real reload was
# observed within ~15 minutes of a model's last use on a real run.
# Deliberately generous and not per-request configurable: nothing here
# needs finer control than "stay loaded through a normal gap between
# calls," only a longer floor than Ollama's own default.
_KEEP_ALIVE = os.environ.get("NEMOCLAW_OLLAMA_KEEP_ALIVE", "30m")

# KV cache element size. Ollama defaults to fp16 unless cache
# quantisation is configured; over-estimating yields a smaller window
# that still works, under-estimating degrades the backend.
_KV_CACHE_BYTES_PER_ELEMENT = 2

# Left free beyond model + KV cache for activations, the runtime, and
# whatever else shares the machine. Observed failures landed within
# ~1.5 GB of the limit, so this is deliberately generous.
_CONTEXT_MEMORY_HEADROOM_BYTES = 3 * 1024 ** 3


def _available_memory_bytes() -> int:
    """Free system memory, from /proc/meminfo. 0 if it cannot be read."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0

# Multiplier from a tag's on-disk size to the VRAM Ollama actually needs
# to serve it fully on GPU. See OllamaEngine.estimated_vram_mib().
VRAM_OVERHEAD_FACTOR = 1.5

_logger = logging.getLogger(__name__)


_REASONING_END = "</think>"
_REASONING_START = "<think>"


def _split_reasoning(content: str, thinking: Optional[str] = None):
    """Separates a reasoning model's hidden thinking from its answer.

    Three behaviours seen live, all handled here:

    - Ollama already separated it into ``message.thinking`` (dense qwen3
      with think enabled). Content is the answer; just carry the field
      through instead of discarding it, as this engine used to.
    - The chat template leaked it inline, ending with ``</think>``
      (qwen3:30b MoE, which also ignores ``"think": false`` entirely). The
      opening tag is usually consumed by the prompt, so split on the last
      closing marker rather than requiring a matched pair.
    - No reasoning at all (llama3.2, gemma3, mistral, or thinking
      disabled). Nothing matches, so this is a no-op.

    Returns (content, reasoning); reasoning is None when there is none.
    """
    if thinking:
        return content, thinking

    if _REASONING_END not in content:
        return content, None

    reasoning, _, answer = content.rpartition(_REASONING_END)
    reasoning = reasoning.replace(_REASONING_START, "").strip()
    return answer.lstrip("\n"), reasoning or None


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


def daemon_launch_spec(pid: int):
    """Reconstructs how a running daemon was launched, so it can be
    restarted with a different CUDA_VISIBLE_DEVICES.

    Reads argv and environment from /proc rather than assuming the
    documented command line - the deployed daemon has drifted from the
    docs before. Returns (argv, env) or None if either is unreadable.
    """
    try:
        with open("/proc/{}/cmdline".format(pid), "rb") as cmdline_file:
            argv = [
                part.decode("utf-8", "replace")
                for part in cmdline_file.read().split(b"\0")
                if part
            ]
        with open("/proc/{}/environ".format(pid), "rb") as environ_file:
            raw_env = environ_file.read().decode("utf-8", "replace")
    except OSError:
        return None

    if not argv:
        return None

    env = {}
    for entry in raw_env.split("\0"):
        if "=" in entry:
            key, value = entry.split("=", 1)
            env[key] = value
    return argv, env


def daemon_models_path() -> Optional[str]:
    """Where the running daemon stores model blobs.

    Read from the daemon's own environment rather than assumed, for the
    same reason daemon_launch_spec does: the deployed daemon sets
    OLLAMA_MODELS at launch and has drifted from the documented setup
    before. Falls back to Ollama's documented default only when the
    daemon is readable but sets nothing; returns None when there is no
    daemon to ask, so a caller reports "unknown" rather than checking
    free space on an unrelated filesystem.
    """
    for pid in find_daemon_pids():
        spec = daemon_launch_spec(pid)
        if spec is None:
            continue
        _, env = spec
        configured = env.get("OLLAMA_MODELS")
        if configured:
            return configured
        home = env.get("HOME")
        if home:
            return os.path.join(home, ".ollama", "models")
    return None


def daemon_log_path(pid: int) -> Optional[str]:
    """Where a running daemon's stdout goes, from /proc/<pid>/fd/1.

    The log destination comes from shell redirection at launch, so it is
    in neither argv nor the environment - without this a restart would
    silently send the daemon's output to /dev/null and lose serve.log.
    """
    try:
        target = os.readlink("/proc/{}/fd/1".format(pid))
    except OSError:
        return None
    if not target.startswith("/") or target.startswith("/dev/") or "(deleted)" in target:
        return None
    return target


def restart_daemon_pinned(pid: int, gpu_indexes, log_path=None):
    """Stops the daemon at `pid` and relaunches it seeing only
    gpu_indexes.

    CUDA_VISIBLE_DEVICES can only be set at process start, so pinning
    means a restart. This is the one place the backend touches the
    daemon's lifecycle, and only when an operator explicitly asks
    (./backend gpu pin-free). Any resident model is dropped and reloaded
    on the next request.

    Terminates by PID, never `pkill -f`: that pattern was confirmed to
    kill the calling SSH session too (docs/ollama-on-ubi-design.md).
    """
    spec = daemon_launch_spec(pid)
    if spec is None:
        raise RuntimeError(
            "Cannot read the launch command of Ollama daemon PID {}; "
            "refusing to restart it blind.".format(pid)
        )

    argv, env = spec
    env = dict(env)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indexes)

    # Keep writing wherever it was already writing; discarding the
    # daemon's log on restart would lose the only record of its
    # scheduling decisions.
    if log_path is None:
        log_path = daemon_log_path(pid)

    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        time.sleep(0.2)
        try:
            os.kill(pid, 0)
        except OSError:
            break
    else:
        raise RuntimeError(
            "Ollama daemon PID {} did not exit after SIGTERM; leaving it "
            "alone rather than forcing a kill.".format(pid)
        )

    log_target = open(log_path, "ab") if log_path else subprocess.DEVNULL
    try:
        process = subprocess.Popen(
            argv,
            env=env,
            stdout=log_target,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=env.get("PWD") or None,
        )
    finally:
        if log_path:
            log_target.close()
    return process.pid


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
    supports_pull = True
    supports_streaming = True
    # The daemon serves several models concurrently, so an embedding model
    # can answer alongside the loaded chat model without a switch.
    supports_embeddings = True

    def __init__(self, config):
        self.config = config
        self.model_id = config.model.id
        self.base_url = config.backend.ollama_host.rstrip("/")
        self.gpu_manager = GPUManager(config)

    def runtime_pids(self) -> List[int]:
        """The daemon and its model-runner children (see
        find_runtime_pids); their VRAM is ours and reclaimable."""
        return find_runtime_pids()

    def model_storage_path(self) -> Optional[str]:
        """Where the daemon keeps blobs - the filesystem a pull fills."""
        return daemon_models_path()

    def pull_model(self, model_id: str, size_guard=None, on_progress=None):
        """Downloads a tag, streaming Ollama's own progress.

        `size_guard` is called with the total byte count as soon as the
        daemon reports it, and may raise to abort the download. The size
        is genuinely not knowable before starting - this is the first
        point at which a caller can check it against free disk (see
        docs/model-pull-design.md Section 3).

        `on_progress` is called with each raw progress event, for callers
        that want to report a running download. This loop is the only
        place the byte count exists: the call returns when the transfer is
        already over, so anything not handed out here is lost.

        Returns the total bytes reported, or None if the daemon never
        reported one.
        """
        total_bytes = None
        checked = False
        for event in self._post_stream(
            "/api/pull", {"model": model_id, "stream": True}
        ):
            error = event.get("error")
            if error:
                raise ModelUnavailableError(
                    "Ollama daemon at {} could not pull '{}': {}".format(
                        self.base_url, model_id, error
                    )
                )
            if on_progress is not None:
                try:
                    on_progress(event)
                except Exception:
                    # Reporting is a nicety; a broken reporter must not
                    # abandon a transfer that is working. Dropped rather
                    # than retried, so one bad callback cannot log once
                    # per event for the rest of a 9GB download.
                    _logger.warning(
                        "Pull progress reporting failed; continuing without it",
                        exc_info=True,
                    )
                    on_progress = None
            reported = event.get("total")
            if reported and not checked:
                total_bytes = int(reported)
                checked = True
                if size_guard is not None:
                    # Raising here abandons the response, which closes the
                    # connection and stops the transfer.
                    size_guard(total_bytes)
        return total_bytes

    def model_disk_size_mib(self, model_id: Optional[str] = None) -> Optional[int]:
        """On-disk size of a tag from GET /api/tags, or None if unknown."""
        target = self.model_id if model_id is None else model_id
        for model in self._get_tags().get("models") or []:
            if isinstance(model, dict) and model.get("name") == target:
                size = model.get("size")
                if isinstance(size, (int, float)):
                    return int(size / (1024 * 1024))
        return None

    def estimated_vram_mib(self, model_id: Optional[str] = None) -> Optional[int]:
        """Rough VRAM needed to serve a tag fully on GPU.

        Ollama only reports the true figure (memory.required.full) after
        it has loaded the model, so this scales the on-disk weight size by
        VRAM_OVERHEAD_FACTOR. Measured on UBI: qwen3:30b is 17.28 GiB on
        disk and Ollama reported 25.2 GiB required - a 1.46x ratio, driven
        by the KV cache, compute graph and parallel request slots. The
        default rounds that up, because over-estimating costs one extra
        GPU while under-estimating spills a model onto CPU or fails.
        """
        disk_mib = self.model_disk_size_mib(model_id)
        if disk_mib is None:
            return None
        return int(disk_mib * VRAM_OVERHEAD_FACTOR)

    def installed_models(self) -> List[str]:
        """Every tag actually pulled on the daemon, catalogued or not."""
        return sorted(self.pulled_model_sizes())

    def model_runtime_info(self, model_ids: List[str]) -> dict:
        """Which configured tags are actually pulled, how big they are,
        and whether they fit the GPUs this daemon can reach."""
        sizes = self.pulled_model_sizes()
        available = self.visible_vram_mib()

        info = {}
        for model_id in model_ids:
            required = sizes.get(model_id)
            entry = {"pulled": model_id in sizes}
            if required is not None:
                entry["size_mib"] = required
                if available is not None:
                    entry["fits"] = required <= available
            info[model_id] = entry
        return info

    def pulled_model_sizes(self) -> dict:
        """{tag: estimated VRAM MiB} for every tag pulled on the daemon."""
        sizes = {}
        for model in self._get_tags().get("models") or []:
            if not isinstance(model, dict):
                continue
            name, size = model.get("name"), model.get("size")
            if name and isinstance(size, (int, float)):
                mib = int(size / (1024 * 1024))
                sizes[name] = int(mib * VRAM_OVERHEAD_FACTOR)
        return sizes

    def visible_vram_mib(self) -> Optional[int]:
        """Total VRAM the daemon can actually reach, i.e. the sum over the
        GPUs its own CUDA_VISIBLE_DEVICES exposes.

        The daemon's visible set only changes when it restarts, while the
        served model can change at any time, so a switch can outgrow the
        pin it was placed under. None when visibility can't be determined.
        """
        pids = find_daemon_pids()
        if not pids:
            return None

        visible = self.gpu_manager.visible_gpu_indexes_for_process(pids[0])
        if visible is None:
            return None

        visible_set = set(visible)
        total = 0
        for gpu in self.gpu_manager.detect_gpus():
            if str(gpu.index) in visible_set and gpu.memory_total_mib:
                total += gpu.memory_total_mib
        return total or None

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

    def vram_warning_for(self, model_id: str) -> Optional[str]:
        """Logs when a switch target looks too big for the GPUs the daemon
        can currently reach.

        The daemon's GPU set is fixed until it restarts (./backend gpu
        pin-free), but the served model is chosen by the frontend and can
        change at any time - so a switch can outgrow the pin it was placed
        under. Only a warning: the requirement is an estimate, and Ollama
        can still run with layers offloaded to CPU, so refusing on a
        heuristic would block legitimate switches.
        """
        try:
            required = self.estimated_vram_mib(model_id)
            available = self.visible_vram_mib()
        except Exception:
            return None

        if required is None or available is None or required <= available:
            return None

        return (
            "Model '{}' needs roughly {} MiB but the Ollama daemon can only "
            "reach {} MiB of VRAM; it may run partly on CPU or fail to load. "
            "Run './backend gpu pin-free' to re-select GPUs for it.".format(
                model_id, required, available
            )
        )

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
        num_ctx: Optional[int] = None,
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
            "keep_alive": _KEEP_ALIVE,
            "options": {"temperature": temp, "num_predict": max_new_tokens},
        }
        self._apply_num_ctx(payload, num_ctx)
        self._apply_think(payload, think)
        response = self._post("/api/chat", payload)
        message = response.get("message") or {}
        content, reasoning = _split_reasoning(
            message.get("content", ""), message.get("thinking")
        )
        prompt_tokens, completion_tokens = self._usage(response)

        return {
            "content": content,
            "reasoning": reasoning,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def chat_stream(
        self,
        messages: List,
        max_tokens: Optional[int],
        temperature: Optional[float],
        requested_model: Optional[str] = None,
        think: Optional[bool] = None,
            num_ctx: Optional[int] = None,
    ):
        """Streams deltas from Ollama's NDJSON /api/chat.

        Reasoning is kept out of "content" here exactly as in the
        non-streaming path, but the split has to be decided *before* any
        token is emitted, since a stream cannot retract what it already
        sent. `/api/show`'s capabilities list says up front whether a
        model reasons at all, so this is a deterministic decision rather
        than a guess from the token text:

        - No "thinking" capability: every token is the answer. Streams
          immediately, which is the common case (llama3.2, gemma3,
          mistral).
        - Ollama supplies separate `thinking` deltas: emit them as
          reasoning and content as content, both immediately.
        - Thinking-capable but leaking inline (qwen3:30b): the reasoning
          prefix has to be buffered until `</think>` proves where it ends;
          the answer after it streams normally. If the marker never
          arrives the model did not actually reason, so the buffer is
          flushed as content rather than mislabelled.

        Deliberately *not* a generator itself, for the same reason
        InferenceService.chat_stream() is not: the model-resolution guard
        must reject before the response starts. As a generator function
        the guard below would not run until first iteration, by which
        point HTTP 200 has been sent and a 404 is impossible - the client
        would instead see the connection die mid-stream with no [DONE]
        and no in-band error.
        """
        self._check_requested_model(requested_model)
        max_new_tokens = (
            self.config.max_tokens_default if max_tokens is None else max_tokens
        )
        temp = self.config.temperature_default if temperature is None else temperature

        payload = {
            "model": self.model_id,
            "messages": self._message_dicts(messages),
            "stream": True,
            "keep_alive": _KEEP_ALIVE,
            "options": {"temperature": temp, "num_predict": max_new_tokens},
        }
        self._apply_num_ctx(payload, num_ctx)
        self._apply_think(payload, think)

        return self._chat_deltas(payload, "thinking" in self.model_capabilities())

    def _chat_deltas(self, payload: dict, may_reason: bool):
        pending = []
        # The blank line separating </think> from the answer often lands in
        # a later chunk than the marker itself, so trimming only the
        # marker's own chunk would leak a leading newline into content.
        trim_leading = False

        for chunk in self._post_stream("/api/chat", payload):
            message = chunk.get("message") or {}

            thinking = message.get("thinking")
            if thinking:
                # Ollama separated it for us; nothing to disambiguate.
                may_reason = False
                yield {"reasoning": thinking}

            piece = message.get("content") or ""
            if piece:
                if trim_leading:
                    piece = piece.lstrip("\n")
                    if not piece:
                        continue
                    trim_leading = False
                if not may_reason:
                    yield {"content": piece}
                else:
                    pending.append(piece)
                    buffered = "".join(pending)
                    if _REASONING_END in buffered:
                        reasoning, _, answer = buffered.rpartition(_REASONING_END)
                        reasoning = reasoning.replace(_REASONING_START, "").strip()
                        if reasoning:
                            yield {"reasoning": reasoning}
                        answer = answer.lstrip("\n")
                        if answer:
                            yield {"content": answer}
                        else:
                            trim_leading = True
                        pending, may_reason = [], False

            if chunk.get("done"):
                if pending:
                    # No marker ever arrived: this was the answer, not
                    # reasoning. Emitting it as reasoning would hide the
                    # whole response.
                    yield {"content": "".join(pending)}
                prompt_tokens, completion_tokens = self._usage(chunk)
                yield {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                }

    def model_capabilities(self, model_id: Optional[str] = None) -> List[str]:
        """Capability list from POST /api/show (e.g. completion, tools,
        thinking). Empty when unknown, which callers must read as "no
        special capability proven" rather than an error."""
        target = self.model_id if model_id is None else model_id
        try:
            shown = self._post("/api/show", {"model": target})
        except EngineUnavailableError:
            return []
        capabilities = shown.get("capabilities")
        return capabilities if isinstance(capabilities, list) else []

    def model_context_length(self, model_id: Optional[str] = None) -> int:
        """The loaded model's own maximum context, from POST /api/show.

        0 when unknown, which callers must read as "could not tell" rather
        than "no context" - it is a signal to fall back, never a value to
        request.

        Exists so nobody has to write this number down. It is a property
        of the model and it changes whenever the loaded model changes, so
        every copy of it in a config file is a stale cache waiting to be
        wrong: the frontend carried a hand-set 131072 for a machine that
        had since switched models, and a 32768 measured for a model no
        longer loaded. The daemon holding the weights is the only thing
        that actually knows.

        The key is architecture-prefixed (qwen3moe.context_length,
        llama.context_length, ...), so match on the suffix rather than
        guessing the prefix; take the max when several are present.
        """
        target = self.model_id if model_id is None else model_id
        try:
            shown = self._post("/api/show", {"model": target})
        except EngineUnavailableError:
            return 0
        info = shown.get("model_info")
        if not isinstance(info, dict):
            return 0
        lengths = []
        for key, value in info.items():
            if str(key).endswith(".context_length"):
                try:
                    lengths.append(int(value))
                except (TypeError, ValueError):
                    continue
        return max(lengths) if lengths else 0

    def servable_context_length(self, model_id: Optional[str] = None) -> int:
        """The largest context THIS MACHINE can actually hold right now.

        model_context_length() reports what the model supports; this
        reports what the hardware in front of it can serve. They are not
        the same number, and treating them as one is what degraded nodes
        repeatedly: qwen3:30b declares 262,144 and needs 41.1 GiB of KV
        cache to use it, on a box with ~32 GiB.

        Computed, never configured, because it depends on the loaded MODEL
        and the FREE MEMORY together and both move: /model swaps the model
        from a chat message, and another user's job takes memory on a
        shared machine. Any number written into a file is stale the moment
        either changes - which is why the frontend's per-node NUM_CTX
        settings ended up either throttling nodes or crashing them.

        This belongs here rather than in the frontend because the backend
        is the component standing on the hardware. Every machine runs its
        own backend beside its frontend, so a local answer is available
        either way - but UBI runs a backend with NO frontend, and only
        this side can answer for it.

            kv_bytes_per_token = layers * kv_heads * (key_len + value_len) * 2
            servable = (free + reclaimable - model_size - headroom) / kv_per_token

        Returns 0 when any input is unknown, which callers must read as
        "could not tell" and fall back, never as "no context".
        """
        target = self.model_id if model_id is None else model_id
        try:
            shown = self._post("/api/show", {"model": target})
        except EngineUnavailableError:
            return 0
        info = shown.get("model_info")
        if not isinstance(info, dict):
            return 0

        def field(suffix: str) -> int:
            for key, value in info.items():
                if str(key).endswith(suffix):
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return 0
            return 0

        layers = field(".block_count")
        kv_heads = field(".attention.head_count_kv")
        key_len = field(".attention.key_length")
        value_len = field(".attention.value_length") or key_len
        if not (layers and kv_heads and key_len):
            return 0
        kv_per_token = layers * kv_heads * (key_len + value_len) * _KV_CACHE_BYTES_PER_ELEMENT

        model_max = self.model_context_length(target)
        if model_max <= 0:
            return 0

        available = _available_memory_bytes()
        if available <= 0:
            return 0

        # Resident weights are NOT treated as reclaimable. The reasoning
        # that they are - "the model is reloaded at the new context, not
        # loaded twice" - is plausible and empirically wrong: adding them
        # back produced 262,144 on a machine where 104,312 had just
        # degraded the backend, because MemAvailable is already optimistic
        # (it counts reclaimable page cache that a large contiguous
        # allocation cannot actually use).
        #
        # Without the add-back the formula matches every measurement on
        # this machine: it yields ~95k, and 96,156 worked while 104,312
        # and 131,072 both failed. Measurement wins over the model of the
        # world that contradicts it.
        model_bytes = 0
        try:
            for entry in self._get_tags().get("models") or []:
                if entry.get("name") == target or entry.get("model") == target:
                    model_bytes = int(entry.get("size") or 0)
                    break
        except EngineUnavailableError:
            return 0
        if model_bytes <= 0:
            return 0

        usable = available - model_bytes - _CONTEXT_MEMORY_HEADROOM_BYTES
        if usable <= 0:
            return 0
        return max(0, min(model_max, usable // kv_per_token))

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
            "keep_alive": _KEEP_ALIVE,
            "options": {"temperature": temperature, "num_predict": max_new_tokens},
        }
        self._apply_think(payload, think)
        response = self._post("/api/generate", payload)
        text, reasoning = _split_reasoning(
            response.get("response", ""), response.get("thinking")
        )
        return {"model": self.model_id, "response": text, "reasoning": reasoning}

    def embed(self, texts: List[str], model: str) -> List[List[float]]:
        """Embeddings from the daemon, one vector per input, in order.

        Deliberately does NOT call _check_requested_model(): an embedding
        model is a different model from the loaded chat model, and
        rejecting it would make the endpoint unusable. Nothing here
        touches self.model_id or the lifecycle - the daemon serves both
        concurrently.

        Tries /api/embed (current, batch: {"input": [...]} ->
        {"embeddings": [[...]]}) and falls back to the older
        /api/embeddings (single: {"prompt": str} -> {"embedding": [...]}),
        because which one a daemon has depends on its version and the
        older route is still what some installs expose.
        """
        if not texts:
            return []
        try:
            response = self._post(
                "/api/embed", {"model": model, "input": list(texts), "keep_alive": _KEEP_ALIVE}
            )
            vectors = response.get("embeddings")
            if isinstance(vectors, list) and len(vectors) == len(texts):
                return [list(vector) for vector in vectors]
        except EngineUnavailableError:
            # Falls through to the legacy route below rather than failing:
            # an older daemon answers 404 here but serves /api/embeddings.
            pass
        vectors = []
        try:
            for text in texts:
                response = self._post(
                    "/api/embeddings", {"model": model, "prompt": text, "keep_alive": _KEEP_ALIVE}
                )
                vector = response.get("embedding")
                if not isinstance(vector, list):
                    raise EngineUnavailableError(
                        "Ollama returned no embedding for model '{}'".format(model)
                    )
                vectors.append(list(vector))
        except EngineUnavailableError:
            # _request() collapses every HTTP error into
            # EngineUnavailableError, but an un-pulled embedding model is
            # a 404 from a perfectly healthy daemon ("model ... not found,
            # try pulling it first"). Reporting that as 503 would send a
            # caller debugging their daemon when the fix is `ollama pull`,
            # so ask the tag list which of the two it actually is. If the
            # daemon really is down this call raises too, and 503 stands.
            self._require_embedding_model(model)
            raise
        return vectors

    def _require_embedding_model(self, model: str) -> None:
        if model not in self._tag_names(self._get_tags()):
            raise ModelNotFoundError(model, self.model_id)

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

    def _apply_num_ctx(self, payload: dict, num_ctx: Optional[int]) -> None:
        """Sets payload["options"]["num_ctx"] when the caller asked for one.

        Unlike `think`, this goes INSIDE "options" - that is where Ollama
        reads it. Omitted entirely when unset, so Ollama keeps its own
        default and nothing changes for callers that do not ask.

        Why this exists: that Ollama default is 4096, and until 2026-08-12
        this backend never sent num_ctx at all, so every request it served
        was silently truncated to 4096 tokens regardless of what the caller
        assembled. Measured against a ~40k-token prompt, this backend
        reported prompt_tokens=4096 while the same prompt sent straight to
        Ollama with options.num_ctx processed 26,046.

        Non-positive values are ignored rather than forwarded: 0 or a
        negative would be a caller bug, and Ollama's behaviour for them is
        not something to inherit silently.
        """
        if num_ctx is not None and num_ctx > 0:
            payload.setdefault("options", {})["num_ctx"] = num_ctx

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

    def _get_running(self) -> dict:
        """GET /api/ps - models resident right now, with their real
        in-memory size (weights plus any cache already allocated)."""
        request = urllib.request.Request("{}/api/ps".format(self.base_url))
        return self._request(request, _TAGS_TIMEOUT_SECONDS, "/api/ps")

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "{}{}".format(self.base_url, path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request(request, _GENERATE_TIMEOUT_SECONDS, path)

    def _post_stream(self, path: str, payload: dict):
        """Yields parsed objects from an NDJSON response, one per line.

        Kept separate from _post() because the response must be consumed
        lazily - buffering it whole would defeat the point of streaming.
        """
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "{}{}".format(self.base_url, path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=_GENERATE_TIMEOUT_SECONDS
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        decoded = json.loads(line)
                    except ValueError:
                        raise EngineUnavailableError(
                            "Ollama daemon at {} returned invalid JSON from "
                            "{}".format(self.base_url, path)
                        )
                    if isinstance(decoded, dict):
                        yield decoded
        except (urllib.error.URLError, TimeoutError) as exc:
            raise EngineUnavailableError(
                "Ollama daemon is unreachable at {}: {}".format(self.base_url, exc)
            )

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
