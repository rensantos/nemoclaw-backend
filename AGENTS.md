# AGENTS.md

Behavioral contract for coding agents working on `nemoclaw-backend`.

## Project Identity

`nemoclaw-backend` is a unified inference management backend. It is not only a
Transformers server, and it is not an orchestrator.

Current runtime: FastAPI + Hugging Face Transformers serving a local model
through an OpenAI-compatible API. Future provider/engine support belongs here.

## Critical Boundary

Nemoclaw Core owns:

- agents
- memory
- planning
- skills
- RAG
- research workflows
- orchestration

Nemoclaw Backend owns all LLM/inference functionality:

- engines and model providers
- model selection and model metadata
- inference API
- benchmarking
- GPU/runtime inspection

Never add Core logic to this repo. Never duplicate backend-owned model listing,
model selection, benchmarking, provider/engine support, or GPU/runtime
inspection in Core.

## Architecture Rules

```text
CLI (Typer)
  -> Services: InferenceService, ModelManager, GPUManager, BenchmarkService
  -> Engines: InferenceEngine, TransformersEngine
  -> CUDA / GPU
```

Future engines may include Ollama, vLLM, llama.cpp, and OpenAI-compatible
providers. Do not implement them until explicitly requested.

- Every capability has exactly one owner.
- Services own business capabilities: inference coordination, model
  management, GPU inspection, benchmarking, and lifecycle.
- Engines own runtime-specific implementation: how Transformers or future
  Ollama/vLLM/llama.cpp/OpenAI-compatible backends load models and run
  inference.
- CLI commands and FastAPI routes are delivery surfaces, never owners. They
  validate input, delegate to a service, and format output.
- Before implementing a new capability, identify its owner. If no existing
  service or engine is the clear owner, introduce the correct one first.
- CLI commands delegate to services.
- Do not put business logic, timing logic, model loading, GPU inspection, or
  provider logic directly in `cli.py` or FastAPI routes.
- New backend capabilities should become services or engines.
- Benchmarks must go through the local OpenAI-compatible endpoint, never
  Transformers directly.
- `api.py` must stay independent of Transformers/CUDA internals.
- `ModelManager` owns configured models and selected/default model metadata.
- `InferenceService` owns runtime inference boundaries.
- `GPUManager` owns GPU discovery/status.
- `BenchmarkService` owns benchmark execution/formatting.

## Environment

- Deployment: UBI server, Ubuntu 18, RTX A4000 16GB.
- Conda: `source ~/miniforge3/bin/activate` then `conda activate llm`.
- Default server: `127.0.0.1:8000`.
- CLI dependencies such as `typer` are expected inside `llm`; `./backend ...`
  may fail outside that environment.

## API Stability

Keep these endpoints OpenAI-compatible:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

No breaking changes to `/v1/*` without explicit approval. Preserve
`uvicorn server:app --host 127.0.0.1 --port 8000` compatibility.

`openapi/backend-node.openapi.yaml` is the authoritative API contract. Any
endpoint addition or change (new route, changed request/response shape,
changed status code, changed `x-implementation-status`) must update that
file in the same increment. See `docs/api-contract.md` for the contract's
tier model and stability rules.

## Versioning
The project is pre-1.0. Minor bumps mark backend capabilities or architecture
milestones. Patch bumps mark fixes, hardening, or docs. Do not break `/v1/*`
without explicit architectural approval. Tags should mark validated runtime
milestones, not merely passing tests.

## Engineering Rules

- Keep code modular, testable, explicit, and production-quality.
- Keep dependencies conservative; no new frameworks without strong
  justification.
- Do not introduce Docker, LangChain, vLLM, Ollama, Core orchestration, or new
  providers unless explicitly requested.
- Every implementation phase needs tests; existing tests must keep passing.
- Update `README.md` and relevant `docs/` files with each phase.
- At the end of every phase, update `README.md`, relevant `docs/`, and the
  Current State section of this file. Do not let `AGENTS.md` drift from the
  code.
- Risky phases involving runtime state, GPU memory, process management, or
  active requests need a design document in `docs/` before implementation.
- Never fake unavailable functionality. Report it clearly, as
  `BenchmarkService.first_token_latency` does while streaming is unavailable.
- Do not overwrite `requirements.txt` with `pip freeze`; it is
  human-maintained direct runtime dependencies only.
- Every completed increment or phase ends with: run the full test suite,
  commit all changes with a concise descriptive message, and push to origin
  main. Work is not "done" until it is pushed. Never leave completed work
  uncommitted.

## Current State

This section must be updated at the end of every phase so agent guidance stays
aligned with the code.

Completed through Phase 4:

- FastAPI OpenAI-compatible API
- YAML config with env overrides
- Typer CLI: start/stop/restart/status/health/config/logs
- `InferenceService`, `InferenceEngine`, `TransformersEngine`
- `ModelManager`, `GPUManager`, `BenchmarkService`
- model, GPU, and benchmark CLI commands

Phase 5 Increment 1 (state reporting only, no load/unload/switch):

- `LifecycleState` enum in `services/lifecycle.py`: `unloaded`, `loading`,
  `ready`, `unloading`, `switching`, `degraded`. Matches the state set and
  transition table in `docs/model-lifecycle-design.md`.
- `InferenceService` owns `lifecycle_state`, currently always `ready` after
  the existing startup load.
- `/health` includes `lifecycle_state` alongside the existing `status`,
  `model`, `cuda`, `gpu` fields.
- `./backend status` prints `Lifecycle: <state>`.
- No runtime load/unload/switch, no worker supervision, no CUDA changes yet.

Phase 5 Increment 2 (command surface stubs, no lifecycle behavior):

- Management endpoints under `/admin/model/` (`load`, `unload`, `switch`),
  separate from `/v1/*`, defined in `docs/model-lifecycle-design.md`.
- For well-formed requests, each returns HTTP `501` with a fixed JSON body
  built by `InferenceService.lifecycle_stub_response()` /
  `services/lifecycle.lifecycle_not_implemented_response()`. `load` and
  `switch` require a `ModelLifecycleRequest` body (`model_id: str`); a
  missing/malformed body fails FastAPI validation first and returns the
  standard `422`, not the `501` stub. `unload` takes no body and always
  returns `501`. Calling any of them never changes `lifecycle_state` and
  never touches the engine or CUDA.
- `./backend model load|unload|switch` call those endpoints, print the
  `detail` message, and exit non-zero. No timeout/wait/progress logic yet;
  that belongs to the real implementation.

OllamaEngine Increment 1 (config + engine factory):

- `config.yaml`'s `backend.engine` (`transformers` | `ollama`, default
  `transformers`), `ENGINE` env override, fail-fast on invalid values.
- `services/inference.py`'s `_build_engine()` factory selects the engine
  class. `engine: transformers` behavior is unchanged.

OllamaEngine Increment 2 (real read paths, live-validated on a Local Node
running actual Ollama — never UBI, per `docs/architecture.md`'s Target
deployment topology):

- `backend.ollama_host` (`config.yaml`, default `http://127.0.0.1:11434`,
  `OLLAMA_HOST` env override) is the daemon's base URL.
- `OllamaEngine.load_model()` calls `GET /api/tags` and confirms the
  configured `model.id` tag is present; never pulls. A missing tag raises
  a clear startup error naming `ollama pull <tag>`.
- `OllamaEngine.health()` / `list_models()` also use `GET /api/tags`;
  `list_models()` returns only the one configured tag, `owned_by:
  "ollama"`. `cuda`/`gpu` are sourced from `GPUManager` (new
  `GPUManager.gpu_name()`), not a direct `torch.cuda` call, since the
  backend process doesn't own Ollama's CUDA context.
- New `engines.base.EngineUnavailableError` (optional `partial_health`
  payload) is raised when the daemon is unreachable.
  `InferenceService.health()` catches it, transitions `lifecycle_state` to
  `degraded`, and projects `HealthResponse.status` from `lifecycle_state`
  via the new `services/lifecycle.health_status_for_lifecycle_state()`
  (`ready`->`ok`, `degraded`->`degraded`, else->`unavailable`) instead of
  trusting whatever the engine returns — matching the mapping already
  pinned in `openapi/backend-node.openapi.yaml`'s `HealthResponse.status`.
- `unload_model()` still raises `NotImplementedError` (Increment 4).

OllamaEngine Increment 3 (`chat()`/`generate_text()`, live-validated
against a real Local Node and real Ollama models):

- `OllamaEngine.chat()` posts `POST /api/chat`; `generate_text()` posts
  `POST /api/generate`. Both use a separate, generous fixed 120s timeout
  (distinct from read-path calls' 5s), resolving the "Timeout behavior"
  question `docs/ollama-engine-design.md` Section 7 left open for this
  increment — full operator-configurability of that timeout is deferred,
  not decided as unnecessary.
- Model-resolution decision (`docs/ollama-engine-design.md` Section 1) is
  enforced only in `OllamaEngine.chat()`: a request naming a `model` other
  than this instance's configured/servable model raises new
  `engines.base.ModelNotFoundError` before any daemon call, which `api.py`
  turns into HTTP `404` with the pinned `model_not_found` error shape.
  `TransformersEngine.chat()` keeps its existing echo-and-serve quirk
  unchanged (out of scope, per that section's explicit scope boundary).
  `GenerateRequest` has no `model` field, so `/generate` has no
  model-resolution check to make.
  `InferenceEngine.chat()`'s signature grew an optional `requested_model`
  parameter (both engines implement it; `TransformersEngine` accepts and
  ignores it) to carry the client's requested model id down to the engine
  that needs to check it.
- `EngineUnavailableError` during `chat()`/`generate_text()` now also
  transitions `InferenceService`'s `lifecycle_state` to `degraded` (same
  as a failed `health()`); `api.py` turns it into HTTP `503` on both
  `/v1/chat/completions` and `/generate`.
- Token-usage mapping (`docs/ollama-engine-design.md` Section 4):
  `prompt_eval_count`/`eval_count` -> `usage.prompt_tokens`/
  `completion_tokens`; missing fields report `0` and log a warning,
  per AGENTS.md's "never fake unavailable functionality" rule.
- `openapi/backend-node.openapi.yaml` amended in this increment (per the
  AGENTS.md rule that contract changes land with the behavior that needs
  them): new `ModelNotFoundResponse` schema, `404`/`503` responses on
  `POST /v1/chat/completions`, `503` on `POST /generate`, updated
  `x-current-behavior` describing the now-engine-dependent model
  resolution. Also fixed a pre-existing drift found while doing this:
  `ModelObject.owned_by` was pinned `const: local`, which `OllamaEngine`
  already violated since Increment 2 (`"ollama"`) — now `type: string`
  with both values documented.
- Live-validated against a real Ollama daemon and real pulled models:
  successful chat with correct token usage, `404 model_not_found` on a
  mismatched model (via real HTTP through `./backend start`), and
  `generate_text()` against `/generate`. Noted, not fixed (out of scope):
  reasoning-capable models like `qwen3` can return empty `content` if the
  token budget is exhausted by hidden "thinking" before an answer is
  reached — confirmed as an honest pass-through of the daemon's own
  response, not an engine bug; Ollama's `"think": false` option is not
  wired up (undecided by the design doc, left for a future increment if
  needed).

OllamaEngine Increment 4 (`unload_model()`, live-validated against a real
Local Node):

- `OllamaEngine.unload_model()` posts `POST /api/generate` with
  `{"model": <configured tag>, "keep_alive": 0}`, asking the daemon to
  evict the model from memory. Best-effort only: the backend does not
  stop or supervise the daemon process, and eviction is not guaranteed to
  be synchronous — live testing showed the real daemon's `GET /api/ps`
  can still list the model for a few seconds after the unload call
  returns before it actually clears.
- Not wired to any live HTTP endpoint (Non-goal, Section 5):
  `/admin/model/unload` stays a `501` stub regardless of engine, exactly
  as `TransformersEngine.unload_model()` remains unreferenced today. This
  completes `OllamaEngine`'s implementation of every `InferenceEngine`
  method per `docs/ollama-engine-design.md`.

OllamaEngine implementation (Increments 1-4) is now complete: config
selection, real read paths, real chat/generation, and best-effort unload,
all live-validated against a real Ollama daemon and real models.

Next milestones: Phase 5 Increment 3 (real model load/unload/switch
behavior, `docs/model-lifecycle-design.md`) is open. So is migrating an
existing Nemoclaw application (e.g. the user's frontend, or the Research
Assistant per `docs/future-tasks.md`) off calling Ollama directly and onto
this backend's `/v1/chat/completions`, now that `OllamaEngine` can
actually serve requests end to end.

## Commands

Run real CLI commands inside the `llm` Conda environment:

```bash
./backend start|stop|restart|status|health|config|logs
./backend model list|current|use <model_id>|info <model_id>
./backend model load|unload|switch <model_id>   # stubs: 501 not implemented
./backend gpu list|current|monitor
./backend benchmark latency|throughput|vram|first-token-latency
```

Tests:

```bash
python3 -m unittest discover -s tests
```

Useful docs: `docs/architecture.md`, `docs/developed.md`,
`docs/future-tasks.md`, `docs/problems.md`, `docs/model-lifecycle-design.md`,
`docs/ollama-engine-design.md`, `docs/api-contract.md`,
`openapi/backend-node.openapi.yaml`.
