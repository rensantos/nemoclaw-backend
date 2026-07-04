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

Next milestone: Phase 5 Increment 3, real model load/unload/switch behavior
per `docs/model-lifecycle-design.md`.

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
`docs/future-tasks.md`, `docs/problems.md`, `docs/model-lifecycle-design.md`.
