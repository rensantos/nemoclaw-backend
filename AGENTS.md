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

- CLI commands delegate to services.
- Do not put business logic, timing logic, model loading, GPU inspection, or
  provider logic directly in `cli.py`.
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

## Engineering Rules

- Keep code modular, testable, explicit, and production-quality.
- Keep dependencies conservative; no new frameworks without strong
  justification.
- Do not introduce Docker, LangChain, vLLM, Ollama, Core orchestration, or new
  providers unless explicitly requested.
- Every implementation phase needs tests; existing tests must keep passing.
- Update `README.md` and relevant `docs/` files with each phase.
- Risky phases involving runtime state, GPU memory, process management, or
  active requests need a design document in `docs/` before implementation.
- Never fake unavailable functionality. Report it clearly, as
  `BenchmarkService.first_token_latency` does while streaming is unavailable.
- Do not overwrite `requirements.txt` with `pip freeze`; it is
  human-maintained direct runtime dependencies only.

## Current State

Completed through Phase 4:

- FastAPI OpenAI-compatible API
- YAML config with env overrides
- Typer CLI: start/stop/restart/status/health/config/logs
- `InferenceService`, `InferenceEngine`, `TransformersEngine`
- `ModelManager`, `GPUManager`, `BenchmarkService`
- model, GPU, and benchmark CLI commands

Next milestone: Phase 5 Model Lifecycle. Design first:
`docs/model-lifecycle-design.md`.

## Commands

Run real CLI commands inside the `llm` Conda environment:

```bash
./backend start
./backend stop
./backend restart
./backend status
./backend health
./backend config
./backend logs
./backend model list
./backend model current
./backend model use <model_id>
./backend model info <model_id>
./backend gpu list
./backend gpu current
./backend gpu monitor
./backend benchmark latency
./backend benchmark throughput
./backend benchmark vram
./backend benchmark first-token-latency
```

Tests:

```bash
python3 -m unittest discover -s tests
```

Useful docs: `docs/architecture.md`, `docs/developed.md`,
`docs/future-tasks.md`, `docs/problems.md`, `docs/model-lifecycle-design.md`.
