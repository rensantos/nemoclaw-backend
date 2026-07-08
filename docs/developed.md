# Developed So Far

This backend is a small OpenAI-compatible HTTP API for serving one local
Hugging Face Transformers causal language model on the UBI machine.

## Current Architecture

- `app.py` creates the FastAPI app and includes the API router.
- `api.py` defines the HTTP endpoints:
  - `GET /health`
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `POST /generate` as a small compatibility endpoint.
- `services/inference.py` contains `InferenceService`, which is the application
  boundary between FastAPI and inference engines.
- `engines/base.py` defines the minimal `InferenceEngine` interface.
- `engines/transformers_engine.py` contains all Hugging Face Transformers,
  PyTorch, tokenizer, and CUDA-specific runtime logic.
- `config.py` loads configuration from:
  1. environment variables
  2. `config/config.yaml`
  3. hardcoded defaults
- `model_runtime.py` is a thin compatibility facade over the inference service.
- `schemas.py` contains Pydantic request models.
- `server.py` preserves `uvicorn server:app` compatibility and can also run
  the server directly with `python server.py`.
- `backend` and `cli.py` provide the Typer command-line interface:
  `backend start`, `backend stop`, `backend restart`, `backend status`,
  `backend health`, `backend config`, `backend logs`, and `backend model ...`.
- `backend model list`, `backend model current`, `backend model use`, and
  `backend model info` manage configured model selection in YAML. Runtime
  hot-switching is not implemented.
- `services/model.py` contains `ModelManager`, which owns configured model
  metadata, selected/default model validation, and YAML selection updates.
- `config.py` acts as the configuration provider; model business logic lives in
  `ModelManager`.
- `services/gpu.py` contains `GPUManager`, which is the single service used for
  GPU discovery and status reporting through `nvidia-smi` and optional
  `torch.cuda` checks.
- `backend gpu list`, `backend gpu current`, and `backend gpu monitor` expose
  informational GPU status in the CLI without adding GPU selection or
  scheduling.
- `services/benchmark.py` contains `BenchmarkService`, which owns benchmark
  execution against the local OpenAI-compatible HTTP API and formats benchmark
  results for CLI or JSON output.
- `backend benchmark latency`, `backend benchmark throughput`, `backend
  benchmark vram`, and `backend benchmark first-token-latency` expose Phase 4
  benchmarks without placing timing logic inside the CLI.
- Benchmarking measures the backend as a client would by calling
  `/v1/chat/completions`; it does not call Transformers or engines directly.
- First-token latency reports unavailable while streaming is not implemented,
  rather than inventing a metric.
- The CLI launches Uvicorn with the resolved YAML/env configuration, writes
  `run/backend.pid`, writes `logs/backend.log`, reports health, and can show or
  follow logs.
- `backend status` uses PID, `/health`, configured port connectivity, and
  backend process matching so externally started backends are reported as
  running even when they are not managed by the CLI.
- `backend stop` only stops CLI-managed PIDs. If an external launcher or systemd
  service owns the process, it reports that state and avoids killing unrelated
  Python or Uvicorn processes.
- PID-file stop behavior validates the PID command line before sending signals
  so a recycled PID is not treated as a managed backend.
- `scripts/` contains temporary shell wrappers around the Python CLI plus config
  editing.
- `tests/test_cli.py` contains stdlib helper tests for CLI command generation,
  health parsing, GPU status parsing, PID parsing, unmanaged status detection,
  and safe stop behavior.
- `tests/test_inference_service.py` contains stdlib tests for service
  delegation and the API boundary that keeps Transformers out of `api.py`.
- `tests/test_model_manager.py` contains stdlib tests for configured model
  discovery and YAML selection updates.
- `tests/test_gpu_manager.py` contains stdlib tests for GPUManager parsing and
  current-GPU reporting.
- `tests/test_benchmark_service.py` contains stdlib tests for latency,
  throughput, VRAM snapshots, first-token-latency availability, and concurrency
  reporting.
- `requirements.txt` is human-maintained and records direct runtime
  dependencies only. It should not be generated from `pip freeze`.
- `services/lifecycle.py` contains the `LifecycleState` enum and the fixed
  stub response builder for `/admin/model/*` (Phase 5).
- `InferenceService` owns `lifecycle_state`, currently always `ready`;
  `/health` reports it alongside `status`, `model`, `cuda`, and `gpu`.
  `backend status` prints `Lifecycle: <state>`.
- `api.py` exposes `/admin/model/load`, `/admin/model/unload`, and
  `/admin/model/switch` under `/admin/`, separate from `/v1/*`. For a
  well-formed request each returns HTTP `501` with a fixed JSON body and
  never changes `lifecycle_state` or touches the engine/CUDA. `load` and
  `switch` require a `ModelLifecycleRequest` body (`model_id: str`); a
  missing/malformed body returns FastAPI's standard `422` instead of the
  `501` stub. `unload` takes no body and always returns `501`.
- `backend model load|unload|switch` call those endpoints, print the
  `detail` message, and exit non-zero. No timeout/wait/progress logic yet.
- The `/admin/model/*` namespace is the adopted contract for control
  operations, superseding the Nemoclaw system spec's flat `/models/load`,
  `/models/unload`, `/engines/switch` paths (spec §6.5), to keep unstable
  admin operations separate from the stable `/v1/*` surface.
- The authoritative API contract is pinned in
  `openapi/backend-node.openapi.yaml` (OpenAPI 3.1), with a human-readable
  companion in `docs/api-contract.md` and a forward contract for Backend
  Registry self-registration in `docs/registration-schema.json`.
- `docs/ollama-engine-design.md` is the design document for `OllamaEngine`.
  It decides model-resolution semantics (reject unservable models with a
  404 `model_not_found`, response `"model"` always the served model) and
  model-listing semantics (`/v1/models` lists only the currently servable
  model, raw Ollama tags as ids, no namespacing) for all engines, keeps one
  engine active per backend instance for this phase, and maps each
  `InferenceEngine` method to the Ollama daemon's HTTP API.
- OllamaEngine Increment 1 (config + engine factory, no Ollama HTTP logic
  yet): `config.yaml`'s `backend.engine` (`transformers` | `ollama`,
  default `transformers`) selects the active engine, with an `ENGINE` env
  override; `config.py`'s `load_config()` fails fast with a clear error if
  the value isn't one of the two. `services/inference.py`'s new
  `_build_engine(config)` helper is the factory `create_inference_service()`
  calls to construct the selected engine, replacing the old unconditional
  `TransformersEngine(settings)` construction. `engines/ollama_engine.py`
  contains a skeleton `OllamaEngine`: `load_model()` is a safe no-op (so
  `engine: ollama` starts up cleanly, since `InferenceService.__init__`
  calls `load_model()` eagerly); every other method raises
  `NotImplementedError` pointing at the design doc until later increments
  implement it. With `engine: transformers` (the default), behavior is
  unchanged from before this increment.

## Configuration

Normal operation is controlled by `config/config.yaml`.

Environment variables can override YAML values for one-off runs:

- `MODEL_ID`
- `GPU`
- `HOST`
- `PORT`
- `MAX_TOKENS_DEFAULT`
- `TEMPERATURE_DEFAULT`

## Compatibility

The backend intentionally avoids Docker, vLLM, LangChain, Ollama, OpenAI cloud
calls, routing, VPN, and SSH logic. Its only responsibility is serving a local
Transformers model through an OpenAI-compatible API.
