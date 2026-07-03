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
