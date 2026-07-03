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
- `config.py` loads configuration from:
  1. environment variables
  2. `config/config.yaml`
  3. hardcoded defaults
- `model_runtime.py` loads the tokenizer and model once at startup using
  Transformers, `torch_dtype=torch.float16`, and `device_map="auto"`.
- `schemas.py` contains Pydantic request models.
- `server.py` preserves `uvicorn server:app` compatibility and can also run
  the server directly with `python server.py`.
- `backend` and `cli.py` provide the Typer command-line interface:
  `backend start`, `backend stop`, `backend restart`, `backend status`,
  `backend health`, `backend config`, and `backend logs`.
- The CLI launches Uvicorn with the resolved YAML/env configuration, writes
  `run/backend.pid`, writes `logs/backend.log`, reports health, and can show or
  follow logs.
- `scripts/` contains temporary shell wrappers around the Python CLI plus config
  editing.
- `tests/test_cli.py` contains stdlib helper tests for CLI command generation,
  health parsing, GPU status parsing, and PID parsing.
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
