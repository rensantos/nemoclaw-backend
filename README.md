# Nemoclaw Backend

OpenAI-compatible FastAPI backend for local and future provider-backed
inference management.

Nemoclaw Backend is the reusable, unified inference management backend for
Nemoclaw. The current runtime serves one local Hugging Face Transformers causal
language model, but the backend boundary is broader than Transformers.

Nemoclaw Backend owns:

- LLM and inference functionality
- model providers and engines
- model selection and model metadata
- the inference API
- benchmarking
- GPU and runtime inspection
- future Ollama, vLLM, llama.cpp, and OpenAI-compatible engines

Nemoclaw Core owns agents, memory, planning, skills, RAG, research workflows,
and orchestration. Core should call the backend API for inference, model
listing, benchmarking, and runtime inspection instead of duplicating those
capabilities.

## Dependency Management

`requirements.txt` is human-maintained and lists only direct runtime
dependencies. Do not overwrite `requirements.txt` with `pip freeze`.

`requirements-lock.txt` may be generated later with `pip freeze` for exact
reproducibility, but it is ignored for now.

Install dependencies inside the `llm` Conda environment:

```bash
source ~/miniforge3/bin/activate
conda activate llm
pip install -r requirements.txt
```

On the current UBI server, PyTorch is installed through Conda because of the
old NVIDIA driver/CUDA stack:

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
```

## Project Notes

This project should keep a short written trail for every maintenance pass:

- `docs/architecture.md` explains the API, service, and engine layers.
- `docs/developed.md` records what has been built.
- `docs/problems.md` records known problems and verification gaps.
- `docs/future-tasks.md` records follow-up work.

When behavior changes, update the relevant doc in the same pass.

## Configuration

Normal operation is controlled by `config/config.yaml`:

```yaml
backend:
  host: 127.0.0.1
  port: 8000
  gpu: 0

model:
  id: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  max_tokens_default: 256
  temperature_default: 0.7
```

Configuration priority is:

1. Environment variables
2. `config/config.yaml`
3. Hardcoded defaults

Supported environment variable overrides:

- `MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `GPU=0`
- `HOST=127.0.0.1`
- `PORT=8000`
- `MAX_TOKENS_DEFAULT=256`
- `TEMPERATURE_DEFAULT=0.7`

Edit the configuration file with:

```bash
./scripts/edit-config.sh
```

To change the model, edit `model.id`:

```yaml
model:
  id: TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

To change the GPU, edit `backend.gpu`:

```yaml
backend:
  gpu: 0
```

For one-off overrides, keep the YAML unchanged and pass environment variables:

```bash
MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0 GPU=0 ./scripts/start.sh
```

## CLI

Use the Python CLI for normal operations:

```bash
./backend
./backend start
./backend stop
./backend restart
./backend status
./backend health
./backend config
./backend logs
./backend model list
./backend model current
./backend model use TinyLlama/TinyLlama-1.1B-Chat-v1.0
./backend model info TinyLlama/TinyLlama-1.1B-Chat-v1.0
./backend gpu list
./backend gpu current
./backend gpu monitor
./backend benchmark latency
./backend benchmark throughput
./backend benchmark vram
./backend benchmark first-token-latency
```

`backend status` prints the active model, GPU, host, port, health, lifecycle
state, VRAM, and temperature. `backend health` calls `/health`. `backend
config` prints the active configuration after YAML and environment overrides.
`backend logs` shows `logs/backend.log`, and `backend logs --follow` tails it
continuously.

`Lifecycle` reports the runtime state owned by `InferenceService`: `ready`,
`loading`, `unloading`, `switching`, `unloaded`, or `degraded`. Today
`InferenceService` always reports `ready` after startup; load/unload/switch
transitions are a future increment. See
`docs/model-lifecycle-design.md`.

Status uses multiple signals so it still reflects reality when the backend was
started outside the CLI:

- `run/backend.pid`, when present
- `/health`
- configured host/port connectivity
- a narrow backend process match

`Managed by CLI: yes` means `backend stop` can safely stop the PID recorded by
the CLI. The CLI also checks that the PID still looks like a Nemoclaw backend
process before stopping it. If the backend is running but unmanaged, `backend
stop` reports that state and refuses to kill processes automatically.

The CLI stores runtime state in ignored local directories:

- `run/backend.pid`
- `logs/backend.log`

The existing shell scripts are kept temporarily as wrappers around the CLI.

## Model Management

Model management in Phase 2 is configuration-level only:

- Configured models are entries in `config/config.yaml`.
- The selected/default model is `model.id` in `config/config.yaml`.
- The loaded model is whatever the currently running backend process loaded at
  startup.

The CLI delegates model metadata and selection work to `ModelManager` in
`services/model.py`. `backend model use <model_id>` updates the selected/default
model in YAML. It does not hot-switch the running backend. If the backend is
already running, the CLI prints that a restart is required.

Examples:

```bash
./backend model list
./backend model current
./backend model info TinyLlama/TinyLlama-1.1B-Chat-v1.0
./backend model use TinyLlama/TinyLlama-1.1B-Chat-v1.0
./backend restart
```

## GPU Management

GPU management in Phase 3 is informational only. The CLI delegates GPU discovery
and monitoring to `GPUManager` in `services/gpu.py`; GPU commands do not run
`nvidia-smi` directly.

Examples:

```bash
./backend gpu list
./backend gpu current
./backend gpu monitor
./backend gpu monitor --interval 5
```

`backend gpu list` shows detected GPU index, name, total/used/free VRAM,
temperature, utilization, and driver version.

`backend gpu current` shows the configured backend GPU, selected CUDA device,
current model, available memory, CUDA availability, and driver version.

`backend gpu monitor` refreshes utilization, VRAM usage, and temperature until
Ctrl+C.

This phase does not implement GPU selection, multi-GPU scheduling, MIG, CUDA
affinity, or dashboards. Benchmark commands are provided separately by
`BenchmarkService`.

## Benchmarking

Benchmarking in Phase 4 is owned by `BenchmarkService` in
`services/benchmark.py`. The CLI delegates to the service, and the service
benchmarks the backend through the local OpenAI-compatible HTTP endpoint:

```text
CLI
  -> BenchmarkService
    -> http://HOST:PORT/v1/chat/completions
```

Examples:

```bash
./backend benchmark latency
./backend benchmark throughput --runs 5 --max-tokens 128
./backend benchmark vram --prompt "Summarize Nemoclaw in one sentence."
./backend benchmark first-token-latency
./backend benchmark latency --json
```

Supported options:

- `--prompt`
- `--max-tokens`
- `--runs`
- `--concurrency`
- `--json`

`--concurrency` is accepted so command shape is stable for future automation,
but Phase 4 still runs requests sequentially. `first-token-latency` reports
that the metric is unavailable until streaming is implemented; it does not fake
the number.

## Start

```bash
./backend start
```

Example output:

```text
Backend started with PID 12345
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
GPU: 0
URL: http://127.0.0.1:8000
Log: /home/renatobox/ubi-a4000/logs/backend.log
Health: ok
```

Wrapper command:

```bash
source ~/miniforge3/bin/activate
conda activate llm
./scripts/start.sh
```

## Stop, Status, Logs

```bash
./backend status
./backend logs
./backend logs --follow
./backend stop
```

Example status output:

```text
Backend status
Running: yes
Managed by CLI: no
PID: 12345
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
GPU: 0
Host: 127.0.0.1
Port: 8000
Health: ok
Lifecycle: ready
Port open: yes
Process match: yes
VRAM: 512 / 16384 MiB
Temperature: 45 C
Log: /home/renatobox/ubi-a4000/logs/backend.log
```

## Local Tests

```bash
python -m unittest discover -s tests
```

## API Tests

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {"role": "user", "content": "Say hello from Nemoclaw in one sentence."}
    ],
    "max_tokens": 64,
    "temperature": 0.7
  }'
```
