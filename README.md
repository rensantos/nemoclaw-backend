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
  engine: transformers  # or: ollama
  ollama_host: http://127.0.0.1:11434  # only used when engine: ollama

model:
  id: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  max_tokens_default: 256
  temperature_default: 0.7
  quantization: none  # or: 4bit | 8bit (TransformersEngine only)
```

`engine: ollama` targets a live Ollama daemon instead of loading a
Transformers model in-process — see `docs/ollama-engine-design.md` and
`docs/architecture.md`'s Target deployment topology for where each engine
is meant to run.

`model.quantization` (`TransformersEngine` only) trades model quality for
VRAM: `4bit`/`8bit` use `bitsandbytes` to fit larger models than fp16
alone allows on a given GPU — see `docs/quantization-design.md`. Requires
`bitsandbytes` installed (in `requirements.txt`).

`model.think_default` (`OllamaEngine` only) controls whether a
reasoning-capable model (e.g. Qwen3) emits a `<think>` trace before its
answer, via Ollama's `think` request field. `null`/omitted (the config
default) leaves Ollama's own per-model default alone. `false` suppresses
reasoning by default — useful since it otherwise silently spends part of
`max_tokens` on the trace before reaching an answer, which can truncate
short responses entirely. A per-request `"think"` field on
`POST /v1/chat/completions` or `POST /generate` overrides this default
either way. `TransformersEngine` ignores this field — no equivalent
concept.

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
- `ENGINE=transformers`
- `OLLAMA_HOST=http://127.0.0.1:11434`
- `MODEL_QUANTIZATION=none`
- `MODEL_THINK_DEFAULT` (unset by default; `true`/`false` otherwise -
  OllamaEngine-only, see below)

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
./backend model local
./backend model current
./backend model use TinyLlama/TinyLlama-1.1B-Chat-v1.0
./backend model info TinyLlama/TinyLlama-1.1B-Chat-v1.0
./backend model load TinyLlama/TinyLlama-1.1B-Chat-v1.0
./backend model unload
./backend model switch TinyLlama/TinyLlama-1.1B-Chat-v1.0
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
`loading`, `unloading`, `switching`, `unloaded`, or `degraded`. `status` also
prints the runtime `Loaded model` when it differs from the configured
`model.id`, and the `Target model` during a transition.

`backend model load`, `backend model unload`, and `backend model switch` call
management endpoints under `/admin/model/` on the running backend and perform
real transitions:

```bash
./backend model switch qwen3:1.7b            # runtime only; restart reverts it
./backend model switch qwen3:1.7b --persist  # also rewrites config.yaml
./backend model unload --timeout 30
./backend model load qwen3:30b --json
```

The target must be listed in `model.available` in `config/config.yaml`;
anything else is rejected before the engine is touched. `--persist` is the
difference between `model switch` (runtime, live, reverts on restart) and
`model use` (config, needs a restart). `--wait` and `--poll-interval` are not
implemented: transitions are synchronous, so there is nothing to poll.

This works only for engines that can do it safely. `OllamaEngine` supports it
because the Ollama daemon owns the CUDA context. `TransformersEngine` refuses
with `501`: its CUDA state is owned in-process, where cleanup is best-effort
and a swap can fragment VRAM enough to break later loads — change `model.id`
and restart instead. See `docs/model-lifecycle-design.md`.

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

`config/config.yaml`'s `model.available` catalog is static and human-maintained
— nothing keeps it in sync with what's actually downloaded. `backend model
list|current|info` show a `Cached locally: yes (<size>) / no` line per entry
(TransformersEngine only, via a read-only Hugging Face cache scan); `backend
model local` lists everything actually cached on disk, including models not
in the catalog. This is discovery only — it never edits `config.yaml` or
downloads/deletes anything.

Examples:

```bash
./backend model list
./backend model local
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

Both `backend status` and `backend gpu current` also show "Other GPU usage":
a warning if `backend.gpu`'s configured index(es) already have significant
memory used by another process. `InferenceService` runs this same check
once at startup and logs a warning if triggered - useful on a shared box
where another user's job might already occupy the GPU this backend is
about to use (see `docs/problems.md`). There's no per-process attribution
(`nvidia-smi`'s basic query doesn't provide it).

`backend status` and `backend gpu list` additionally print a **GPU
availability** census of every card on the box:

```text
GPU availability: 2 of 4 GPU(s) in use by other processes, 2 free
  GPU 0 ('RTX A4000'): IN USE - 6658 MiB / 16117 MiB, 80%
  GPU 1 ('RTX A4000'): IN USE - 6658 MiB / 16117 MiB, 87%
  GPU 2 ('RTX A4000'): free - 3 MiB / 16117 MiB
  GPU 3 ('RTX A4000'): free - 3 MiB / 16117 MiB
```

A GPU counts as in use if **either** its memory (>500 MiB) or its
utilization (>10%) is above threshold - memory alone misses a
compute-heavy job with a small resident footprint. Which indexes are busy
is always determined live; nothing assumes a fixed "safe" pair.

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

Before actually starting, `./backend start` checks whether `backend.gpu`'s
configured index(es) already show significant usage from another process
(see the "Other GPU usage" note under GPU Management below). If so:

- an idle GPU exists elsewhere on the box: **refuses to start** and exits
  non-zero, telling you which GPU(s) are idle so you can update
  `backend.gpu` and retry - or rerun with `--force` to start on the busy
  GPU anyway.
- no idle GPU exists anywhere: **asks for interactive confirmation**
  ("Continue starting on the busy GPU(s) anyway?") instead of refusing
  outright, since there's no alternative to suggest.
- `--force` / `-f` skips the refusal and the prompt, but is not silent: it
  names the GPU(s) being overridden and warns that another user's job may
  be disrupted.

**External model runtimes.** With `engine: ollama` the backend never loads
a model itself - the Ollama daemon does, using whatever devices *it* was
launched with. `backend.gpu` constrains only this backend's process, so it
says nothing about where Ollama will place weights. `./backend start`
therefore also reads the daemon's own `CUDA_VISIBLE_DEVICES` and refuses
to start if the daemon can reach a GPU someone else is using:

```text
WARNING: the Ollama daemon (PID 23823) can reach GPU(s) that another process is already using:
  GPU 0 ('RTX A4000'): 6658 MiB / 16117 MiB, 80%
  backend.gpu (2,3) does NOT constrain the daemon - it places models itself.
  Restart the daemon pinned to the free GPU(s):
    CUDA_VISIBLE_DEVICES=2,3 ollama serve
Refusing to start while the model runtime can reach a busy GPU.
```

The backend reports and refuses rather than restarting the daemon: it does
not own the daemon's lifecycle, and stopping it would drop other work.
Pinning the daemon is an operator action.

`./backend restart` passes `--force` through to the `start` step the same
way.

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

## Lifecycle Management Endpoints

`/admin/model/*` are management endpoints, separate from the OpenAI-compatible
`/v1/*` namespace, and carry no API stability guarantee:

```bash
curl -i -X POST http://127.0.0.1:8000/admin/model/switch \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "qwen3:1.7b", "persist": false}'
curl -i -X POST http://127.0.0.1:8000/admin/model/unload
curl -i -X POST http://127.0.0.1:8000/admin/model/load \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "qwen3:30b"}'
```

Success body:

```json
{
  "status": "ok",
  "lifecycle_state": "ready",
  "loaded_model": "qwen3:1.7b",
  "previous_model": "qwen3:30b",
  "elapsed_seconds": 0.42,
  "persisted": false
}
```

Failures return `{"error": ..., "detail": ..., "lifecycle_state": ...}` with
`404` (model not in `model.available`), `409` (illegal from the current state,
e.g. loading a different model while one is ready), `501` (the active engine
does not support runtime lifecycle), or `503` (engine unreachable, or the
target Ollama tag is not pulled).

While a transition is in progress, `/v1/chat/completions` and `/generate`
return `503`. Requests are rejected, never queued. In-flight requests are
drained first.

See `docs/model-lifecycle-design.md` for the full design and
`openapi/backend-node.openapi.yaml` for the authoritative contract.
