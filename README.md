# Nemoclaw Backend

Small OpenAI-compatible FastAPI backend for serving one local Hugging Face
Transformers causal language model.

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
```

`backend status` prints the active model, GPU, host, port, health, VRAM, and
temperature. `backend health` calls `/health`. `backend config` prints the
active configuration after YAML and environment overrides. `backend logs` shows
`logs/backend.log`, and `backend logs --follow` tails it continuously.

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
