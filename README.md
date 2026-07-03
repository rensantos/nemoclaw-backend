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

## Start

```bash
./scripts/start.sh
```

Equivalent manual command:

```bash
source ~/miniforge3/bin/activate
conda activate llm
python server.py
```

## Stop and Status

```bash
./scripts/status.sh
./scripts/stop.sh
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
