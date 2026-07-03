# Nemoclaw Backend

Small OpenAI-compatible FastAPI backend for serving one local Hugging Face
Transformers causal language model.

## Environment

Defaults are intentionally conservative:

- `MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `GPU=0`
- `HOST=127.0.0.1`
- `PORT=8000`
- `MAX_TOKENS_DEFAULT=256`
- `TEMPERATURE_DEFAULT=0.7`

## Start

```bash
./scripts/start.sh
```

Equivalent manual command:

```bash
source ~/miniforge3/bin/activate
conda activate llm
CUDA_VISIBLE_DEVICES="${GPU:-0}" uvicorn server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
```

Override model or GPU:

```bash
MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0 GPU=0 ./scripts/start.sh
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
