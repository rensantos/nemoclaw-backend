# Known Problems

## Runtime Not Fully Verified In Sandbox

The code has passed syntax checks and config-loading checks in the current
workspace, but the live server has not been fully exercised here because this
sandbox does not expose the expected `~/miniforge3` Conda installation.

Before treating a deployment as healthy, run:

```bash
./scripts/start.sh
./scripts/status.sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

And test chat completion:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Say hello from Nemoclaw in one sentence."}
    ],
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

## PyYAML Is Now Required

`config.py` uses PyYAML. The `llm` Conda environment must include `yaml`
import support:

```bash
python -c "import yaml; print(yaml.__version__)"
```

## Typer Is Required For CLI Operations

The `backend` command uses Typer. The `llm` Conda environment must include
`typer` import support:

```bash
python -c "import typer; print(typer.__version__)"
```

## Streaming Is Not Implemented

`stream: true` requests return a `400` response. The endpoint accepts the field
for OpenAI-style request compatibility, but streaming output is future work.

## Benchmarks Require A Running Backend

`backend benchmark ...` commands call the local OpenAI-compatible HTTP endpoint.
They can be unit-tested without a GPU, but live benchmark numbers require the
backend to be running with a model loaded:

```bash
./backend start
./backend benchmark latency
./backend benchmark throughput
./backend benchmark vram
```
