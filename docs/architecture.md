# Architecture

Nemoclaw Backend is the reusable, unified inference management backend for
Nemoclaw. It is not only a Transformers server; the current Transformers
runtime is the first engine implementation behind a backend-owned inference
API.

The backend owns LLM and inference functionality:

- model providers and inference engines
- model selection and model metadata
- OpenAI-compatible inference API
- benchmarking
- GPU and runtime inspection
- future Ollama, vLLM, llama.cpp, and OpenAI-compatible engine support

Nemoclaw Core owns higher-level intelligence and workflow behavior:

- agents
- memory
- planning
- skills
- RAG
- research workflows
- orchestration

Core should call the backend API for inference, model listing, model selection
state, benchmarking, and runtime inspection. It should not duplicate model
provider, model-listing, benchmarking, or GPU/runtime logic.

## Layers

```text
Nemoclaw Core
  -> Nemoclaw Backend API
    -> InferenceService
      -> InferenceEngine
        -> TransformersEngine
        -> future OllamaEngine
        -> future VLLMEngine
        -> future LlamaCppEngine
        -> future OpenAICompatibleEngine

FastAPI
  -> InferenceService
    -> InferenceEngine
      -> TransformersEngine
        -> CUDA / GPU

CLI
  -> ModelManager
    -> Configuration

CLI
  -> GPUManager
    -> nvidia-smi / torch.cuda

CLI
  -> BenchmarkService
    -> OpenAI-compatible backend HTTP API
    -> ModelManager / GPUManager
```

## API Layer

`api.py` owns HTTP routing and OpenAI-compatible response formatting.

It does not know about tokenizer creation, model loading, CUDA, or Hugging Face
Transformers internals. It validates request-level concerns, such as unsupported
streaming, then delegates inference work to `InferenceService`.

This API is the boundary Nemoclaw Core should use. Core should not import
backend internals or reimplement backend-owned capabilities such as model
listing, benchmarking, provider selection, or GPU inspection.

## Inference Service

`services/inference.py` contains `InferenceService`, the application boundary
between FastAPI and inference engines.

The service exposes only the operations currently needed by the backend:

- `health`
- `list_models`
- `chat`
- `generate_text` for the existing compatibility endpoint

The service delegates to one configured engine. It does not implement model
management, routing, benchmarking, monitoring, or multi-model behavior.

As the backend grows, `InferenceService` remains the stable boundary between
the HTTP API and engine implementations. New providers belong behind this
service through `InferenceEngine` implementations, not in Nemoclaw Core.

## Engine Interface

`engines/base.py` defines `InferenceEngine`, the minimal engine contract:

- `load_model`
- `unload_model`
- `health`
- `list_models`
- `chat`
- `generate_text`

The interface is intentionally small. Future engines such as `OllamaEngine`,
`VLLMEngine`, `LlamaCppEngine`, `OpenAICompatibleEngine`, `TensorRTEngine`, or
`ONNXEngine` should implement this contract without requiring changes to
`api.py` or Nemoclaw Core.

Future provider support belongs in Nemoclaw Backend. Core should select or ask
for configured backend capabilities through the backend API rather than
embedding provider-specific clients or model catalogs.

## Transformers Engine

`engines/transformers_engine.py` contains all Hugging Face Transformers,
tokenizer, PyTorch, and CUDA-specific logic.

It preserves the current deployment behavior:

- default model from `config/config.yaml`
- `torch_dtype=torch.float16`
- `device_map="auto"`
- `import torch.fx` for compatibility with old PyTorch 1.12.1
- OpenAI-compatible chat token usage accounting

Transformers is the current engine, not the permanent boundary of the backend.
Do not add future provider logic to Core just because the current backend engine
is Transformers.

## Compatibility

`model_runtime.py` remains as a thin compatibility facade over the inference
service for any old local imports. New code should use the service layer.

## Model Configuration

Phase 2 model management is configuration-level only.

- Configured models live in `config/config.yaml` under `model.available`.
- The selected/default model is `model.id`.
- The loaded model is owned by the running backend process and is chosen at
  startup.

`services/model.py` contains `ModelManager`, the single service responsible for:

- configured models
- selected/default model
- model validation
- model metadata
- selected-model configuration updates

`config.py` is a configuration provider. Model business rules belong in
`ModelManager`.

`backend model use <model_id>` calls `ModelManager` and changes YAML selection
only. Runtime model switching, model lifecycle management, GPU management,
benchmarking, monitoring, RAG, agents, and orchestration are outside this phase.

`ModelManager` does not know about Transformers, CUDA, tokenizers, or loaded
runtime models. `InferenceService` remains responsible for runtime inference.

## GPU Management

`services/gpu.py` contains `GPUManager`, the single service responsible for GPU
discovery and status reporting.

The CLI calls `GPUManager` for:

- GPU list
- selected/current backend GPU
- VRAM usage
- utilization
- temperature
- CUDA availability
- driver information

The implementation is intentionally lightweight and uses `nvidia-smi` plus
optional `torch.cuda` checks. It does not introduce NVML bindings or monitoring
frameworks.

Phase 3 is informational only. GPU selection, multi-GPU scheduling, MIG, CUDA
affinity, and dashboards are future work.

## Benchmarking

`services/benchmark.py` contains `BenchmarkService`, the single service
responsible for benchmark execution and result formatting.

The CLI calls `BenchmarkService` for:

- latency
- throughput
- VRAM before/peak/after
- first-token latency availability

Benchmarks call the local OpenAI-compatible endpoint exactly as a client would:

```text
BenchmarkService
  -> http://HOST:PORT/v1/chat/completions
```

`BenchmarkService` may read model metadata from `ModelManager` and GPU state
from `GPUManager`, but it does not call Transformers directly. This keeps
benchmarking reusable for future monitoring and automation without coupling it
to a specific inference engine.

Phase 4 does not implement Prometheus, Grafana, dashboards, continuous
monitoring, distributed benchmarking, or a load-testing framework. Concurrency
is accepted in command options for API stability, but requests are still run
sequentially. First-token latency remains unavailable until streaming exists.
