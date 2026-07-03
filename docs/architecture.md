# Architecture

Nemoclaw Backend is an independent OpenAI-compatible inference server. It does
not contain Nemoclaw Core orchestration, memory, agents, RAG, Telegram, or
research logic.

## Layers

```text
FastAPI
  -> InferenceService
    -> InferenceEngine
      -> TransformersEngine
        -> CUDA / GPU

CLI
  -> GPUManager
    -> nvidia-smi / torch.cuda
```

## API Layer

`api.py` owns HTTP routing and OpenAI-compatible response formatting.

It does not know about tokenizer creation, model loading, CUDA, or Hugging Face
Transformers internals. It validates request-level concerns, such as unsupported
streaming, then delegates inference work to `InferenceService`.

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

## Engine Interface

`engines/base.py` defines `InferenceEngine`, the minimal engine contract:

- `load_model`
- `unload_model`
- `health`
- `list_models`
- `chat`
- `generate_text`

The interface is intentionally small. Future engines such as `VLLMEngine`,
`LlamaCppEngine`, `TensorRTEngine`, or `ONNXEngine` should implement this
contract without requiring changes to `api.py`.

## Transformers Engine

`engines/transformers_engine.py` contains all Hugging Face Transformers,
tokenizer, PyTorch, and CUDA-specific logic.

It preserves the current deployment behavior:

- default model from `config/config.yaml`
- `torch_dtype=torch.float16`
- `device_map="auto"`
- `import torch.fx` for compatibility with old PyTorch 1.12.1
- OpenAI-compatible chat token usage accounting

## Compatibility

`model_runtime.py` remains as a thin compatibility facade over the inference
service for any old local imports. New code should use the service layer.

## Model Configuration

Phase 2 model management is configuration-level only.

- Configured models live in `config/config.yaml` under `model.available`.
- The selected/default model is `model.id`.
- The loaded model is owned by the running backend process and is chosen at
  startup.

`backend model use <model_id>` changes YAML selection only. Runtime model
switching, model lifecycle management, GPU management, benchmarking, monitoring,
RAG, agents, and orchestration are outside this phase.

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
affinity, benchmarking, and dashboards are future work.
