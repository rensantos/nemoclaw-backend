# Phase 5 Model Lifecycle Design

This is a design document only. It does not implement model lifecycle commands.

Phase 5 adds operational lifecycle commands:

```bash
./backend model load <model_id>
./backend model unload
./backend model switch <model_id>
```

Nemoclaw Backend remains the unified inference management layer. Nemoclaw Core
continues to own agents, memory, planning, skills, RAG, research workflows, and
orchestration. Core should call backend lifecycle APIs or CLI commands; it
should not duplicate model loading, model listing, benchmarking, provider
selection, or GPU/runtime logic.

## Recommendation

Use a supervised inference worker restart for `load`, `unload`, and `switch`.

Do not use repeated in-process Transformers load/unload cycles as the primary
production lifecycle mechanism.

Reason:

- Transformers models own Python objects, CUDA tensors, kernels, allocator
  state, and sometimes library-level caches.
- `del model`, `gc.collect()`, and `torch.cuda.empty_cache()` can reduce visible
  allocations but do not reliably return all VRAM to the driver.
- Repeated in-process model swaps can fragment GPU memory and make later loads
  fail even when total free memory appears sufficient.
- A worker process restart is the clearest boundary for releasing CUDA context
  state and recovering from partial load failures.

In-process engine methods still exist in the engine contract for engine
agnosticism and testability, but the recommended Transformers implementation
uses them inside a worker lifecycle, not as the long-running server's primary
swap strategy.

Comparison:

| Approach | Strengths | Risks | Decision |
| --- | --- | --- | --- |
| In-process swapping | Fewer processes, simpler first prototype, can work for small CPU-only models | Unreliable CUDA cleanup, VRAM fragmentation, harder rollback, half-loaded state can poison the serving process | Not recommended for production Transformers lifecycle |
| Supervised worker restart | Strong cleanup boundary, clearer rollback, API process can report transition status, works across future engines | Requires worker supervision and readiness checks | Recommended Phase 5 direction |

## Ownership

`InferenceService` is the single source of truth for loaded runtime model state.

It owns:

- current lifecycle state
- loaded model id
- loading/unloading/switching transition state
- active request accounting
- delegating runtime operations to `InferenceEngine`
- health details for runtime readiness

`ModelManager` owns configuration and metadata only:

- configured models
- selected/default model
- model validation
- model metadata
- YAML selection updates

`ModelManager` does not know whether a model is loaded. `InferenceService`
asks `ModelManager` to validate and describe a requested model, then performs
runtime lifecycle work through the selected engine or worker supervisor.

## Lifecycle States

Use explicit runtime states:

- `unloaded`
- `loading`
- `ready`
- `unloading`
- `switching`
- `degraded`

`ready` means a model is loaded and chat requests can be served.

`degraded` means lifecycle recovery failed and the backend can report its state,
but inference may not be available. This should be rare with the recommended
worker restart strategy.

### State Transition Table

| From | May move to |
| --- | --- |
| `unloaded` | `loading` |
| `loading` | `ready`, `degraded` |
| `ready` | `unloading`, `switching` |
| `unloading` | `unloaded`, `degraded` |
| `switching` | `ready`, `degraded` |
| `degraded` | `loading`, `unloaded` |

This table is descriptive of the state machine the Minimal Implementation Plan
below will implement. Increment 1 (current) only introduces the state values
and reports them; it does not yet implement the transitions themselves. Every
runtime `InferenceService` currently reports a fixed `ready` state, matching
the existing startup-load-then-serve behavior. The code's state values (see
`services/lifecycle.py`) must not diverge from this table.

## Runtime Architecture

Target lifecycle architecture:

```text
CLI / future lifecycle HTTP endpoint
  -> ModelManager validates configured model
  -> InferenceService owns runtime state
  -> InferenceWorker supervisor
  -> InferenceEngine
  -> TransformersEngine / future engines
```

The current FastAPI process can remain the management/API process. The model
runtime should move into a supervised worker process when lifecycle commands are
implemented. For the first safe increment, the existing Uvicorn process may be
restarted as a whole by the CLI, but the design target is a managed inference
worker so the API can keep reporting transition status.

## Engine Contract

`InferenceEngine` should define lifecycle methods:

```python
load_model(model_id: str, model_info: dict) -> None
unload_model() -> None
switch_model(model_id: str, model_info: dict) -> None
health() -> dict
list_models() -> list
chat(messages, max_tokens, temperature) -> dict
```

Semantics:

- `load_model` loads exactly one model into the engine.
- `unload_model` releases the engine's loaded model if the engine owns runtime
  state.
- `switch_model` transitions from the current model to a requested model. The
  default implementation may be unload then load, but engines can override it.

Future engine differences:

- `TransformersEngine` owns Python and CUDA state. It should prefer worker
  process replacement for reliable cleanup.
- `OllamaEngine` would delegate lifecycle to an Ollama daemon. Backend state
  tracks requested and observed daemon state, but CUDA cleanup is daemon-owned.
- `VLLMEngine` would delegate to a vLLM worker/server process. Backend should
  supervise that process rather than embedding vLLM internals in the API layer.
- `LlamaCppEngine` may own CPU/GPU memory in-process or through a worker,
  depending on deployment mode.
- `OpenAICompatibleEngine` usually does not own model memory; lifecycle may be
  validation or remote provider selection only.

Do not implement future engines in Phase 5.

## Concurrency Model

Lifecycle transitions must be explicit and predictable.

### `model load <model_id>`

When current state is `unloaded`:

- New chat requests during `loading`: reject with HTTP `503 Service Unavailable`.
- Response body should keep the OpenAI-compatible error shape if an error
  schema exists by then; otherwise use the current FastAPI error style.
- `/health` reports `status: loading`, `ready: false`, `model: <model_id>`.
- `/v1/models` continues to report configured models and may include the
  selected/default model. It must not falsely claim the loading model is ready.

When current state is `ready`:

- `load <same_model_id>` is idempotent and returns success.
- `load <different_model_id>` should fail with a clear message directing the
  operator to use `model switch <model_id>`.

No in-flight inference requests exist in `unloaded`. If a load starts while a
request arrives, the request is rejected with `503`; it is not queued.

### `model unload`

When current state is `ready`:

- Existing in-flight requests are drained until completion or a configured
  timeout.
- New chat requests during `unloading`: reject with HTTP `503 Service
  Unavailable`.
- `/health` reports `status: unloading`, `ready: false`, and the model being
  unloaded.
- `/v1/models` still reports configured models; loaded runtime state is reported
  through health/status, not by removing configured models.

After drain timeout:

- The backend should stop accepting more work and restart the worker anyway.
- In-flight requests that exceed the timeout may receive `503` or connection
  termination depending on worker supervision details. Prefer a controlled
  `503` from the management process where possible.

When current state is `unloaded`, `unload` is idempotent and returns success.

### `model switch <model_id>`

Switch is a single transition from one ready model to another.

- Existing in-flight requests for the old model are drained until completion or
  timeout.
- New chat requests during `switching`: reject with HTTP `503 Service
  Unavailable`.
- Requests are not queued. Queuing hides latency spikes and can produce
  surprising responses from either the old or new model.
- `/health` reports `status: switching`, `ready: false`, `from_model`, and
  `to_model`.
- `/v1/models` reports configured models. It should not represent the target
  model as loaded until the worker is ready.

After the new worker is healthy:

- Runtime state becomes `ready`.
- `loaded_model_id` becomes the target model id.
- New requests are served by the new model.

## API Compatibility

No breaking changes to `/v1/*` endpoints.

Existing endpoints remain:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

`/v1/chat/completions` keeps the current OpenAI-compatible success shape. During
lifecycle transitions, new inference requests may receive HTTP `503` with a
clear error message. This is compatible with operational unavailability and
does not change successful response schemas.

`/v1/models` continues to describe configured/available models. Loaded runtime
state should be exposed through `/health`, CLI status, or a future management
endpoint, not by changing the meaning of OpenAI model listing.

## CLI Behavior

Lifecycle commands should provide timeout and progress feedback:

```bash
./backend model load <model_id> --timeout 600 --wait
./backend model unload --timeout 120 --wait
./backend model switch <model_id> --timeout 600 --wait
```

Suggested options:

- `--timeout <seconds>`: maximum time to wait for the lifecycle transition.
- `--wait/--no-wait`: wait for readiness or return after transition starts.
- `--poll-interval <seconds>`: progress refresh interval.
- `--json`: structured output for automation.

Progress output for slow loads:

```text
Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
State: loading
Elapsed: 45s
Health: loading
GPU: 0
VRAM used: 6210 MiB
```

CLI should report:

- requested model id
- previous loaded model id, if any
- lifecycle state
- elapsed time
- health
- worker PID or process state when available
- GPU and VRAM snapshot when available
- final result or failure reason

The CLI should delegate lifecycle work to backend services or lifecycle API
calls. It should not perform model loading, CUDA cleanup, or engine-specific
logic directly.

## Failure Modes And Rollback

Failed load must never leave a half-loaded runtime.

Recommended switch rollback:

1. Keep the old worker serving while the new worker loads when enough resources
   allow side-by-side load.
2. Health-check the new worker.
3. Atomically route new requests to the new worker.
4. Drain and stop the old worker.

If side-by-side load is not possible because of VRAM limits:

1. Drain the old worker.
2. Stop the old worker and release the process.
3. Start the new worker.
4. If the new worker fails, attempt to restart the old worker.
5. If old-worker rollback also fails, report `degraded` clearly.

This means `switch` has two modes:

- `safe-side-by-side` when resources permit; previous model continues serving
  until the new model is ready.
- `single-gpu-restart` when resources do not permit; service enters
  `switching` and rejects new inference requests until success or rollback.

The first implementation on the RTX A4000 should assume `single-gpu-restart`
unless GPU memory checks prove side-by-side loading is safe.

Failure examples:

- model id is not configured: reject before any runtime change.
- model files cannot be downloaded or opened: previous model keeps serving when
  side-by-side mode is used; otherwise rollback attempts to restart previous
  worker.
- CUDA out of memory: report failure, include GPU snapshot, rollback according
  to mode.
- worker never becomes healthy: stop failed worker and rollback or report
  degraded.
- unload cleanup incomplete: prefer process termination over relying on
  in-process CUDA cleanup.

## CUDA Cleanup

For `TransformersEngine`, cleanup inside one process is best effort only:

```python
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()
```

This sequence is useful for tests and emergency cleanup but is not a reliable
production lifecycle boundary. It may leave CUDA context allocations, allocator
fragmentation, kernels, or cached memory behind.

Production cleanup should terminate the inference worker process. The operating
system and NVIDIA driver then reclaim the process-owned CUDA context. After
termination, `GPUManager` should sample VRAM so CLI/API status can report the
observed result.

## Streaming Assumptions

Streaming is not implemented yet.

This design assumes each active request is a bounded non-streaming request that
can be counted, drained, or rejected. A future streaming phase must integrate
with the same active request accounting:

- active streams count as active requests.
- lifecycle transitions stop accepting new streams.
- existing streams are drained until completion or timeout.
- timeout closes remaining streams with a controlled error if possible.

The lifecycle state machine should be implemented before streaming so streaming
does not create a second request-drain model.

## Minimal Implementation Plan

1. Add runtime lifecycle state to `InferenceService`.
   - Track state, loaded model id, target model id, transition start time, and
     active request count.
   - Reject new chat requests with `503` when state is not `ready`.

2. Extend `InferenceEngine` contract.
   - Add explicit lifecycle method signatures.
   - Keep `TransformersEngine` behavior compatible with current startup load.

3. Add management methods to `InferenceService`.
   - `load_model(model_id)`
   - `unload_model()`
   - `switch_model(model_id)`
   - validate model ids through `ModelManager`.

4. Implement the first safe lifecycle path as process restart.
   - Start with CLI-managed full backend restart if needed.
   - Move toward a supervised inference worker while keeping FastAPI available
     for status reporting during transitions.

5. Add CLI commands.
   - `backend model load <model_id>`
   - `backend model unload`
   - `backend model switch <model_id>`
   - include `--timeout`, `--wait/--no-wait`, `--poll-interval`, and `--json`.

6. Add health/status reporting.
   - `/health` reports lifecycle state and loaded/target model.
   - Existing success schemas remain unchanged.

7. Add worker supervision.
   - Worker start, stop, readiness polling, timeout, and rollback.
   - Process termination is the cleanup boundary for Transformers.

8. Add optional side-by-side switch only after GPU checks and tests prove it is
   safe for target deployments.

## Required Tests

Unit tests:

- `ModelManager` rejects unknown lifecycle target model ids.
- `InferenceService` state transitions: `unloaded -> loading -> ready`.
- `InferenceService` state transitions: `ready -> unloading -> unloaded`.
- `InferenceService` state transitions: `ready -> switching -> ready`.
- New chat requests return `503` during `loading`, `unloading`, and
  `switching`.
- In-flight request accounting drains before unload/switch.
- Failed load in side-by-side mode leaves previous model serving.
- Failed load in single-GPU restart mode attempts rollback and reports
  `degraded` if rollback fails.
- `/health` reports `loading`, `unloading`, `switching`, `ready`, and
  `degraded` correctly.
- `/v1/models` remains compatible and does not become a loaded-state endpoint.
- CLI commands pass timeout, wait, poll interval, and JSON options to the
  lifecycle service/API.

Engine tests:

- `TransformersEngine.unload_model()` calls best-effort cleanup hooks.
- Worker supervisor terminates the worker process on unload.
- VRAM after unload is sampled through `GPUManager`.
- VRAM after unload test: mock or integration-test that used VRAM after worker
  termination is less than or equal to used VRAM before unload, allowing a
  tolerance for unrelated system activity.

Integration tests on UBI:

- Load configured TinyLlama model and verify `/health` becomes `ready`.
- Run chat completion after load.
- Unload and verify chat completion receives `503`.
- Switch to a configured test model and verify new model id in health/status.
- Simulate load failure and verify previous model remains serving or degraded
  state is explicit.
- Measure VRAM before load, after load, and after unload using `GPUManager`.

## Non-Goals

Phase 5 does not add:

- agents
- memory
- RAG
- planning
- research workflows
- orchestration
- Core logic
- Ollama, vLLM, llama.cpp, or OpenAI-compatible engine implementations
- benchmarking changes
- monitoring dashboards
