# Future Tasks

## Product Direction

- Treat Nemoclaw Backend as the reusable inference management backend, not only
  as a Transformers server.
- Keep Nemoclaw Core focused on agents, memory, planning, skills, RAG, research
  workflows, and orchestration.
- Keep LLM provider support, engine integration, model listing, model
  selection, benchmarking, and GPU/runtime inspection inside Nemoclaw Backend.
- Do not duplicate backend-owned model catalogs, benchmark commands, provider
  clients, or GPU/runtime inspection logic in Nemoclaw Core.

## Operational Follow-up

- Verify `./scripts/start.sh` on the UBI machine inside the `llm` Conda env.
- Verify `/health`, `/v1/models`, and `/v1/chat/completions` with the configured
  model loaded on the RTX A4000.
- Verify the Typer CLI on the UBI machine, especially `backend start`,
  `backend status`, `backend health`, and `backend logs`.
- Verify `backend status` against the existing development launcher and the
  future systemd service once Phase 8 exists.
- Decide when to remove the temporary shell wrappers after CLI usage settles.
- Verify `backend model use` on the UBI machine and restart the backend to
  confirm the selected model is loaded at process start.
- Verify `backend gpu list`, `backend gpu current`, and `backend gpu monitor`
  on the UBI machine with the RTX A4000.
- Verify `backend benchmark latency`, `backend benchmark throughput`, and
  `backend benchmark vram` against the running UBI backend after the model is
  loaded.

## API Follow-up

- Add clearer validation for unsupported request fields if Nemoclaw clients
  start sending more OpenAI parameters.
- Consider implementing streaming later if the client needs token-by-token
  responses.
- Add response timing metadata only if it remains outside the OpenAI-compatible
  response body or is explicitly accepted by clients.
- Add future engines behind `InferenceEngine` only when a phase explicitly calls
  for them. Do not change `api.py` or Nemoclaw Core for engine-specific work.
- Future `OllamaEngine`, `VLLMEngine`, `LlamaCppEngine`, and
  `OpenAICompatibleEngine` support belongs inside Nemoclaw Backend.
- Do not add Ollama, vLLM, llama.cpp, OpenAI-compatible provider clients, or any
  other new engine until an explicit implementation phase asks for it.
- The `/admin/model/load|unload|switch` endpoint surface and the matching
  `backend model load|unload|switch` CLI commands exist (Phase 5 Increment 2)
  but are stubs: every call returns HTTP `501` (or `422` for `load`/`switch`
  if the request body fails validation) and never changes runtime state.
  Real lifecycle behavior — actual load/unload/switch, worker supervision,
  and CUDA cleanup — remains future work per
  `docs/model-lifecycle-design.md` (Increment 3+).
- Future `backend model load`, `backend model unload`, and `backend model switch`
  behavior should build on `ModelManager` without moving inference logic into it.
- Implement real concurrent benchmark execution when needed. Phase 4 accepts
  `--concurrency` but still runs requests sequentially.
- Implement first-token latency only after streaming responses exist.
- GPU selection, multi-GPU scheduling, MIG support, CUDA affinity, and
  monitoring dashboards remain future work.

## OllamaEngine Implementation (per docs/ollama-engine-design.md)

- Increment 1 (done): config (`backend.engine`, default `transformers`,
  `ENGINE` env override, fail-fast on invalid values) + `services/inference.
  _build_engine()` factory called from `create_inference_service()` +
  `engines/ollama_engine.py` skeleton (`load_model()` is a safe no-op so
  startup doesn't crash; every other method raises `NotImplementedError`).
  Unit tests: `tests/test_config.py` (precedence, invalid-value fail-fast),
  `tests/test_engine_factory.py` (factory selection, stub construction and
  behavior). `engine: transformers` (default) behavior is unchanged.
  Operator validation still needed on UBI (Step 7): `./backend restart &&
  ./backend status` and a live chat completion should be unchanged;
  `ENGINE=ollama` startup should construct cleanly and fail requests with
  `NotImplementedError`-derived errors, not hang or crash the process.
- Increment 2 (next): `OllamaEngine` read paths — `health()`, `list_models()`,
  `load_model()` (tag-presence validation, no pulling). Unit tests with
  mocked Ollama HTTP responses; live validation on the Ollama-hosting Local
  Node (`docs/architecture.md`'s Target deployment topology — not UBI,
  which runs `TransformersEngine`) with a small pulled model.
- Increment 3: `OllamaEngine` `chat()` / `generate_text()`, including the
  model-resolution decision (404 `model_not_found` on mismatch,
  `EngineUnavailableError` -> `503` on daemon-down) and token-usage mapping
  (`prompt_eval_count`/`eval_count`, with the documented `0`-fallback and
  warning log when counts are missing). Unit tests with mocked responses;
  live validation on the Local Node via `curl /v1/chat/completions`.
- Increment 4: `OllamaEngine.unload_model()` (`keep_alive: 0` mapping),
  tested as an engine method only — not wired to any live endpoint.
- Increment 5 (separate, code-adjacent): the `openapi/backend-node.openapi.yaml`
  amendments this design requires (new `model_not_found` error schema,
  optional `requested_model` field, `/health` status-value widening for
  daemon-down reporting) must land in the same increment as the runtime
  behavior that needs them, not bundled into Increments 1-4.
- Apply the model-resolution decision (`docs/ollama-engine-design.md`
  Section 1) to the existing `TransformersEngine`/`api.py`
  `/v1/chat/completions` path. Deliberately deferred out of the
  `OllamaEngine` increments — this is a separate future task so that
  closing a documented drift item on the existing engine doesn't expand an
  engine-integration increment into a behavior change for current callers.
- Operator prerequisite: install Ollama on the Local Node (not UBI, which
  runs `TransformersEngine` only — see `docs/architecture.md`'s Target
  deployment topology) before Increment 2's live validation; verify with
  `ollama --version`, `ollama list`, and `curl
  http://127.0.0.1:11434/api/tags` on that node. Verify OS/glibc
  compatibility with the current Ollama release for the Local Node's OS
  before installing (see Risks in `docs/ollama-engine-design.md` — the
  Ubuntu 18/glibc note there is marked not applicable to UBI now that
  Ollama runs on the Local Node instead).
- Backend Registry (`docs/registration-schema.json`) is deferred until a
  real second Backend Node exists (e.g. the Local Node). At Registry
  design time, the registration schema may need amendment — e.g.
  advertising the enabled engine and hardware traits per node;
  `docs/registration-schema.json` remains authoritative and unchanged
  until that phase.

## OpenAICompatibleEngine / Remote API Node (future)

- Per the target deployment topology (`docs/architecture.md`), the Remote
  API Node runs `OpenAICompatibleEngine`, adapting remote OpenAI-compatible
  services (OpenAI, Gemini, future compatible providers) into the Backend
  contract so Core sees a Backend Node, not individual providers.
- API keys must never be stored in `config/config.yaml` (committed to
  git). Keys come from environment variables or an untracked secrets
  file, consistent with the existing env-override pattern. Engine phase
  contract applies: `__init__` side-effect free; `load_model()` =
  lightweight key/endpoint validation; `health()` = API reachability.

## Core / application integration

- All Nemoclaw applications (existing Research Assistant, future synthetic
  data generation, assistants, Telegram interface, CLI tools) consume the
  backend exclusively through the OpenAI-compatible surface. No
  per-application endpoints in the backend.
- Validate backend interoperability by migrating the existing Research
  Assistant without modifying its reasoning logic — only its model access
  layer (expected: base_url/config change on an OpenAI-style client). If the
  migration requires rewriting any RAG pipeline logic, treat that as
  evidence the Backend API is missing something; report the gap against the
  pinned contract rather than working around it.
- Remove any remaining direct model-loading or direct runtime calls from
  migrated applications, per: Core never talks directly to inference
  runtimes.
- Use the migrated Research Assistant as an end-to-end validation workload
  when OllamaEngine lands: same queries, switch engine in config, compare
  behavior.
- Research Assistant migration is an early contract-validation task after
  OllamaEngine if its current workflows do not require SSE streaming. If
  streaming is required for meaningful validation, defer migration until
  after the SSE streaming phase.

## Speculative / unscheduled

- Possible future engines: TensorRTEngine (NVIDIA-optimized inference),
  ONNXEngine (portable CPU/edge inference). Not on roadmap; behind
  `InferenceEngine` if ever added.

## Testing Follow-up

- Add lightweight unit tests for config precedence:
  environment variables over YAML over defaults.
- Add API tests with a mocked model runtime so endpoint response shapes can be
  checked without loading a GPU model.
- Add CLI integration tests in the `llm` Conda environment after Typer is
  installed there.
- Add a deployment smoke-test checklist for the UBI machine.
