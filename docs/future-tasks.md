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
  _build_engine()` factory called from `create_inference_service()`.
  Unit tests: `tests/test_config.py` (precedence, invalid-value fail-fast),
  `tests/test_engine_factory.py` (factory selection). `engine: transformers`
  (default) behavior is unchanged. UBI keeps running `TransformersEngine`
  only, per `docs/architecture.md`'s Target deployment topology — no
  operator validation of `engine: ollama` on UBI is expected or needed.
- Increment 2 (done): `OllamaEngine` real read paths — `health()`,
  `list_models()`, `load_model()` (tag-presence validation, no pulling),
  against `GET /api/tags`. New `backend.ollama_host` config
  (default `http://127.0.0.1:11434`, `OLLAMA_HOST` env override); new
  `engines.base.EngineUnavailableError`; `InferenceService.health()` now
  catches it and projects `HealthResponse.status` from `lifecycle_state`
  via `services/lifecycle.health_status_for_lifecycle_state()`
  (`ready -> ok`, `degraded -> degraded`, else `-> unavailable`), closing
  the gap flagged in `openapi/backend-node.openapi.yaml`'s `/health`
  description. `cuda`/`gpu` sourced from `GPUManager` (new
  `GPUManager.gpu_name()`), not `torch.cuda`. Unit tests with mocked
  `urllib` responses (`tests/test_engine_factory.py`); live-validated on a
  real Local Node (an operator's own machine running Ollama, per
  `docs/architecture.md`'s Target deployment topology): `./backend start`
  with `ENGINE=ollama`, `GET /health`, `GET /v1/models`, missing-tag
  startup failure, daemon-unreachable startup failure, and the
  `ready -> degraded` transition when the daemon goes down mid-session —
  all confirmed against real `ollama list` models, not just mocks.
- Increment 3 (done): `OllamaEngine` `chat()` / `generate_text()` against
  `POST /api/chat` / `POST /api/generate`, with a separate, generous fixed
  120s timeout (distinct from the read paths' 5s) — resolves the "Timeout
  behavior" question the design doc left open for this increment; full
  operator-configurability is still deferred. Model-resolution decision:
  `OllamaEngine.chat()` raises new `engines.base.ModelNotFoundError` on a
  mismatched requested `model`, before any daemon call; `api.py` maps it
  to `404 model_not_found`. `InferenceEngine.chat()` gained an optional
  `requested_model` parameter (`TransformersEngine` accepts and ignores
  it, keeping its existing echo-and-serve quirk exactly as before — out
  of scope per Section 1). `GenerateRequest` has no `model` field, so
  `/generate` has no equivalent check. `EngineUnavailableError` during
  either call now also flips `lifecycle_state` to `degraded` and becomes
  `503` via `api.py`. Token-usage mapping
  (`prompt_eval_count`/`eval_count`, `0`-fallback + warning log when
  missing) implemented as designed. Unit tests with mocked responses
  (`tests/test_engine_factory.py`, `tests/test_inference_service.py`);
  live-validated on a real Local Node against real Ollama models: full
  HTTP round trip through `./backend start` for a successful chat (real
  token counts), the `404` rejection on a mismatched model, and
  `/generate`. Not implemented: `requested_model` on the *response* —
  Section 1's optional additive field for "tolerated mismatch" cases has
  no code path that would ever populate it in the single-active-engine
  design, so it was skipped rather than added unused.
- Increment 4 (done): `OllamaEngine.unload_model()` posts
  `POST /api/generate` with `{"model": <configured tag>, "keep_alive":
  0}`. Best-effort: the backend does not stop or supervise the daemon
  process. Unit test with a mocked response asserting the exact request
  payload; live-validated against a real Ollama daemon — loaded the model
  via a real request, confirmed it in `GET /api/ps`, called
  `unload_model()`, confirmed `GET /api/ps` cleared it (with a few
  seconds' lag before the daemon's own eviction actually completed — not
  synchronous, matches the design's "best-effort" framing). Tested as an
  engine method only, not wired to any live endpoint: `/admin/model/unload`
  stays a `501` stub for every engine. This completes `OllamaEngine`'s
  implementation of every `InferenceEngine` method (Increments 1-4 all
  done); Increment 5 (openapi amendments) landed alongside Increment 3.
- Increment 5 (done, landed with Increment 3): the
  `openapi/backend-node.openapi.yaml` amendments this design requires.
  `/health` status-value widening had already landed ahead of Increment 2,
  in commits `c0ffe4b`/`208823a`. Increment 3 added: new
  `ModelNotFoundResponse` schema, `404`/`503` responses on
  `POST /v1/chat/completions`, `503` on `POST /generate`, and updated
  `x-current-behavior`/field descriptions to state that model-resolution
  behavior is now engine-dependent. Also fixed, while in the file: a
  pre-existing drift where `ModelObject.owned_by` was pinned `const:
  local`, which `OllamaEngine` had already violated (`"ollama"`) since
  Increment 2 shipped without this file being updated for it — now
  `type: string` with both values documented. The `requested_model`
  request/response field was deliberately not added (see Increment 3
  above).
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
  **Trigger condition now met** (2026-07-30): the Local Node (real Ollama,
  Increments 1-4) and the UBI Node are both live and validated — a real
  second Backend Node exists. Concrete motivating use case from the
  frontend side: the user wants their frontend (`nemoclaw-research-assistant`)
  to let the end user pick, per request, which node/model to use (Local
  Ollama vs UBI) instead of a static `.env`/`NEMOCLAW_LLM_PROVIDER` choice
  requiring a restart. This is exactly the "how does Core/a frontend
  discover and choose among Backend Nodes" question the Registry exists
  to answer. Not started — recorded here as a real trigger, not a design.
  **Constraint noted by the user (2026-07-30):** reaching the UBI Node
  today requires both an SSH connection to the UBI machine and valid UBI
  account credentials (the backend binds `127.0.0.1` only, no auth of its
  own — see the SSH tunnel note in `AGENTS.md`'s Current State). Any
  future automatic Local-vs-UBI selection is not a same-machine HTTP
  choice like Local-vs-Remote-API-Node would be; it needs to account for
  who already has UBI SSH access, not just which base_url to call.
  **First concrete design pass (2026-07-31), frontend side:** written up
  in `nemoclaw-research-assistant`'s `NEMOCLAW_SETUP.md` ("Future
  development: user-facing model/node choice") after actually standing
  up the UBI Node end to end and manually pointing the frontend at it.
  Findings that matter for this Registry's eventual schema: a Backend
  Node needs to advertise *which models it actually has* (UBI had only
  3 of the frontend's ~10 configured model roles pulled) and *safe
  per-node resource ceilings* (a context-window setting fine on one node
  made every request 500 on another, model-size-dependent). Also
  confirmed live: this repo's OpenAI-compatible surface has no
  embeddings endpoint, which matters if a Registry-aware frontend
  expects every node to serve every capability a node might be asked
  for — some capabilities (embeddings, today) only exist on the
  native-Ollama path, not through this backend at all.

## TransformersEngine Quantization (per docs/quantization-design.md)

- Done: `model.quantization` config (`none`/`4bit`/`8bit`,
  `MODEL_QUANTIZATION` env override, fail-fast on invalid values),
  `bitsandbytes` added to `requirements.txt`, `TransformersEngine.
  load_model()` branches to a `BitsAndBytesConfig` (NF4 + double quant for
  `4bit`, plain `load_in_8bit` for `8bit`) instead of `torch_dtype=
  torch.float16` for `none`. `./backend config` prints the configured
  value. First-ever `TransformersEngine` unit tests
  (`tests/test_transformers_engine.py`), mocking `from_pretrained()` —
  no GPU or `bitsandbytes` install needed to run them.
- Done: live validation on UBI, for `engine: transformers` +
  `quantization: none`. Long chain of real issues, full postmortem in
  `docs/problems.md`: `google/gemma-4-26B-A4B-it` (4-bit) failed on an
  outdated `transformers` and wouldn't have fit UBI's real free disk
  (~32GB of a 908GB disk, 97% used by other users' data) even if fixed.
  Switched to `Qwen/Qwen3-8B` (4-bit) — hit a broken `torchvision`
  leftover crashing `transformers`' import chain (fixed: uninstalled,
  unneeded for text-only inference), then a `device_map="auto"` bug
  mis-sizing quantized loads against the model's *unquantized* footprint
  versus real free VRAM (shared with another user's concurrent training
  job across all 4 GPUs), which tried to CPU-offload part of the model
  and got refused by bitsandbytes (fixed: new `_device_map()` helper in
  `engines/transformers_engine.py`, pins `device_map={"": 0}` for
  single-GPU quantized loads; regression tests in
  `tests/test_transformers_engine.py`). Then the real wall: UBI's driver
  (470.86, CUDA 11.4 max, no sudo) can't run `torch >= 2.1`, and
  `transformers >= 4.51` (needed for Qwen3) requires it — confirmed not
  Qwen3-specific, since `TinyLlama` hit the identical `torch.compiler`
  failure under `transformers` 4.51 too. Resolved: pinned
  `transformers==4.36.0` + `torch==2.0.1+cu117` (+ `Pillow==9.0.1` for an
  unrelated `libstdc++` incompatibility), now recorded in
  `requirements.txt` with comments. **Consequence**: UBI can only serve
  architectures `transformers` 4.36.0 already supported (confirmed:
  Llama family) — Qwen3 and Gemma are both off the table on this
  hardware until the driver gets upgraded by someone with admin access.
  `config/config.yaml` is back on `TinyLlama/TinyLlama-1.1B-Chat-v1.0`,
  confirmed live via `/health` and a real `/v1/chat/completions` call.
- Done (2026-07-31): picked a better TinyLlama-replacement. New
  `model.revision` config field (`MODEL_REVISION` env override,
  `config.py`/`engines/transformers_engine.py`) pins the Hub commit
  `TransformersEngine.load_model()` loads from — needed because
  `mistralai/Mistral-7B-Instruct-v0.2`'s `main` tokenizer.json was
  re-saved by a newer `tokenizers` library than UBI's pinned one can
  parse (docs/problems.md). Pinning `revision=
  dca6e4b60aca009ed25ffa70c9bb65e46960a573` (pre-re-save) resolves it.
  `NousResearch/Llama-2-7b-chat-hf` also confirmed working at `main`,
  no pinning needed. `config/config.yaml` now runs
  `mistralai/Mistral-7B-Instruct-v0.2` (revision-pinned), live-validated
  via `/health`, `/v1/models`, and a real `/v1/chat/completions` call.
  Discovered doing this: `bitsandbytes` is currently broken on UBI
  (`import bitsandbytes` itself fails under the pinned `torch==2.0.1`
  — see `docs/problems.md`), so `quantization` stays `none`; fp16 7B
  needs ~14.8GB VRAM, which doesn't fit GPU 0/1's ~9.4GB free (shared
  with another user's job) — `backend.gpu` moved from `0` to `2` (idle)
  to accommodate. Follow-up, not done: find a `bitsandbytes` build
  compatible with `torch==2.0.1+cu117` so 4-bit/8-bit quantization
  actually works on UBI, which would remove the VRAM/GPU-choice
  constraint above.
- Done (2026-07-31): multi-GPU, first real use. UBI has **4x RTX A4000**
  (64GB combined VRAM) — discovered 2026-07-30, confirmed working
  2026-07-31 with `NousResearch/Meta-Llama-3-8B-Instruct`, whose fp16
  footprint (~16.8GB - Llama-3's 128k-token vocabulary makes it larger
  than a same-labeled Llama-2/Mistral 7B) doesn't fit one 16GB card.
  No code change was needed: `TransformersEngine.load_model()` already
  called `device_map="auto"`, and setting `config/config.yaml`'s
  `backend.gpu: "2,3"` (a comma-separated list, exposing 2 visible
  devices via `CUDA_VISIBLE_DEVICES`) was sufficient - `accelerate`
  split the model ~7.5GB/~9.2GB across both with no CPU offload,
  confirmed live via `/health`, `/v1/models`, and a real
  `/v1/chat/completions` call. The known `GPUManager` caveat below is
  now observed live, not just theoretical: `./backend status`/`gpu
  current` show VRAM/temperature as unavailable for this config
  (`_gpu_by_index()` can't match a multi-index string against a single
  `nvidia-smi` row) - cosmetic only; `/health` reads
  `torch.cuda.get_device_name(0)` directly, unaffected, and actual
  inference is unaffected. Follow-up, not done: fix `GPUManager` to
  report real per-GPU info for a multi-GPU `backend.gpu` value, if
  accurate multi-GPU status reporting becomes important enough to
  justify it.
  Ceiling found the same session: disk, not VRAM. This box's single
  908GB volume is ~97-99% used by other users' data (13-28GB free
  depending on what's cached locally); a 70B-class model needs ~140GB
  for fp16 weights alone, which doesn't fit regardless of combined GPU
  VRAM. Practical "go bigger" ceiling is roughly the 13B-34B range
  (Llama-family) until disk changes.
- Done (2026-07-31): multi-family compatibility sweep, full pass/fail
  matrix and root causes in `docs/problems.md`'s "Multi-family model
  compatibility sweep" section. Confirmed working, not yet deployed:
  `Qwen/Qwen-7B-Chat` (needs `trust_remote_code=True` +
  `tiktoken`/`einops`/`transformers_stream_generator`),
  `deepseek-ai/deepseek-llm-7b-chat`,
  `deepseek-ai/deepseek-coder-6.7b-instruct`,
  `mistralai/Mistral-7B-Instruct-v0.1` (revision-pinned, same fix
  pattern as v0.2), `tiiuae/falcon-7b-instruct` (also needs
  `trust_remote_code=True`). Confirmed blocked, with root cause each
  time: Qwen1.5/2/3 (no native class, no `trust_remote_code`
  fallback), Gemma (same), Mistral-v0.3 and DeepSeek-R1-Distill-Llama-8B
  (both shipped only in a newer file/config format with no older
  revision to fall back to), Mistral-Nemo-12B (`head_dim` config field
  4.36.0 doesn't read). `microsoft/phi-2` is its own case: loads
  without error but silently random-initializes its entire attention
  stack from a weight-naming mismatch - flagged as the session's most
  important methodology lesson (exit code 0 isn't proof of a working
  model for `trust_remote_code` repos; always check generation
  coherence). Follow-up, not done: pick one of the confirmed-working
  candidates above to actually deploy, if Llama-3-8B ever needs
  replacing.
- Done: local Hugging Face cache discovery
  (`engines.transformers_engine.scan_local_cache()`,
  `./backend model list|current|info`'s new `Cached locally:` line,
  new `./backend model local` command). Read-only — surfaces drift
  between `config.yaml`'s catalog and what's actually on disk, doesn't
  auto-fix it. See `tests/test_transformers_engine.py`'s
  `ScanLocalCacheTests`.

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
