# AGENTS.md

Behavioral contract for coding agents working on `nemoclaw-backend`.

## Project Identity

`nemoclaw-backend` is a unified inference management backend. It is not only a
Transformers server, and it is not an orchestrator.

Current runtime: FastAPI + Hugging Face Transformers serving a local model
through an OpenAI-compatible API. Future provider/engine support belongs here.

## Critical Boundary

Nemoclaw Core owns:

- agents
- memory
- planning
- skills
- RAG
- research workflows
- orchestration

Nemoclaw Backend owns all LLM/inference functionality:

- engines and model providers
- model selection and model metadata
- inference API
- benchmarking
- GPU/runtime inspection

Never add Core logic to this repo. Never duplicate backend-owned model listing,
model selection, benchmarking, provider/engine support, or GPU/runtime
inspection in Core.

## Architecture Rules

```text
CLI (Typer)
  -> Services: InferenceService, ModelManager, GPUManager, BenchmarkService
  -> Engines: InferenceEngine, TransformersEngine
  -> CUDA / GPU
```

Future engines may include Ollama, vLLM, llama.cpp, and OpenAI-compatible
providers. Do not implement them until explicitly requested.

- Every capability has exactly one owner.
- Services own business capabilities: inference coordination, model
  management, GPU inspection, benchmarking, and lifecycle.
- Engines own runtime-specific implementation: how Transformers or future
  Ollama/vLLM/llama.cpp/OpenAI-compatible backends load models and run
  inference.
- CLI commands and FastAPI routes are delivery surfaces, never owners. They
  validate input, delegate to a service, and format output.
- Before implementing a new capability, identify its owner. If no existing
  service or engine is the clear owner, introduce the correct one first.
- CLI commands delegate to services.
- Do not put business logic, timing logic, model loading, GPU inspection, or
  provider logic directly in `cli.py` or FastAPI routes.
- New backend capabilities should become services or engines.
- Benchmarks must go through the local OpenAI-compatible endpoint, never
  Transformers directly.
- `api.py` must stay independent of Transformers/CUDA internals.
- `ModelManager` owns configured models and selected/default model metadata.
- `InferenceService` owns runtime inference boundaries.
- `GPUManager` owns GPU discovery/status.
- `BenchmarkService` owns benchmark execution/formatting.

## Environment

- Deployment: UBI server, Ubuntu 18, RTX A4000 16GB.
- Conda: `source ~/miniforge3/bin/activate` then `conda activate llm`.
- Default server: `127.0.0.1:8000`.
- CLI dependencies such as `typer` are expected inside `llm`; `./backend ...`
  may fail outside that environment.

## API Stability

Keep these endpoints OpenAI-compatible:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

No breaking changes to `/v1/*` without explicit approval. Preserve
`uvicorn server:app --host 127.0.0.1 --port 8000` compatibility.

`openapi/backend-node.openapi.yaml` is the authoritative API contract. Any
endpoint addition or change (new route, changed request/response shape,
changed status code, changed `x-implementation-status`) must update that
file in the same increment. See `docs/api-contract.md` for the contract's
tier model and stability rules.

## Versioning
The project is pre-1.0. Minor bumps mark backend capabilities or architecture
milestones. Patch bumps mark fixes, hardening, or docs. Do not break `/v1/*`
without explicit architectural approval. Tags should mark validated runtime
milestones, not merely passing tests.

## Engineering Rules

- Keep code modular, testable, explicit, and production-quality.
- Keep dependencies conservative; no new frameworks without strong
  justification.
- Do not introduce Docker, LangChain, vLLM, Ollama, Core orchestration, or new
  providers unless explicitly requested.
- Every implementation phase needs tests; existing tests must keep passing.
- Update `README.md` and relevant `docs/` files with each phase.
- At the end of every phase, update `README.md`, relevant `docs/`, and the
  Current State section of this file. Do not let `AGENTS.md` drift from the
  code.
- Risky phases involving runtime state, GPU memory, process management, or
  active requests need a design document in `docs/` before implementation.
- Never fake unavailable functionality. Report it clearly, as
  `BenchmarkService.first_token_latency` does when a stream produces only
  reasoning and there is no answer latency to report.
- Do not overwrite `requirements.txt` with `pip freeze`; it is
  human-maintained direct runtime dependencies only.
- Every completed increment or phase ends with: run the full test suite,
  commit all changes with a concise descriptive message, and push to origin
  main. Work is not "done" until it is pushed. Never leave completed work
  uncommitted.

## Current State

This section must be updated at the end of every phase so agent guidance stays
aligned with the code.

Completed through Phase 4:

- FastAPI OpenAI-compatible API
- YAML config with env overrides
- Typer CLI: start/stop/restart/status/health/config/logs
- `InferenceService`, `InferenceEngine`, `TransformersEngine`
- `ModelManager`, `GPUManager`, `BenchmarkService`
- model, GPU, and benchmark CLI commands

Phase 5 Increment 1 (state reporting only, no load/unload/switch) —
**superseded by Increment 3 below; `lifecycle_state` is no longer fixed at
`ready`. Historical record only:**

- `LifecycleState` enum in `services/lifecycle.py`: `unloaded`, `loading`,
  `ready`, `unloading`, `switching`, `degraded`. Matches the state set and
  transition table in `docs/model-lifecycle-design.md`.
- `InferenceService` owns `lifecycle_state`, currently always `ready` after
  the existing startup load.
- `/health` includes `lifecycle_state` alongside the existing `status`,
  `model`, `cuda`, `gpu` fields.
- `./backend status` prints `Lifecycle: <state>`.
- No runtime load/unload/switch, no worker supervision, no CUDA changes yet.

Phase 5 Increment 2 (command surface stubs, no lifecycle behavior) —
**superseded by Increment 3 below, which made these endpoints real. The
`501` stub bodies and their helpers no longer exist; kept here as
historical record only:**

- Management endpoints under `/admin/model/` (`load`, `unload`, `switch`),
  separate from `/v1/*`, defined in `docs/model-lifecycle-design.md`.
- For well-formed requests, each returns HTTP `501` with a fixed JSON body
  built by `InferenceService.lifecycle_stub_response()` /
  `services/lifecycle.lifecycle_not_implemented_response()`. `load` and
  `switch` require a `ModelLifecycleRequest` body (`model_id: str`); a
  missing/malformed body fails FastAPI validation first and returns the
  standard `422`, not the `501` stub. `unload` takes no body and always
  returns `501`. Calling any of them never changes `lifecycle_state` and
  never touches the engine or CUDA.
- `./backend model load|unload|switch` call those endpoints, print the
  `detail` message, and exit non-zero. No timeout/wait/progress logic yet;
  that belongs to the real implementation.

OllamaEngine Increment 1 (config + engine factory):

- `config.yaml`'s `backend.engine` (`transformers` | `ollama`, default
  `transformers`), `ENGINE` env override, fail-fast on invalid values.
- `services/inference.py`'s `_build_engine()` factory selects the engine
  class. `engine: transformers` behavior is unchanged.

OllamaEngine Increment 2 (real read paths, live-validated on a Local Node
running actual Ollama — never UBI, per `docs/architecture.md`'s Target
deployment topology):

- `backend.ollama_host` (`config.yaml`, default `http://127.0.0.1:11434`,
  `OLLAMA_HOST` env override) is the daemon's base URL.
- `OllamaEngine.load_model()` calls `GET /api/tags` and confirms the
  configured `model.id` tag is present; never pulls. A missing tag raises
  a clear startup error naming `ollama pull <tag>`.
- `OllamaEngine.health()` / `list_models()` also use `GET /api/tags`;
  `list_models()` returns only the one configured tag, `owned_by:
  "ollama"`. `cuda`/`gpu` are sourced from `GPUManager` (new
  `GPUManager.gpu_name()`), not a direct `torch.cuda` call, since the
  backend process doesn't own Ollama's CUDA context.
- New `engines.base.EngineUnavailableError` (optional `partial_health`
  payload) is raised when the daemon is unreachable.
  `InferenceService.health()` catches it, transitions `lifecycle_state` to
  `degraded`, and projects `HealthResponse.status` from `lifecycle_state`
  via the new `services/lifecycle.health_status_for_lifecycle_state()`
  (`ready`->`ok`, `degraded`->`degraded`, else->`unavailable`) instead of
  trusting whatever the engine returns — matching the mapping already
  pinned in `openapi/backend-node.openapi.yaml`'s `HealthResponse.status`.
- `unload_model()` still raises `NotImplementedError` (Increment 4).

OllamaEngine Increment 3 (`chat()`/`generate_text()`, live-validated
against a real Local Node and real Ollama models):

- `OllamaEngine.chat()` posts `POST /api/chat`; `generate_text()` posts
  `POST /api/generate`. Both use a separate, generous fixed 120s timeout
  (distinct from read-path calls' 5s), resolving the "Timeout behavior"
  question `docs/ollama-engine-design.md` Section 7 left open for this
  increment — full operator-configurability of that timeout is deferred,
  not decided as unnecessary.
- Model-resolution decision (`docs/ollama-engine-design.md` Section 1) is
  enforced only in `OllamaEngine.chat()`: a request naming a `model` other
  than this instance's configured/servable model raises new
  `engines.base.ModelNotFoundError` before any daemon call, which `api.py`
  turns into HTTP `404` with the pinned `model_not_found` error shape.
  `TransformersEngine.chat()` keeps its existing echo-and-serve quirk
  unchanged (out of scope, per that section's explicit scope boundary).
  `GenerateRequest` has no `model` field, so `/generate` has no
  model-resolution check to make.
  `InferenceEngine.chat()`'s signature grew an optional `requested_model`
  parameter (both engines implement it; `TransformersEngine` accepts and
  ignores it) to carry the client's requested model id down to the engine
  that needs to check it.
- `EngineUnavailableError` during `chat()`/`generate_text()` now also
  transitions `InferenceService`'s `lifecycle_state` to `degraded` (same
  as a failed `health()`); `api.py` turns it into HTTP `503` on both
  `/v1/chat/completions` and `/generate`.
- Token-usage mapping (`docs/ollama-engine-design.md` Section 4):
  `prompt_eval_count`/`eval_count` -> `usage.prompt_tokens`/
  `completion_tokens`; missing fields report `0` and log a warning,
  per AGENTS.md's "never fake unavailable functionality" rule.
- `openapi/backend-node.openapi.yaml` amended in this increment (per the
  AGENTS.md rule that contract changes land with the behavior that needs
  them): new `ModelNotFoundResponse` schema, `404`/`503` responses on
  `POST /v1/chat/completions`, `503` on `POST /generate`, updated
  `x-current-behavior` describing the now-engine-dependent model
  resolution. Also fixed a pre-existing drift found while doing this:
  `ModelObject.owned_by` was pinned `const: local`, which `OllamaEngine`
  already violated since Increment 2 (`"ollama"`) — now `type: string`
  with both values documented.
- Live-validated against a real Ollama daemon and real pulled models:
  successful chat with correct token usage, `404 model_not_found` on a
  mismatched model (via real HTTP through `./backend start`), and
  `generate_text()` against `/generate`. Noted, not fixed (out of scope):
  reasoning-capable models like `qwen3` can return empty `content` if the
  token budget is exhausted by hidden "thinking" before an answer is
  reached — confirmed as an honest pass-through of the daemon's own
  response, not an engine bug; Ollama's `"think": false` option is not
  wired up (undecided by the design doc, left for a future increment if
  needed).

OllamaEngine Increment 4 (`unload_model()`, live-validated against a real
Local Node):

- `OllamaEngine.unload_model()` posts `POST /api/generate` with
  `{"model": <configured tag>, "keep_alive": 0}`, asking the daemon to
  evict the model from memory. Best-effort only: the backend does not
  stop or supervise the daemon process, and eviction is not guaranteed to
  be synchronous — live testing showed the real daemon's `GET /api/ps`
  can still list the model for a few seconds after the unload call
  returns before it actually clears.
- Not wired to any live HTTP endpoint (Non-goal, Section 5):
  `/admin/model/unload` stays a `501` stub regardless of engine, exactly
  as `TransformersEngine.unload_model()` remains unreferenced today. This
  completes `OllamaEngine`'s implementation of every `InferenceEngine`
  method per `docs/ollama-engine-design.md`.

OllamaEngine implementation (Increments 1-4) is now complete: config
selection, real read paths, real chat/generation, and best-effort unload,
all live-validated against a real Ollama daemon and real models. The
user's frontend (`nemoclaw-research-assistant`) has since been migrated
onto this backend's `/v1/chat/completions` via its `OpenAICompatibleProvider`
and an SSH tunnel to the UBI Node (`ssh -L 8000:127.0.0.1:8000 ubi-a4000`,
since the backend has no auth and binds `127.0.0.1` only) — confirmed
live end to end.

`TransformersEngine` quantized loading (`docs/quantization-design.md`):
new `model.quantization` config (`none` default, `4bit`, `8bit` via
`bitsandbytes`/`BitsAndBytesConfig`, `MODEL_QUANTIZATION` env override),
so UBI's RTX A4000 can serve a larger model than fp16 alone allows
(~7B was the prior practical ceiling). `none` is unchanged existing
behavior. First-ever `TransformersEngine` unit tests added
(`tests/test_transformers_engine.py`).

Live validation on UBI is **done** for `engine: transformers` +
`quantization: none`, after a long chain of real environment issues
(`docs/problems.md` has the full postmortem — read it before running
`pip install -U` on UBI again). In order: `google/gemma-4-26B-A4B-it`
(4-bit) failed on an outdated `transformers` and, separately, wouldn't
have fit UBI's real free disk (~32GB of a 908GB disk, 97% used by other
users' data). Switched to `Qwen/Qwen3-8B` (4-bit) — hit a broken
`torchvision` (fixed: uninstalled, unneeded), then a `device_map="auto"`
bug mis-sizing quantized loads against unquantized footprint (fixed:
new `_device_map()` in `engines/transformers_engine.py`, pins
`device_map={"": 0}` for single-GPU quantized loads), then the real
wall: UBI's driver (470.86, CUDA 11.4 max, no sudo to upgrade) cannot
run `torch >= 2.1`, and `transformers >= 4.51` (needed for Qwen3)
requires it — confirmed this isn't Qwen3-specific, since even `TinyLlama`
hit the identical `torch.compiler` failure under `transformers` 4.51.
Resolved by pinning `transformers==4.36.0` + `torch==2.0.1+cu117` (also
needed `Pillow==9.0.1`, newer Pillow needs a `libstdc++` symbol Ubuntu 18
doesn't have) — both now pinned in `requirements.txt` with comments.
**Practical consequence:** UBI can only serve architectures
`transformers` 4.36.0 already supported (confirmed: Llama family) — not
Qwen3, not Gemma. `config/config.yaml` is back on
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `quantization: none`, confirmed
live via `/health` and a real `/v1/chat/completions` call. Picking a
better model for UBI now means picking a `transformers`-4.36-era
architecture (e.g. Llama 2/3, Mistral), not just "recent and small
enough" — check `docs/problems.md` before assuming a model will load.

Discovered while investigating disk space: UBI actually has **4x RTX
A4000** (64GB combined VRAM), not 1. Not yet used — see
`docs/future-tasks.md`'s Multi-GPU entry.

Local model cache discovery: new `engines.transformers_engine.
scan_local_cache()` reads the local Hugging Face cache (read-only, never
downloads/deletes) — the `TransformersEngine`-specific counterpart to
`OllamaEngine`'s `GET /api/tags` check, since `config.yaml`'s
`model.available` is a static human-maintained catalog never verified
against reality. `./backend model list|current|info` now show `Cached
locally: yes (<size>) / no` per entry; new `./backend model local` lists
everything actually cached, including repos not in `config.yaml`. New
`huggingface_hub` dependency (was already transitive via `transformers`).
Discovery only — nothing auto-writes `config.yaml`.

Model-revision pinning + UBI model upgrade (2026-07-31): new
`model.revision` config field (`MODEL_REVISION` env override,
`config.py`/`engines/transformers_engine.py`) pins the Hub commit
`TransformersEngine.load_model()` loads from, since a model's `main`
tokenizer.json can be re-saved by a newer `tokenizers` library than
UBI's pinned one can parse, independent of architecture support
(`docs/problems.md`'s Mistral case). `NousResearch/Llama-2-7b-chat-hf`
(ungated Llama-2-7B-chat mirror) confirmed working at `main`, no
pinning needed. `mistralai/Mistral-7B-Instruct-v0.2` confirmed working
pinned to `revision: dca6e4b60aca009ed25ffa70c9bb65e46960a573` (predates
the tokenizer.json re-save) — now `config/config.yaml`'s live default,
replacing TinyLlama, confirmed via `/health`, `/v1/models`, and a real
`/v1/chat/completions` call with correct token usage. Also discovered:
`bitsandbytes` is currently broken on UBI regardless of model choice
(`import bitsandbytes` itself fails under the pinned `torch==2.0.1` —
see `docs/problems.md`), so `quantization` stays `none`; fp16 Mistral-7B
needs ~14.8GB VRAM, which doesn't fit GPU 0/1's ~9.4GB free (frequently
occupied by another user's job) — `backend.gpu` moved from `0` to `2`
(the idle card) to accommodate. Not fixed: a `bitsandbytes` build
compatible with `torch==2.0.1+cu117`.

Multi-GPU sharding, first real use (2026-07-31): UBI's other 3 GPUs
(discovered 2026-07-30, unused until now) got their first live
exercise. `NousResearch/Meta-Llama-3-8B-Instruct` (ungated Llama-3-8B
mirror, no revision pin needed, confirmed working under
`transformers==4.36.0`) has a much larger vocabulary (128k tokens)
than Llama-2/Mistral (32k), so its fp16 footprint (~16.8GB) doesn't
fit one 16GB card. No code change was needed —
`TransformersEngine.load_model()` already called `device_map="auto"`;
setting `config/config.yaml`'s `backend.gpu: "2,3"` was sufficient,
and `accelerate` split the model ~7.5GB/~9.2GB across both GPUs with
no CPU offload. `NousResearch/Meta-Llama-3-8B-Instruct` is now the
live default on UBI, replacing Mistral-7B, confirmed via `/health`,
`/v1/models`, and a real `/v1/chat/completions` call. `GPUManager`'s
known multi-GPU caveat (`docs/future-tasks.md`) is now observed live:
`./backend status`/`gpu current` show VRAM/temperature as unavailable
for a comma-separated `backend.gpu`, cosmetic only. All testing and
deployment this session stayed on GPU 2/3 — GPU 0/1 (another user's
concurrent training job) were deliberately never touched, checked via
`nvidia-smi` before and after each step. Found the same session: the
real ceiling on "how big a model" isn't VRAM, it's disk — this box's
single volume is ~97-99% used by other users' data (13-28GB free
depending what's locally cached), and a 70B-class model needs ~140GB
for fp16 weights alone regardless of combined GPU VRAM. Practical
ceiling is roughly the 13B-34B range (Llama-family) until disk
changes.

Multi-family compatibility sweep (2026-07-31): systematically tested
Qwen (1/1.5/2/3), DeepSeek, more Mistral versions, Gemma, Falcon, and
Phi against the pinned `transformers==4.36.0` stack - full pass/fail
matrix with root causes in `docs/problems.md`. Confirmed working,
available as future deploy options: `Qwen/Qwen-7B-Chat` (needs
`trust_remote_code=True`), `deepseek-ai/deepseek-llm-7b-chat`,
`deepseek-ai/deepseek-coder-6.7b-instruct`,
`mistralai/Mistral-7B-Instruct-v0.1` (revision-pinned),
`tiiuae/falcon-7b-instruct`. `NousResearch/Meta-Llama-3-8B-Instruct`
remains the live default. Key methodology finding: `microsoft/phi-2`
loads with no exception but silently random-initializes its attention
weights from a naming mismatch - "no crash" isn't proof of a working
model for `trust_remote_code` repos; always check generation
coherence. All testing stayed on GPU 2/3 - GPU 0/1 (another user's
concurrent job) were never touched.

Qwen version investigation, resolved (2026-07-31, isolated
`llm-qwen-test` conda env, never touching the production `llm` env -
`conda create --clone` proved unfaithful for this, see
`docs/problems.md`): Qwen2 and Qwen2.5 both work under
`transformers==4.37.0` (the smallest possible bump from the pinned
4.36.0) on UBI's unchanged `torch==2.0.1+cu117` - `qwen2` was added to
`CONFIG_MAPPING_NAMES` exactly at 4.37.0, no torch/driver change needed.
Caught along the way: Qwen2.5-1.5B produces silently-garbage output
under forced `float16` (what `TransformersEngine`'s `_load_kwargs()`
hardcodes today) - needs `bfloat16` (the model's own native dtype);
promoting any Qwen2/2.5 model to `TransformersEngine` needs that code
fix too, not just a `requirements.txt` bump. **Qwen3 is a confirmed dead
end for `TransformersEngine`** on UBI's driver: `qwen3` only enters
`CONFIG_MAPPING_NAMES` at `transformers==4.51.0`, but the
`torch.compiler`/`torch>=2.1` break was confirmed already present by
`4.50.0` (binary-searched: `4.49.0` still works) - no version has both.
Regression-checked on 4.37.0: TinyLlama, Mistral-7B, and
`Llama-3-8B` (then-live) all still load correctly - not yet promoted to
`requirements.txt`, since the deployment decision below superseded it.

Ollama-on-UBI (2026-07-31, `docs/ollama-on-ubi-design.md`): Ollama's
llama.cpp/GGML runtime turned out to have far looser driver requirements
than `transformers`/`torch` - it runs with genuine GPU acceleration
directly on UBI under the same pinned driver (470.86), sidestepping the
Qwen3 dead end above entirely. Root-free install (standard `curl | sh`
needs sudo, not available on `d3894`): standalone tarball, binary-searched
to exactly `v0.9.2` - the newest release satisfying two independent
constraints simultaneously (glibc <= UBI's `2.27`, since Ubuntu 18.04
predates `2.28`; and the bundled `cuda_v11` runtime, since driver 470.86
maxes at CUDA 11.4 and `v0.9.3`+ dropped CUDA 11 support entirely).
Installed at `~/ollama/` on UBI, `OLLAMA_MODELS=~/ollama/models`, started
manually (mirrors `./backend start`'s own manual-start convention) with a
`crontab -e` `@reboot` entry for resilience across an actual UBI reboot
(no systemd without root). `config/config.yaml` now runs `engine: ollama`
against this same-machine daemon (`ollama_host` stays default
`http://127.0.0.1:11434`) - initially `model.id: qwen3:8b`, live-validated
via `/health`, `/v1/models`, and a real `/v1/chat/completions` call with
correct token usage. `TransformersEngine`'s config (quantization,
revision, prior `available` entries) is left intact for a one-line
revert. `docs/architecture.md`'s Target deployment topology updated: the
UBI Node is no longer `TransformersEngine`-only: `OllamaEngine` now runs
as a genuine second deployment of itself (UBI Node and Local Node both
run `OllamaEngine`, on different machines - two Backend Nodes, one
engine). Not done: no CLI wiring for the Ollama daemon's own lifecycle
(deliberately out of scope, same boundary as the Local Node case - this
backend doesn't own Ollama's CUDA context).

Bigger Qwen3 sizes (2026-07-31, same day): `qwen3:30b-a3b` (MoE, ~18GB)
and `qwen3:32b` (dense, ~20GB) both confirmed working, each correctly
tensor-split across both GPUs by Ollama automatically (`qwen3:32b`:
~12.3GB/~12.3GB on GPU 2/3, comfortable headroom under 16GB each;
`qwen3:30b-a3b`: ~10.6GB/~10.2GB) - no code or config change needed
beyond `CUDA_VISIBLE_DEVICES=2,3` already being set for the daemon.
**`model.id: qwen3:32b` is now the live default** (best quality
validated so far), chosen over keeping `30b-a3b` pulled at the same time
purely on disk grounds: UBI's disk is a hard, shared constraint (~26GB
free at the start of this investigation, from a box already ~98% full
with other users' data) - `30b-a3b` and `32b` together don't fit
(~38GB), so `30b-a3b` was deleted (`ollama rm`) before pulling `32b`.
Pulling it again is a normal `ollama pull qwen3:30b-a3b` if wanted, but
**check `df -h` before pulling anything else** - deploying `32b` alone
left only ~2GB free, tighter than is comfortable on a box this shared.
`config/config.yaml`'s `model.available` lists all four validated Qwen3
sizes (`1.7b`/`8b`/`30b-a3b`/`32b`) with a note on which are actually
pulled right now vs. just validated-and-available-to-repull.

Shared-GPU busy check (2026-07-31, same day): neither engine, nor
`GPUManager`, previously checked whether `backend.gpu`'s configured
index(es) were already in use by another process before loading a
model - purely a static assumption in config, verified only by manually
running `nvidia-smi` throughout this session's work. New
`GPUManager.busy_gpus(threshold_mib=500)` checks each configured GPU
index (handles comma-separated multi-GPU) and returns any already
showing usage above the threshold. `InferenceService` now optionally
takes a `gpu_manager` and logs a warning for each busy GPU found,
checked once at startup before `engine.load_model()` runs (so any usage
found at that point cannot be this process's own - by definition,
before it has loaded anything). `create_inference_service()` wires this
up for the real server; test construction (`InferenceService(engine)`
with no `gpu_manager`) is unaffected, since it's an optional parameter
skipped when absent. `./backend status` and `./backend gpu current`
both surface it as an "Other GPU usage" line. No per-process
attribution (`nvidia-smi`'s basic query doesn't provide it) and nothing
is blocked - this is visibility, not an enforced lock, live-verified by
triggering a real warning off the Ollama daemon's own still-warm
`qwen3:32b` session.

Shared-GPU busy check, enforcement follow-up (2026-07-31, same day):
the above was warn-only - `./backend start` would proceed onto a busy
GPU regardless. New `GPUManager.idle_alternative_gpus()` (GPUs outside
`backend.gpu`'s configured indexes that are themselves idle) pairs with
`busy_gpus()` in a new `cli._check_gpu_before_start()`: if the
configured GPU is busy and an idle alternative exists elsewhere,
`./backend start` **refuses outright** (non-zero exit, telling the
operator which GPU(s) are idle so `backend.gpu` can be repointed) rather
than starting on top of someone else's job. If no idle alternative
exists anywhere on the box, it doesn't just refuse - it asks for
interactive confirmation (`typer.confirm`) instead, since there's no
better option to suggest. New `--force`/`-f` flag on both `start` and
`restart` (which passes it through) skips the check entirely for
scripted/non-interactive use. Live-verified on UBI: warmed
`qwen3:32b`'s VRAM, confirmed `./backend start` refused (exit 1, GPU
0/1 correctly detected as idle alternatives, no server process left
running) and `./backend start --force` started successfully anyway. The
interactive confirm-prompt branch is unit-tested with a mocked
`typer.confirm` (not live-triggered - doing so for real would require
occupying all 4 GPUs, including GPU 0/1, which this project has
deliberately never touched).

Ollama `think` wiring (2026-07-31, same day): `OllamaEngine.chat()`/
`generate_text()` never exposed Ollama's `think` field (noted as
deliberately out of scope in the Increment 3 notes above) - a real gap
found live-testing the frontend integration, where `qwen3:32b`'s
`<think>` reasoning trace silently consumed a small `max_tokens` budget
before reaching an answer, truncating the response entirely. New
`model.think_default` config (`MODEL_THINK_DEFAULT` env override, tri-
state: `null`/unset leaves Ollama's own per-model default alone, `true`/
`false` forces it) plus a new per-request `think` field on both
`ChatCompletionRequest` and `GenerateRequest`, resolved in that priority
order by `OllamaEngine._apply_think()`. `TransformersEngine` accepts and
ignores both new `think` parameters, matching the existing
`requested_model` interface-parity precedent - no reasoning-mode concept
there. `openapi/backend-node.openapi.yaml` amended in this increment
(new `think` property on both request schemas, marked as a Nemoclaw
extension not part of the OpenAI spec). `config/config.yaml`'s live
`think_default` is now `false` - confirmed live: default chat responses
now skip reasoning entirely (a "12" answer cost 3 completion tokens,
down from 98-146 spent on `<think>` before), a `"think": true` per-
request override still works with enough `max_tokens` headroom, and the
frontend's own `scripts/test_llm_provider.py` smoke test (which is what
surfaced the original truncation) now returns a complete, coherent
response with the default 80-token budget instead of an empty/cut-off
one.

Model family compatibility + context windows (2026-07-31, same day):
found live that `qwen3`'s native context is only 40,960 tokens (its own
GGUF metadata), an architectural ceiling, not a memory one - motivated
checking what else this pinned Ollama (`v0.9.2`) can actually run.
Confirmed via real pull attempts (the version gate fires at the
manifest stage, before downloading weights, so this is cheap):
`llama3.1`, `llama3.2`, `mistral`, `deepseek-r1`, `phi4`, `gemma3`,
`codellama`, `qwen2.5`, `command-r`, `hermes3` all pass. `gemma4` is
**blocked** - `412: requires a newer version of Ollama`, because it's
multimodal (ships a vision projector) and needs Ollama's newer engine,
same category of wall as the `transformers`/Qwen3 story just from
Ollama's side. `gemma3:4b` pulled and load-tested fully (coherent
output); the rest passed the pull check but weren't individually load-
tested. Also confirmed: `ollama pull hf.co/<repo>:<quant>` pulls GGUF
files directly from Hugging Face, not just Ollama's curated library -
real download confirmed against a Qwen2.5 GGUF repo, no version block.
Native context windows (`gemma3`/`llama3.1`/`llama3.2`/`deepseek-r1`/
`hermes3` all ~131K, `mistral`/`qwen2.5` 32,768, `phi4`/`codellama`
16,384) recorded in `docs/ollama-on-ubi-design.md`'s latest update -
`qwen3` and `gemma3:4b` verified directly on UBI, the rest from each
model's published HuggingFace config (not independently re-verified
against UBI's specific GGUF). Disk ended this session at ~3GB free
after deleting `qwen3:8b`/`qwen3:1.7b` to test the (blocked) `gemma4`
pull, then pulling `gemma3:4b` instead - check `df -h` before pulling
anything else.

Live default switched to llama3.2-vision:11b (2026-08-01): `qwen3:32b`'s
native context (40,960) was far short of the ~131K several other
confirmed-working families offer (`docs/ollama-on-ubi-design.md`).
`llama3.2-vision:11b` (7.8GB, Meta's `mllama` multimodal architecture)
pulled successfully and works for text-only use - unlike `gemma4`, not
blocked by the newer-engine version gate, likely because `mllama`
predates whatever generic multimodal support `gemma4` needs. Loads
**fully GPU-resident on a single GPU** (41/41 layers, ~12.1GB) - lighter
than `qwen3:32b` needing 2+ GPUs. `llama3.2-vision:90b` (54.6GB) is
infeasible on UBI's disk regardless of the version gate. `model.id` is
now `llama3.2-vision:11b`, live-validated via `/health`, `/v1/models`,
a real `/v1/chat/completions` call, and the full test suite (141 tests,
all passing). All previously pulled models (`qwen3:32b`, `gemma3:4b`,
`llama3.2:3b`) were deleted to free disk during this exploration and
are no longer pulled, but remain validated options in
`config/config.yaml`'s `model.available`. UBI disk: ~18GB free after
this pull.

Live default switched to qwen3:30b (2026-08-01, same day): tried
`qwen3.5:35b` first (24GB, multimodal) - **blocked**, same `412:
requires a newer version of Ollama` wall as `gemma4`, confirming this
isn't a `gemma4`-specific quirk but a general multimodal-architecture
gate on the pinned `v0.9.2` daemon. Both `qwen3.5` and `gemma4` are dead
ends until a driver/OS upgrade (Ubuntu 18.04's glibc 2.27 + driver
470.86's CUDA 11.4 ceiling - the same root cause already blocking Qwen3
under `TransformersEngine` and `bitsandbytes`). User has agreed to
pursue that upgrade (Ubuntu 24.04 + accompanying driver reinstall, RTX
A4000/Ampere has no hardware blocker) with whoever administers UBI -
not yet scheduled, no code dependency on it.

Pivoted to `qwen3:30b` (MoE, `qwen3moe` architecture, ~18GB) as an
interim default instead: **262,144 native context**, confirmed via
`ollama show qwen3:30b` directly on UBI - far larger than dense `qwen3`
(32b/8b/1.7b, 40,960; do not conflate the two, `config/config.yaml`'s
`model.available` previously had this mistagged as `qwen3:30b-a3b` at
40,960, now corrected). `llama3.2-vision:11b` was deleted to make room
(disk was down to 18GB free); `qwen3:30b` alone leaves UBI at **~7.6GB
free, 100% used** - the tightest this project's disk has ever been.
User has adopted a deliberate single-model-at-a-time policy on UBI
until that improves. `model.id` is now `qwen3:30b`, live-validated via
`/health`, `/v1/models`, a real `/v1/chat/completions` call, and the
full test suite (141 tests, all passing).

Found live and not fixed (documented, not faked, per this file's rule):
`qwen3:30b` does **not** honor `"think": false`, unlike `qwen3:32b`
before it - confirmed by calling the raw Ollama daemon directly
(`/api/chat` with `"think": false"`), bypassing `OllamaEngine` entirely,
and still getting a full `<think>` reasoning trace in the response.
Root cause is presumed to be this MoE variant's chat template lacking
the enable/disable-thinking conditional dense `qwen3` has - not a
backend bug, and not something `OllamaEngine._apply_think()` can work
around from the request side. `think_default: false` is kept anyway
(harmless, matches the deliberate `qwen3:32b`-era choice, and still
correct for any other model that *does* honor it) but does not actually
suppress reasoning for this specific tag. The reasoning preamble
(~48 tokens for a trivial question) fits comfortably inside
`max_tokens_default: 256` so nothing truncates, unlike the original
`qwen3:32b` truncation bug this field was built to fix - just a
standing token-overhead cost on every request. A real fix likely needs
a newer Ollama (same upgrade path as above), not attempted this
session.

Phase 5 Increment 3 (real model load/unload/switch,
`docs/model-lifecycle-design.md`): the `/admin/model/*` endpoints and
`./backend model load|unload|switch` now perform real transitions,
replacing the Increment 2 `501` stubs (`lifecycle_not_implemented_response()`
and the `LifecycleStubResponse` schema are gone).

- **Engine scope is deliberately split.** Engines declare a new
  `supports_runtime_lifecycle` class attribute on `InferenceEngine`
  (default `False`); `InferenceService` enforces the policy. `OllamaEngine`
  sets it `True` — the daemon owns the CUDA context, so "load" is a
  tag-presence check plus a pointer change and "unload" is the existing
  `keep_alive: 0` call. `TransformersEngine` **refuses** every lifecycle
  op with the new `engines.base.LifecycleNotSupportedError` (HTTP `501`),
  pointing the operator at editing `model.id` and restarting. That is the
  honest reading of the design doc's own argument that in-process CUDA
  cleanup is best-effort and can poison later loads — implementing it
  anyway would have faked a working operation. Worker supervision (and
  with it `TransformersEngine` lifecycle support, plus side-by-side
  switching) stays deferred.
- **Two validation gates, catalog-restricted.** `ModelManager.validate_model()`
  rejects any id absent from `config.yaml`'s `model.available` *before the
  engine is touched at all*; only then does the engine check reality (for
  Ollama, that the tag is actually pulled — it still never pulls).
  Switching to an undocumented model is not allowed, by choice.
- **Persistence is opt-in.** Transitions are runtime-only by default;
  `--persist` / `"persist": true` additionally rewrites `model.id` via the
  existing `ModelManager.select_model()`. Keeps `model use` (config, needs
  restart) and `model switch` (runtime, live) distinct, and stops a
  frontend model picker from dirtying a tracked config file implicitly.
- `services/lifecycle.py` now mirrors the design doc's state table as
  `LEGAL_TRANSITIONS` + `validate_transition()`, so an illegal edge raises
  instead of silently corrupting state. New `LifecycleUnavailableError`
  (request arrived while not `ready` -> HTTP `503`, rejected never queued)
  and `LifecycleConflictError` (illegal-from-this-state -> HTTP `409`).
  Note `degraded -> unloaded` is a *direct* edge (there is no
  `degraded -> unloading`), so `unload_model()` has a separate path for it.
- `InferenceService` counts in-flight requests under a
  `threading.Condition` and drains them before calling the engine;
  FastAPI's threadpool makes that concurrency real. Past the drain timeout
  it proceeds anyway with a warning, so an operator's unload can't be
  blocked forever by a stuck request. `/health` gained `loaded_model` and
  `target_model` (the runtime model can differ from config's `model.id`
  after a non-persisted switch); `./backend status` surfaces both.
- **CLI options `--timeout` and `--json` only.** `--wait/--no-wait` and
  `--poll-interval` are deliberately *not* implemented: transitions are
  synchronous for Ollama, so there is no async state to poll and building
  a polling loop would fabricate progress reporting that doesn't exist.
  They belong with worker supervision.
- **Found live on UBI, fixed in this increment:** switching to a tag that
  is in `model.available` but not pulled on the daemon left the backend
  `degraded` — which rejects all inference — even though the engine had
  correctly verified the target *before* releasing the current model, so
  the old model was still loaded and fine. One bad request took a healthy
  backend offline. New `engines.base.ModelUnavailableError` marks "target
  rejected, nothing changed"; `InferenceService._transition()` restores
  the pre-transition state for it (HTTP `409 model_unavailable`) instead
  of degrading, while genuine mid-transition failures still degrade.
  Every unit test passed while this was broken (the fake engines raised
  generic exceptions) — lifecycle work needs live validation.
- `openapi/backend-node.openapi.yaml` amended in the same increment: the
  three endpoints moved from `x-implementation-status: stub` to
  `implemented` with new `LifecycleResultResponse`/`LifecycleErrorResponse`
  schemas and their `404`/`409`/`501`/`503` cases; `persist` added to
  `ModelLifecycleRequest`; `loaded_model`/`target_model` added to
  `HealthResponse`; the chat/generate `503` now also covers
  mid-transition rejection.

Live-validated on UBI end to end against the real Ollama daemon and
`qwen3:30b` (2026-08-02), pulling `qwen3:1.7b` (1.4GB) as a second tag for
the switch test and `ollama rm`-ing it afterwards to restore the
single-model-at-a-time policy — disk returned to 6.5GB free, and GPU 0/1
(another user's job) were never touched, checked before and after.
Confirmed live: `404` on an unconfigured id; `409` on loading a different
model while ready; idempotent load of the current model; `422` on a
malformed body; `409 model_unavailable` on a configured-but-not-pulled tag
with the old model still serving; a real `switch` (`qwen3:30b` ->
`qwen3:1.7b`) reflected in `/health`, `/v1/models`, `./backend status`'s
new runtime-divergence line, and a real chat answered by the new model;
the `model_not_found` guard correctly following the switch (the *old*
model now `404`s); `unload` making chat return `503`; `load` from
`unloaded` recovering; and `--persist` writing `config.yaml` in both
directions, ending byte-identical to the committed file.

Shared-GPU safety, external-runtime gap closed (2026-08-02): the
busy-GPU guard added earlier only ever protected the *backend process*.
`config.py` sets `CUDA_VISIBLE_DEVICES` from `backend.gpu` for itself,
but under `engine: ollama` **the backend never loads a model** - the
Ollama daemon does, using whatever devices *it* was launched with. On UBI
that daemon runs with `CUDA_VISIBLE_DEVICES=0,1,2,3`, so `backend.gpu:
"2,3"` constrained nothing that mattered: Ollama could place weights on
GPU 0/1, another user's cards. It had been landing on 2/3 only because
its own scheduler picks by free VRAM - luck, not enforcement.

- `GPUManager.gpu_is_in_use()` now treats **either** memory (>500MiB) or
  **utilization** (>10%) as busy. Memory alone missed a compute-heavy job
  with a small resident footprint; a false positive (we pick another
  card) costs far less than a false negative (we land on someone's job).
  `busy_gpus()` and `idle_alternative_gpus()` both use it.
- New `GPUManager.availability()` -> `GPUAvailability(in_use, free)`: a
  census of **every** detected GPU, classified dynamically. It never
  assumes which indexes are "the busy ones" - verified by a test where
  GPU 2/3 (this project's usual safe pair) are the busy ones.
- New `GPUManager.visible_gpu_indexes_for_process(pid)` reads a process's
  own `CUDA_VISIBLE_DEVICES` from `/proc/<pid>/environ`. An **unset**
  variable returns every index, because that is CUDA's real default
  (sees all GPUs); unreadable returns `None` ("unknown"), never a guess.
  `unsafe_gpus_for_process(pid)` intersects that with the in-use set.
- New `engines.ollama_engine.find_daemon_pids()` locates `ollama serve`
  (engine-specific knowledge, so it lives with the engine; `GPUManager`
  owns the GPU-safety analysis).
- `cli._check_external_runtime_gpus()` runs before `_check_gpu_before_start`'s
  existing config check. It **warns** when the daemon can reach a busy GPU
  but **refuses only when every reachable GPU is busy** - the case where
  the daemon has no safe placement left. Either way it prints the pinning
  fix (`CUDA_VISIBLE_DEVICES=<free> ollama serve`). The warn/refuse split
  is deliberate: the deployed daemon sees all four GPUs, so
  "a busy GPU is reachable" becomes true the moment one colleague starts a
  job. Refusing on that would block nearly every start on this shared box
  and make `--force` routine, destroying the value of the warning. Ollama
  schedules by free VRAM, so while anything is free it takes that.
  It **reports rather than restarting the daemon** - this backend
  deliberately does not own the daemon's lifecycle, and killing it would
  drop other work.
- `--force` still bypasses, but is no longer silent: it names the GPUs
  being overridden and warns that another user's job may be disrupted.
- `./backend status` and `./backend gpu list` print the availability
  census ("N of M GPU(s) in use by other processes, K free") with a
  per-GPU IN USE/free line.

Live-verified on UBI: real detection read `CUDA_VISIBLE_DEVICES=0,1,2,3`
off both daemon PIDs, and the warn/refuse/`--force` paths were exercised
against a simulated busy GPU 0/1 (the exact situation observed earlier
that day).

Own-usage attribution (2026-08-03): the busy-GPU checks had **no
per-process attribution at all** - the original design leaned on "the
check runs before this backend has loaded anything, so any usage must be
someone else's". That assumption is false under `engine: ollama`, where
the daemon outlives the backend and keeps a model resident for
`OLLAMA_KEEP_ALIVE` (5 min default). Our own model therefore read as
another user's job, and the backend would warn about - or refuse to start
because of - **the very model it was serving**. Verified live: with
`qwen3:30b` resident, `busy_gpus()` returned `['2','3']` and
`_check_gpu_before_start()` refused.

- `GPUManager.gpu_processes()` uses `nvidia-smi --query-compute-apps`
  (works on UBI's 470.86) for real per-PID attribution. That query reports
  `gpu_uuid`, not index, so it maps uuid -> index via a second query.
- `GPUManager.gpu_owner()` classifies each card `free` / `ours` /
  `others`. `ours` requires **every** attributable process on the card to
  be ours - a shared card is never claimed. Usage nvidia-smi cannot
  attribute falls back to the memory/utilization heuristic and counts as
  `others`: unexplained memory is never assumed to be ours.
- `GPUAvailability` gained an `ours` bucket and a `usable` property
  (`free + ours`). `busy_gpus()`, `idle_alternative_gpus()`,
  `availability()` and `unsafe_gpus_for_process()` all take `own_pids`.
- **The VRAM belongs to a child process.** `ollama serve` does not hold
  the model; it spawns an `ollama runner` per loaded model and nvidia-smi
  attributes the memory to that child (confirmed live: serve was 23825,
  the 4x~6GB belonged to runner 17181). New
  `engines.ollama_engine.find_runtime_pids()` walks `/proc` to include
  descendants - matching only the daemon PID would have left the bug in
  place.
- New `InferenceEngine.runtime_pids()` (default `[]`) lets the service
  layer ask the engine which PIDs are its own without knowing anything
  Ollama-specific. `InferenceService._warn_if_gpu_busy()` and the CLI
  checks both pass it through.

Live-verified on UBI in both directions: with our own model resident on
all four GPUs the census read "4 held by our own model (reclaimable)" and
`./backend start` proceeded; with GPU 0/1 simulated as another user's and
GPU 2/3 holding our model, only 0/1 were flagged.

Dynamic GPU pinning, `./backend gpu pin-free` (2026-08-03): Ollama's
scheduler splits a model across **every** GPU it can see once one card
isn't enough - it does not try to use the fewest. Measured on UBI:
`qwen3:30b` needs 25.2 GiB (weights 17.1 GiB + KV 0.75 + graph 1.0,
`parallel=2`), which fits in two 15.6 GiB cards, yet it loaded with
`layers.split=13,12,12,12`, taking ~6GB from all four. Worse, an earlier
log line shows `memory.available="[15.6 15.6 9.1 9.1]"` - it took a slice
of cards another user was already on. `OLLAMA_SCHED_SPREAD` only forces
*more* spreading; v0.9.2 has no "use fewest" setting, so
`CUDA_VISIBLE_DEVICES` is the only lever and it can only be set at daemon
start.

Static pinning was rejected by the user for a good reason: it is fragile
on a shared box - a daemon pinned to `2,3` fails to load when someone
else takes 2/3 even though 0/1 are idle. So the selection is made
**fresh at run time**:

- `GPUManager.select_gpus_for(required_mib, own_pids)` returns the
  **fewest** usable GPUs that fit, **lowest index first**. No pairing
  rule - one, two, three or four, any combination. Cards holding only our
  own model count at full capacity, since restarting releases them.
  Returns `None` when even all free cards are not enough, so the caller
  refuses rather than half-placing a model.
- `OllamaEngine.estimated_vram_mib()` scales the tag's on-disk size by
  `VRAM_OVERHEAD_FACTOR` (1.5). Ollama only reports the true figure after
  loading, so this is an estimate; 1.5 is rounded up from the measured
  25.2/17.28 = 1.46, because over-estimating costs one extra GPU while
  under-estimating spills to CPU or fails.
- `restart_daemon_pinned()` reconstructs the launch command from
  `/proc/<pid>/cmdline` + `environ` (not from the docs - the deployed
  daemon has drifted before), overrides only `CUDA_VISIBLE_DEVICES`, and
  terminates by PID, never `pkill -f`. This is the **one** place the
  backend touches the daemon's lifecycle, and only on explicit operator
  command.
- **Caught during live testing:** the first implementation sent the
  restarted daemon's output to `DEVNULL`, silently losing `serve.log` -
  the log path comes from shell redirection at launch, so it is in
  neither argv nor environ. `daemon_log_path()` now reads
  `/proc/<pid>/fd/1` and the restart inherits it.

Live-verified end to end on UBI: before, `layers.split=13,12,12,12` over
four cards; after `./backend gpu pin-free`, `layers.split=25,24` with
`memory.available="[15.6 GiB 15.6 GiB]"`, GPU 2/3 left at 3 MiB and
genuinely free for someone else. Log destination preserved across the
restart. **Not persistent by design** - a daemon restart or reboot
returns to whatever the `@reboot` crontab sets, and the command is
re-run when the free set changes.

Multi-model sizing (2026-08-03): `pin-free` sizes for `config.model.id`,
which is correct **only because the frontend's model choice is
persisted** - the frontend picks the model and can change it at any time,
so `config.model.id` is the single record of what is actually served.
Frontend switches must therefore use `"persist": true`. Single model
resident at a time is the operating assumption (`OLLAMA_MAX_LOADED_MODELS`
is `0`/auto, so Ollama *could* hold several, but only one of the
available tags is used); nothing sizes for concurrent residency.

The gap that created: the daemon's GPU set only changes when it restarts,
while the served model can change any time, so a switch can outgrow the
pin it was placed under.

- `OllamaEngine.visible_vram_mib()` sums the VRAM of the GPUs the daemon's
  own `CUDA_VISIBLE_DEVICES` exposes; `vram_warning_for(model_id)`
  compares that to the model's estimate.
- New `InferenceEngine.vram_warning_for()` (default `None`) so the service
  can ask without Ollama-specific knowledge.
  `InferenceService._vram_warning()` computes it **once**, logs it, and
  attaches it to the load/switch result as an optional `warning` field
  (added to `LifecycleResultResponse` in the OpenAPI contract). Advisory
  only - the requirement is an estimate and Ollama can offload to CPU, so
  refusing on a heuristic would block legitimate switches, and a failure
  in the check itself never breaks the transition.
- `pin-free` prints a per-model fit table (every pulled tag vs. the
  selected GPUs' total) so an unfittable future switch is visible before
  it bites.

Live-verified on UBI with two tags pulled (`qwen3:30b` 17.28 GiB +
`qwen3:1.7b` 1.27 GiB, disk 8.2 -> 7.0GB): with the daemon pinned to one
GPU for the small model, the table marked `qwen3:30b DOES NOT FIT` and
switching to it returned the `warning` field; after re-running `pin-free`
(2 GPUs) the same switch returned no warning and the model loaded on GPU
0/1 only. `qwen3:1.7b` removed afterwards, disk back to 8.2GB.

Note for future sessions: the running `uvicorn` backend does **not** pick
up code changes until `./backend restart` - a live `/admin/model/switch`
silently returned no `warning` for exactly this reason before the
restart, which looked like a bug and was not.

`/v1/models` now lists selectable models (2026-08-03, **approved Tier 1
behavior change**): it previously returned exactly one entry, the loaded
model, so nothing exposed the *choices* a frontend picker needs. Three
lists disagreed - `/v1/models` (1), `config.yaml`'s catalog (10), actually
pulled (1) - meaning a picker built on the catalog would have offered 10
options of which 9 failed.

- It now enumerates `model.available`, **filtered to the active
  `backend.engine`** (listing Transformers repos while running Ollama
  offered choices the instance cannot serve), joined with runtime facts.
  This matches OpenAI's own `/v1/models` semantics and the contract's
  existing decision record that there must be no separate native
  model-listing endpoint.
- OpenAI fields (`id`, `object`, `created`, `owned_by`) are unchanged, so
  existing clients still parse it. `loaded`, `pulled`, `size_mib`, `fits`
  are additive Nemoclaw extensions. Unknown facts are **omitted, never
  guessed** (a non-pulled model has no `size_mib`/`fits`).
- Ownership split held: `ModelManager` owns the catalog (what may be
  selected), the engine owns runtime facts via the new
  `InferenceEngine.model_runtime_info(model_ids)` (default `{}`),
  `InferenceService.list_models()` joins them. A failure computing
  runtime facts degrades to catalog-only rather than failing the list; an
  `EngineUnavailableError` still propagates, and `api.py` now maps it to
  `503` on `/v1/models` (previously an unhandled `500`).

Live-verified on UBI that the flags are truthful, not decorative:
`/v1/models` returns 7 Ollama entries (3 Transformers filtered out) with
`qwen3:30b` `loaded/pulled/fits` and six `pulled: false`; selecting a
`pulled: false` entry returns `409 model_unavailable`, an uncatalogued id
returns `404 model_not_configured`, and the `pulled/fits` one loads and
serves. Lazy loading needs no backend code - Ollama reloads an evicted
model on the next request (confirmed).

Reasoning separated from answers (2026-08-03): reasoning models were
leaking their hidden thinking into `content`. For "Capital of Portugal?
One word." `qwen3:30b` returned ~142 tokens of monologue, a stray
`</think>`, then "Lisbon" - so any OpenAI client rendering `content`
showed the whole thing, and ~88% of the tokens were invisible-to-the-user
overhead. Three distinct behaviours were measured live, and one
marker-based rule (`engines.ollama_engine._split_reasoning()`) covers all:

- **Ollama already separated it** into `message.thinking` (dense `qwen3`
  with thinking enabled) - the engine was **silently discarding that
  field**. Now carried through.
- **The chat template leaked it inline** (`qwen3:30b` MoE, which also
  ignores `"think": false` entirely - reconfirmed against the raw daemon).
  The opening tag is consumed by the prompt, so only `</think>` appears;
  split on the **last** closing marker rather than requiring a matched
  pair.
- **No reasoning at all** (`llama3.2`, `gemma3`, `mistral`, or thinking
  disabled) - nothing matches and the rule is a no-op. **No per-model
  configuration is needed**, which is the point: it is driven by what the
  model actually emits.

`content` is now the answer only; the thinking is surfaced as a
`reasoning` field on the chat message and on `/generate`, omitted entirely
when there is none (documented as a Nemoclaw extension in the contract).
`TransformersEngine` returns `reasoning: None` for interface parity,
matching how it already ignores `think`.

Live-verified across all three: `qwen3:30b` -> `content` exactly
`"Lisbon"` with 866 chars of reasoning moved aside and no `</think>` left
in content; `qwen3:1.7b` with `think: true` -> clean content plus 468
chars from `message.thinking`; same model with `think: false` -> no
`reasoning` key at all and 3 completion tokens.

Observed, not a backend bug: `qwen3:1.7b` answers **"Porto"** (wrong) with
thinking disabled and "Lisbon" with it enabled - reproducible at the raw
daemon too. Disabling reasoning on small reasoning models trades accuracy
for tokens; worth knowing before setting `think_default: false` on one.

**Operator decisions taken 2026-08-02, deliberately leaving two things
as they are:**

- The UBI daemon stays launched with all four GPUs visible (both the
  running process and the `@reboot` crontab line). The user chose this
  over pinning; the warn/refuse logic above is calibrated for it.
- `OLLAMA_KEEP_ALIVE` stays unset, i.e. Ollama's 5-minute default. Worth
  recording because it was initially mistaken for a GPU-pinning concern:
  `CUDA_VISIBLE_DEVICES` controls *which* GPUs the daemon may use and has
  no bearing on residency, while `OLLAMA_KEEP_ALIVE` controls *how long*
  a model stays in VRAM after the last request. Verified live that an
  idle daemon holds zero VRAM: after testing `qwen3:30b` (~18GB across
  two cards), `ollama ps` was empty and all four GPUs read 3 MiB. The
  GPUs already free themselves without any code or config change.

Audit housekeeping (2026-08-03, `docs/audit-87a844d.md`): a full audit at
`87a844d` found four things worth fixing immediately, all now done.

- **Deleted `model_runtime.py`.** It was imported by nothing, but called
  `create_inference_service()` at *import time* exactly as `api.py` does -
  so any future import would have built a second `InferenceService` and
  triggered a second `engine.load_model()`. `docs/architecture.md` and
  `docs/developed.md` still advertised it as a live compatibility facade,
  pointing future work straight at the hazard. Both references removed.
- **Corrected four stale "`/admin/model/* are 501 stubs`" claims** in
  `docs/api-contract.md` (x2), `docs/developed.md` and
  `docs/architecture.md`. False since Phase 5 Increment 3; `501` now means
  only "this engine does not support runtime lifecycle".
- **`/health` moved from `partial` to `implemented`.** Its
  `x-current-behavior` still claimed no code path could produce
  `degraded`/`unavailable`; both are live and tested. Only
  `/v1/chat/completions` remains `partial`, correctly - that is streaming.
- **Optional-dependency tests now skip instead of erroring.**
  `tests/test_transformers_engine.py` imported `torch` at module scope, so
  every dev machine without it saw a hard `ImportError` in the suite - easy
  to normalise as "expected" and mask a real failure. Both test classes are
  now `skipUnless`-guarded. Local and UBI both report 255 tests; the 10
  Transformers tests genuinely run on UBI and skip elsewhere.

SSE streaming (2026-08-03, `v0.5.0`+): `"stream": true` on
`POST /v1/chat/completions` now returns Server-Sent Events in OpenAI's
chunk format, terminated by `data: [DONE]`. `/v1/chat/completions` moved
to `implemented`; **no endpoint is `partial` any more**.

- New `InferenceEngine.supports_streaming` + `chat_stream()`;
  `OllamaEngine` implements it over Ollama's NDJSON `/api/chat` via a new
  lazy `_post_stream()`. `TransformersEngine` leaves the flag `False` and
  gets a `400` naming itself - it generates a whole response in one call,
  so there is no incremental path to expose and faking one would be a lie.
- `InferenceService.chat_stream()` is **deliberately not a generator**.
  Rejections (engine cannot stream, lifecycle not ready) must surface
  before the response starts; a generator defers them to first iteration,
  by which time the status code is already sent and a `400`/`503` is
  impossible. It validates eagerly and returns `_stream_deltas()`, which
  holds `_serving()` for the stream's whole life - satisfying the design
  doc's requirement that active streams count as active requests and are
  **drained**, not cut off mid-response (unit-tested: a switch blocks
  until the stream finishes).
- **Reasoning in a stream is decided up front, never guessed.** A stream
  cannot retract what it sent, so `/api/show`'s `capabilities` list
  (which includes `thinking`) determines the handling before the first
  token: models that cannot reason stream immediately; models where
  Ollama supplies separate `thinking` deltas stream both immediately;
  a model that leaks inline (`qwen3:30b`) has its reasoning prefix
  buffered until `</think>` proves where it ends, after which the answer
  streams normally. If the marker never arrives the buffer is flushed as
  **content**, since mislabelling it as reasoning would hide the entire
  answer (the `think: false`-honoured case).
- Live-verified on UBI: a 5-city answer streamed as 17 content events
  (`'L','is','bon','\n','Port','o'...`) after 1662 chars of reasoning,
  with usage on the final chunk. Caught and fixed during that run: the
  blank line after `</think>` usually arrives in a *later* chunk than the
  marker, so trimming only the marker's own chunk leaked a leading
  newline into `content`.

The rest of `docs/audit-87a844d.md`'s recommended order is now done too:

- **`first_token_latency` measures for real** over an SSE stream. Reasoning
  deltas are excluded from the metric - for a reasoning model the first
  thing on the wire is hidden thinking, so timing that would flatter the
  number while saying nothing about when the user sees an answer. Live on
  UBI: first reasoning token ~3.4s, first *answer* token 3.5-11.2s. A
  stream that emits only reasoning is reported unavailable rather than
  counted as zero, keeping the never-fake-a-number property.
- **API tests are behavioural.** `api.py` now exposes its service through
  a FastAPI dependency built on first use (plus `set_inference_service()`
  as a test seam), so importing the module no longer loads a model;
  `app.py` triggers construction in a startup hook so the server still
  loads up front and the GPU busy-check still runs before anything is
  loaded. `tests/test_api.py` replaces the source-text greps with 21 real
  request tests. Added `httpx` (test-only, required by starlette's
  TestClient). One source-level assertion is kept deliberately - that
  `api.py` never imports Transformers or torch, an architectural
  constraint rather than a behaviour.
- **GPU policy left `cli.py`.** New `services/gpu_safety.py`:
  `GPUSafetyService.evaluate_start()` returns a `StartDecision`
  (`proceed`/`confirm`/`refuse`) and the CLI only renders it.
  Engine-specific process lookup sits behind `OllamaRuntimeInspector`, so
  the policy knows nothing about Ollama. Behaviour is unchanged,
  re-verified live; `cli.py` is no longer computing decisions it should
  only be formatting.

Next milestones: **see `docs/completion-plan.md`** — the plan to finish the
whole project, written 2026-08-03 at v0.6.0. Its headline is that this
backend is feature-complete for inference and **almost all remaining work
is in the frontend** (`nemoclaw-research-assistant`), which as of its
commit `08b5ee4` uses none of 2026-08-03's work: its
`OpenAICompatibleProvider.chat()` pops `stream` off kwargs and discards
it, and `models()` returns bare ids, throwing away the
`loaded`/`pulled`/`size_mib`/`fits` flags added for a picker. Do not add
backend surface without checking the frontend needs it.

Nothing from `docs/audit-87a844d.md` remains. Still open by earlier
explicit decision, not oversight: Phase 5 worker supervision (which would
unlock `TransformersEngine` lifecycle and side-by-side switching), the
Backend Registry, the UBI OS/driver upgrade, and persistently pinning the
Ollama daemon's GPUs.

Also open: Phase 5 worker supervision (the deferred half of
`docs/model-lifecycle-design.md`'s Minimal Implementation Plan, steps 7-8)
would unlock `TransformersEngine` lifecycle support and side-by-side
switching. So is the Backend Registry (`docs/future-tasks.md`) — its
trigger condition (a real second live Backend Node) is now met, motivated
by the frontend wanting user-facing Local-vs-UBI model choice instead of a
static `.env` restart; runtime `model switch` is now the backend half of
that.
An OS/driver upgrade on UBI (Ubuntu 24.04 + current driver) is agreed
but unscheduled - would remove the root cause behind the Qwen3.5/gemma4
Ollama version gate, the `transformers`/Qwen3 dead end, and broken
`bitsandbytes` all at once.

## Commands

Run real CLI commands inside the `llm` Conda environment:

```bash
./backend start|stop|restart|status|health|config|logs
./backend model list|current|use <model_id>|info <model_id>
./backend model local   # what's actually cached on disk (TransformersEngine only)
./backend model load|unload|switch <model_id>   # stubs: 501 not implemented
./backend gpu list|current|monitor
./backend gpu pin-free   # restart Ollama pinned to the GPUs free right now
./backend benchmark latency|throughput|vram|first-token-latency
```

Tests:

```bash
python3 -m unittest discover -s tests
```

Useful docs: `docs/architecture.md`, `docs/developed.md`,
`docs/future-tasks.md`, `docs/problems.md`, `docs/model-lifecycle-design.md`,
`docs/ollama-engine-design.md`, `docs/api-contract.md`,
`openapi/backend-node.openapi.yaml`.
