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
  `BenchmarkService.first_token_latency` does while streaming is unavailable.
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

Phase 5 Increment 1 (state reporting only, no load/unload/switch):

- `LifecycleState` enum in `services/lifecycle.py`: `unloaded`, `loading`,
  `ready`, `unloading`, `switching`, `degraded`. Matches the state set and
  transition table in `docs/model-lifecycle-design.md`.
- `InferenceService` owns `lifecycle_state`, currently always `ready` after
  the existing startup load.
- `/health` includes `lifecycle_state` alongside the existing `status`,
  `model`, `cuda`, `gpu` fields.
- `./backend status` prints `Lifecycle: <state>`.
- No runtime load/unload/switch, no worker supervision, no CUDA changes yet.

Phase 5 Increment 2 (command surface stubs, no lifecycle behavior):

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

Next milestones: Phase 5 Increment 3 (real model load/unload/switch
behavior, `docs/model-lifecycle-design.md`) is open. So is the Backend
Registry (`docs/future-tasks.md`) — its trigger condition (a real second
live Backend Node) is now met, motivated by the frontend wanting
user-facing Local-vs-UBI model choice instead of a static `.env` restart.
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
