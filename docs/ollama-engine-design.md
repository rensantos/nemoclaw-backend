# OllamaEngine Design

This is a design document only. It does not implement `OllamaEngine`, does
not change `config.py`, `services/`, `engines/`, `api.py`, `schemas.py`, or
any test, and does not edit `openapi/backend-node.openapi.yaml`. It is the
bridge between the pinned Backend Node API contract
(`openapi/backend-node.openapi.yaml`, commit `352f424`) and the actual
`OllamaEngine` implementation that follows it, per the roadmap in
`docs/architecture.md`: `OllamaEngine` -> real model lifecycle -> SSE
streaming -> `MonitorService` -> multi-model/multi-GPU. `OllamaEngine` goes
first specifically to validate the `InferenceEngine` abstraction against a
second, structurally different backend before any CUDA-lifecycle work
begins.

Two semantics are decided here because they apply to every engine, not just
Ollama: model resolution (Section 1) and model listing (Section 2). Both
were flagged as unresolved in the Backend Node API contract's drift report
(commit `352f424`) and are settled below as DECIDED, not left open.

## 1. DECISION: Model resolution semantics

**Decided:** a request naming a model this backend instance cannot serve is
**rejected** with an OpenAI-style 404, not silently served by the loaded
model.

### Rejection shape

```json
{
  "error": {
    "message": "The model 'llama2:7b' does not exist or is not currently loaded by this backend instance.",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

This matches OpenAI's own `model_not_found` convention. It is a new response
shape relative to the pinned contract's existing `SimpleErrorResponse`
(`{"detail": "..."}`, used today for the two `400` cases on
`/v1/chat/completions`). Introducing it requires a
`openapi/backend-node.openapi.yaml` amendment (new error component, new
`404` response entry) in the same code increment that implements this
behavior, per the rule added to `AGENTS.md` in commit `352f424`. That
amendment is not made here.

### Response `"model"` field

**Decided:** the response `"model"` field always equals the **served**
model — the model that actually generated the completion — never the
raw request echo. When a client's requested model differs from the served
model in a way the engine chooses to tolerate (see below), the response may
carry an additive, optional `"requested_model"` field alongside `"model"`
so the discrepancy is visible without breaking strict OpenAI clients (which
ignore unknown top-level fields). Adding this field is also a
`ChatCompletionResponse` schema amendment for the implementation increment,
not made here.

In practice, for the single-active-engine design in Section 3, "served
model" only ever has one possible value per backend instance: the one
model this instance is configured to serve. So the practical behavior is:

- request `"model"` omitted, or equal to the configured/servable model:
  serve it; response `"model"` = the servable model id; no
  `"requested_model"` field.
- request `"model"` names anything else: reject with the `404` above.
  Never silently substitute.

There is no "fallback to loaded/default model while pretending the request
succeeded" path. Silent substitution is what created the drift this
document exists to close: a client asking for a model it didn't get, with
no signal that happened.

### Supersession, explicitly scoped

This decision **supersedes** the `x-current-behavior` documented today on
`POST /v1/chat/completions` in `openapi/backend-node.openapi.yaml`:

> "The response `"model"` field currently echoes the request's `"model"`
> (or the configured default if omitted) as a label. It does NOT select
> which model actually serves the request... A client requesting a
> `"model"` other than the loaded one still gets served by the loaded
> model, labeled with the requested id."

That is the behavior being replaced. **Scope boundary:** this decision
governs the `OllamaEngine` implementation from day one, because it is new
code with no installed base to break. Applying the same fix to the
existing `TransformersEngine`/`api.py` chat-completions path is explicitly
**out of scope for this design and its implementation increment** — it is
a separate future task (recorded in `docs/future-tasks.md`) so that fixing
a documented drift item doesn't silently expand an engine-integration
increment into a `/v1/chat/completions` behavior-change increment for the
existing engine and its existing callers.

## 2. DECISION: Model listing semantics with multiple engines

**Decided:** `GET /v1/models` lists **only the currently servable model(s)
for the active engine** — not every model an engine could theoretically
reach. Concretely, under the single-active-engine design (Section 3), that
means exactly one entry, matching `TransformersEngine.list_models()`'s
existing shape today.

This is deliberate, not just parity for parity's sake. Ollama itself is
capable of serving any locally-pulled tag on demand without a Transformers-style
load/unload cycle — its daemon will lazily load whatever tag a request
names, memory permitting. If `OllamaEngine.list_models()` enumerated every
tag `GET /api/tags` returns, `/v1/models` would advertise models this
backend instance's configured contract (Section 1) would then reject on
the next request, because Section 1 only accepts the one configured model
per instance. Listing what the request-resolution decision won't actually
serve would recreate exactly the kind of contract-vs-behavior gap this
document exists to close. So: **what `/v1/models` lists is exactly what
`/v1/chat/completions` will accept as `"model"` — no more, no less.** This
is what "keeps OpenAI-client compatibility" means here: a client that reads
`/v1/models` and requests one of the listed ids is guaranteed to succeed,
not sometimes rejected.

`/v1/models` remains the sole, authoritative, OpenAI-compatible
model-listing endpoint. No native `/models` endpoint is introduced,
confirming the decision already pinned in
`openapi/backend-node.openapi.yaml` (`GET /v1/models` description: "There
is NO separate native /models endpoint — do not add one"). Nothing in this
design revisits that.

### Engine ownership per model

**Decided:** use the existing OpenAI-compatible `"owned_by"` field on each
`ModelObject`, not a new native extension. `TransformersEngine` already
sets `"owned_by": "local"`; `OllamaEngine` sets `"owned_by": "ollama"`. Real
OpenAI API responses already use arbitrary strings here (`"openai"`,
`"openai-internal"`, `"system"`, organization ids), so OpenAI-compatible
clients already tolerate opaque values — no client-side assumption breaks.
Richer per-engine metadata (loaded/unloaded state, engine type, feature
flags) belongs in the *native* Tier 2 surface — the already-planned `GET
/capabilities` and `GET /engines` endpoints — not in `/v1/models`, keeping
the OpenAI-compatible surface exactly OpenAI-shaped.

### Ollama tag-to-model-id mapping

**Decided:** use the raw Ollama tag as the model id (e.g. `"llama3:8b"`),
**not** a synthesized namespace like `"ollama/llama3:8b"`.

Rationale: namespacing exists to prevent collisions when multiple engines
could simultaneously offer models with the same bare name. Section 3 keeps
exactly one engine active per backend instance, so no such collision is
possible in this phase — there is only ever one servable model id, and it
is unambiguous which engine produced it (`"owned_by"` already says so).
Under that constraint, a synthetic prefix buys nothing and costs
portability: an operator or a Nemoclaw application copying a model id
verbatim between this backend and a raw Ollama-compatible tool (or a future
second Ollama-backed instance) should see the same id both places. Revisit
namespacing if and when the multi-engine-dispatch phase (Section 3,
deferred) makes same-name collisions across simultaneously active engines
possible within one instance.

### `config.yaml model.available` vs Ollama's dynamic tag list

**Decided (authority split):**

- For the `transformers` engine, `config.yaml`'s `model.available` /
  `model.id`, owned by `ModelManager` (`services/model.py`), remain
  authoritative exactly as today — a static, human-curated list.
- For the `ollama` engine, `config.yaml` does **not** get a `model.available`
  equivalent. The daemon's own live state (`GET /api/tags`, `GET /api/ps`)
  is authoritative for "what models exist/are pulled." `config.yaml` for an
  Ollama-configured instance only needs the single tag this instance is
  configured to serve (its `model.id` equivalent), validated dynamically
  against the daemon's live tag list at startup rather than against a
  static YAML enumeration.

This avoids duplicating Ollama's own catalog into YAML, which would drift
the moment an operator runs `ollama pull`/`ollama rm` outside the backend's
knowledge. `ModelManager` is unchanged by this — it continues to own
Transformers configuration only; it does not gain Ollama-awareness (see
Section 3).

## 3. Engine selection and configuration

**Decided:** add `engine: transformers | ollama` under the existing
`backend:` section of `config/config.yaml`, defaulting to `transformers`.
This requires, in the implementation increment:

- `config.py`: add `engine: str` to `BackendConfig`, add
  `"engine": "transformers"` to `DEFAULTS["backend"]`, wire an `ENGINE`
  environment override alongside the existing `HOST`/`PORT`/`GPU`
  overrides.
- `services/inference.py`: `create_inference_service()` branches on
  `settings.backend.engine` to instantiate `TransformersEngine` or a new
  `OllamaEngine`, instead of unconditionally constructing
  `TransformersEngine`.
- `services/model.py` (`ModelManager`): **no change.** It stays
  configuration-only for the Transformers path, per its existing charter
  ("`ModelManager` does not know about Transformers, CUDA, tokenizers, or
  loaded runtime models" — `docs/architecture.md`, Model Configuration).
  `OllamaEngine.list_models()` talks to the Ollama daemon directly, the same
  way `TransformersEngine.list_models()` already builds its own single-entry
  response without consulting `ModelManager` (`InferenceService.list_models()`
  delegates to `engine.list_models()`, never to `ModelManager` —
  `ModelManager` only backs CLI `model list/current/use/info` commands
  today). No architectural boundary needs to move for engine selection to
  work.
- Existing deployments that don't set `engine:` in YAML get identical
  behavior to today — zero migration required for current Transformers-only
  configs.

### Single-active-engine vs multi-engine dispatch

**Recommendation: keep single-active-engine for this phase.** One
`InferenceService` instance holds exactly one engine and one servable
model, as it does today (`InferenceService.__init__` takes one `engine`,
calls `load_model()` once). Multi-engine dispatch — routing a single
backend instance's requests across two or more simultaneously-active
engines by request-level model selection — is **not** introduced here; it
belongs with the roadmap's later "multi-model/multi-GPU support" item.

Justification:

- The stated purpose of putting `OllamaEngine` first is to validate that
  the `InferenceEngine` contract itself is engine-agnostic — that a second,
  structurally different implementation (external daemon, no CUDA
  ownership) can satisfy the same six methods `TransformersEngine`
  satisfies. That validation only requires *one* `OllamaEngine` running at
  a time; it says nothing about dispatch.
- Introducing multi-engine dispatch now would require request-level engine
  routing logic in `InferenceService` or `api.py` — new business logic
  unrelated to "does the contract fit a second engine," widening this
  increment into the multi-model work the roadmap already schedules later
  and deliberately keeps separate.
- Decisions 1 and 2 above (reject-on-mismatch, list-only-servable) are
  specifically simple *because* there is only ever one servable model per
  instance. Multi-engine dispatch would immediately reopen both as
  multi-valued questions — better solved once, deliberately, in the
  multi-model phase, not as a side effect of adding a second engine type.
- Running Transformers and Ollama concurrently is still possible today
  without in-process dispatch: run two separate backend node processes
  (two `InferenceService` instances, two ports), each independently
  registerable with the future Backend Registry
  (`docs/registration-schema.json`) once it exists. That is the interim
  multi-engine story for this phase, not in-process routing.

## 4. OllamaEngine contract mapping

`OllamaEngine` implements `engines/base.py`'s `InferenceEngine` exactly as
`TransformersEngine` does — same six methods, same call sites in
`services/inference.py`. No changes to the interface itself (consistent
with `docs/architecture.md`'s Engine Interface growth policy: methods are
not added or changed ahead of their phase).

| `InferenceEngine` method | Ollama daemon mapping |
| --- | --- |
| `load_model()` | **No pull.** Calls `GET /api/tags`, confirms the configured tag is present locally. If absent, raises a clear startup error directing the operator to `ollama pull <tag>` — mirrors `TransformersEngine.load_model()`'s eager startup load, but validates rather than downloads. May optionally issue one warm `POST /api/chat`/`POST /api/generate` call (empty/minimal prompt) so the model is resident in the daemon's memory before the first real request, matching the "loaded at startup" expectation `/health` and `/v1/models` already carry for `TransformersEngine`. |
| `unload_model()` | `POST /api/generate` (or `/api/chat`) with `"keep_alive": 0` and no meaningful prompt, which asks the daemon to evict the model from memory. Best-effort: the daemon process itself is not stopped or supervised by the backend — only the model's residency in daemon memory is affected. |
| `health()` | `GET /api/tags` (or `GET /api/version`) as a lightweight daemon-reachability probe. See "Daemon availability handling" below for what happens when it fails. |
| `list_models()` | `GET /api/tags`, filtered to the one configured/servable tag (Section 2) — not the full daemon catalog. Returns the same `ModelListResponse` shape `TransformersEngine.list_models()` returns today, with `"owned_by": "ollama"`. |
| `chat(messages, max_tokens, temperature)` | `POST /api/chat` with `{"model": tag, "messages": [...], "stream": false, "options": {"temperature": ...}}`. Ollama's non-streaming response carries `message.content`, `prompt_eval_count`, `eval_count`. See token accounting below. |
| `generate_text(prompt, max_new_tokens, temperature)` | `POST /api/generate` with `{"model": tag, "prompt": ..., "stream": false, "options": {"temperature": ...}}`. Ollama's `response` field is completion-only — it does **not** echo the prompt back, unlike `TransformersEngine.generate_text()`'s documented quirk (full decode including prompt, per the drift noted in commit `352f424`). `/generate` is already Tier 2 legacy/deprecated; this cross-engine inconsistency is noted here as a fact, not fixed, since fixing the Transformers side is out of scope (Section 1's scope boundary applies to `/generate` too). |

### Daemon availability handling

**Decided behavior, with an explicit contract gap flagged:**

- `POST /v1/chat/completions` (and `/generate`): if the daemon is
  unreachable during a call, `OllamaEngine` raises a common, engine-agnostic
  `EngineUnavailableError` (a new small exception type to add to
  `engines/base.py` in the implementation increment — not Ollama-specific,
  so `TransformersEngine` or any future engine can raise the same type).
  `InferenceService`/`api.py` catches it and returns `503 Service
  Unavailable` with a clear message. This is consistent with how
  `docs/model-lifecycle-design.md` already uses `503` for "not currently
  servable" states elsewhere in the system, rather than an unhandled `500`.
- `GET /health`: this is where a real contract gap surfaces. The pinned
  `HealthResponse` schema in `openapi/backend-node.openapi.yaml` defines
  `"status"` as `const: ok` — today's only code path, since
  `TransformersEngine.health()` always succeeds once `load_model()` has run
  and there's no failure branch. A daemon-down `OllamaEngine.health()` has
  no honest way to report that fact within a schema that only permits
  `"ok"`. **This design does not widen that schema** (out of scope: no
  OpenAPI edits this increment). The intended behavior once implemented —
  `"status"` gains a value describing unavailability (e.g. `"error"` or
  `"degraded"`), or `lifecycle_state` alone carries the signal via
  `degraded` — requires a `openapi/backend-node.openapi.yaml` amendment
  in the *same* future increment that implements it, per the rule in
  `AGENTS.md`. Flagged here as exactly the kind of gap the roadmap
  anticipated ("validate the engine abstraction... before touching CUDA");
  it is a contract gap, not a bug, and is reported rather than patched
  around.

### `lifecycle_state` under a daemon-owned engine

Mapping `services/lifecycle.py`'s `LifecycleState` values to an engine that
doesn't own its own process:

| State | Meaning under `OllamaEngine` |
| --- | --- |
| `unloaded` | No model resident in the daemon (before first load, or after an explicit unload). |
| `loading` | Short-lived: the startup tag-existence check and optional warm-up request are in flight. Likely much shorter than a Transformers cold load. |
| `ready` | Daemon reachable; configured tag confirmed present/resident. |
| `unloading` | The `keep_alive: 0` request is in flight. Unlike Transformers, this is a request to the daemon, not a worker-process drain-and-restart. |
| `switching` | Meaningful in principle (change configured tag), but **no-op at the stub level in this phase**: `/admin/model/switch` remains a `501` stub for every engine (Non-goal, Section 5). `OllamaEngine` needs no switching logic beyond what already exists. |
| `degraded` | Daemon unreachable, or the configured tag is missing locally (analogous to a Transformers load failure). |

**Consistency gap with `docs/model-lifecycle-design.md`, reported not
fixed:** that document's State Transition Table (`ready -> unloading,
switching` only) has **no `ready -> degraded` edge**, yet an Ollama daemon
dying mid-request — or, equally, an unrecoverable Transformers/CUDA error
while `ready` — is exactly a `ready -> degraded` scenario. This gap
predates Ollama; it is not introduced by this design, but building
`OllamaEngine` is what surfaces it, since daemon crashes are a realistic
failure mode this engine must reason about. This document does not amend
`docs/model-lifecycle-design.md`'s transition table; that is reported as
drift for a future increment to resolve (likely alongside the real
lifecycle-transition work in Phase 5 Increment 3+, which owns that table).

### Token usage accounting

Ollama's `/api/chat` and `/api/generate` responses report
`prompt_eval_count` and `eval_count`. Mapping:

- `prompt_eval_count` -> `usage.prompt_tokens`
- `eval_count` -> `usage.completion_tokens`
- `usage.total_tokens` = their sum (matching `TransformersEngine`'s
  existing arithmetic)

**Decided fallback when counts are missing** (observed to happen for some
quantizations/cache-hit responses): report `0` for the missing field(s) and
log a warning server-side. The pinned `Usage` schema requires integers, so
there is no `null`/optional path available without an OpenAPI amendment
(out of scope here); `0` is chosen over a fabricated estimate (e.g.
whitespace tokenization) because AGENTS.md's rule — "Never fake unavailable
functionality; report it clearly" — rules out inventing a plausible-looking
but non-tokenizer-derived count. `0` is visibly a sentinel, not a
plausible-looking wrong answer, and the server log is where the honest
"unavailable" signal lives until/unless the wire schema changes. This
matches the spirit of `BenchmarkService.first_token_latency`'s existing
"report unavailable, don't fake" pattern, adapted to a schema that doesn't
have a boolean `available` field to lean on.

### GPU reporting under Ollama

The backend process no longer owns the CUDA context — Ollama's daemon
does. Two things change meaning, one doesn't:

- `GET /health`'s `"gpu"`/`"cuda"` fields, as `TransformersEngine.health()`
  implements them today, answer "does *this process* have a CUDA
  context/device" via a direct `torch.cuda` call. Under Ollama, that
  question stops correlating with "is inference actually running on GPU,"
  since the backend process itself may or may not import `torch` and
  never allocates GPU memory regardless. **Recommendation:** `OllamaEngine`
  should populate these fields via `GPUManager` (`services/gpu.py`)
  instead of a direct `torch.cuda` call — `GPUManager` already answers "is
  there a CUDA-capable GPU on this host" process-independently via
  `nvidia-smi`, and its `_torch_cuda_state()` already degrades gracefully
  (`try/except ImportError`) if `torch` isn't installed at all, which an
  Ollama-only deployment has no reason to require. This reframes the
  fields as host-level GPU presence rather than backend-process CUDA
  ownership — a more honest and engine-agnostic meaning either way.
- CLI `backend gpu list|current|monitor` commands need **no change** and
  remain accurate under Ollama: `GPUManager._detect_with_nvidia_smi()`
  queries whole-GPU memory via `nvidia-smi`, which is process-agnostic —
  it already reports the Ollama daemon's real VRAM usage today, the same
  way it would report any other process's. `CurrentGPUInfo.current_model`
  (`services/gpu.py`) simply echoes `config.model.id`, which is accurate
  regardless of engine.
- Net: GPU visibility at the CLI/`GPUManager` layer already extends
  correctly to Ollama with zero code changes; only `/health`'s
  process-local CUDA fields need to be re-sourced from `GPUManager` to stay
  meaningful.

## 5. Explicit non-goals for the implementation increment that follows

- **No streaming.** `"stream": true` continues to return the documented
  `400`/target-SSE contract exactly as pinned; streaming arrives with the
  dedicated streaming phase later in the roadmap, for every engine at once,
  not piecemeal per engine.
- **No model pulling/downloading via the backend.** `OllamaEngine.load_model()`
  validates the configured tag is present; it never calls anything
  equivalent to `ollama pull`. Pulling is an operator action.
- **No multi-engine simultaneous serving** within one backend instance
  (Section 3). Running two engines means running two backend node
  processes.
- **No embeddings.** `POST /v1/embeddings` stays `unscheduled`, matching
  the pinned contract; no engine gains an `embed()` method in this
  increment.
- **No changes to `/admin/model/*` stub behavior.** All three endpoints
  keep returning their fixed `501` bodies regardless of which engine is
  configured; `OllamaEngine`'s `unload_model()` mapping above exists as an
  engine method, unreferenced by any live HTTP endpoint, exactly as
  `TransformersEngine.unload_model()` is unreferenced today.
- Additionally, and specific to decisions made in this document: **no fix
  to `TransformersEngine`'s model-resolution behavior** (Section 1's scope
  boundary), and **no `config.yaml model.available` equivalent for
  Ollama** (Section 2's authority split).

## 6. Implementation plan sketch

Ordered increments, each independently testable and shippable:

**Increment 1 — Config + engine factory.**
Add `backend.engine` (default `transformers`) to `config.py`/`DEFAULTS`,
with an `ENGINE` env override. Update `create_inference_service()` in
`services/inference.py` to branch on it. Add `engines/ollama_engine.py` as
a skeleton class satisfying `InferenceEngine` (methods may raise
`NotImplementedError` initially, or go straight to Increment 2's read
paths).
*Test strategy:* unit tests for config parsing/precedence
(env > YAML > default) for the new key, and a unit test that
`create_inference_service()` selects the right engine class per config,
using a fake/mock engine the way `tests/test_inference_service.py` already
tests `InferenceService` without a real Transformers load.

**Increment 2 — `OllamaEngine` read paths: `health()`, `list_models()`,
`load_model()`.**
Implement the `GET /api/tags`-based reachability/tag-presence checks.
*Test strategy:* unit tests with mocked HTTP responses (reuse the mocking
style already used for `BenchmarkService`'s HTTP calls in
`tests/test_benchmark_service.py`) covering: daemon reachable + tag
present, daemon reachable + tag absent (startup error), daemon
unreachable. *Live validation on UBI:* install Ollama, `ollama pull` a
small model, point `config.yaml` at it, run `./backend health` and confirm
the reported fields.

**Increment 3 — `chat()` / `generate_text()`.**
Implement the `/api/chat` and `/api/generate` mappings, Section 1's
model-resolution decision (reject-with-404, `EngineUnavailableError` ->
`503`), and Section 4's token-usage mapping (including the `0`-fallback
and warning log for missing counts).
*Test strategy:* unit tests mocking Ollama JSON responses: happy path,
missing `prompt_eval_count`/`eval_count`, model-id mismatch (expect the 404
shape), daemon unreachable mid-call (expect 503). *Live validation on UBI:*
start the backend against the pulled model, `curl /v1/chat/completions`
with the correct and an incorrect model id, confirm token accounting and
the 404 behavior.

**Increment 4 — `unload_model()` mapping.**
Implement the `keep_alive: 0` request. Not wired to any live endpoint
(Non-goal, Section 5) — exists as a tested engine method only, matching
`TransformersEngine.unload_model()`'s current unused status.
*Test strategy:* unit test asserting the expected request payload is sent.

**Increment 5 (separate, code-adjacent) — OpenAPI amendments.**
Once Increment 3 lands with real daemon-down/model-mismatch behavior, file
the accompanying `openapi/backend-node.openapi.yaml` update (new error
schema for `model_not_found`, optional `requested_model` field, `/health`
status-value widening) in the *same* code increment that introduces the
behavior, per the `AGENTS.md` rule — not bundled into Increments 1-4, and
not part of this design document.

### Operator prerequisite: Ollama on UBI

Ollama installation on the UBI machine is an operator action, not something
this backend or its CLI automates (consistent with Non-goal: no
pulling/downloading via the backend, extended here to installation itself).
Manual verification commands before Increment 2's live validation:

```bash
ollama --version
ollama list
curl http://127.0.0.1:11434/api/tags
```

## 7. Risks and open questions

- **Ubuntu 18 compatibility.** The UBI machine runs Ubuntu 18 (per
  `AGENTS.md`, Environment). Current Ollama installer/release requirements
  (glibc version, kernel features) have not been verified against Ubuntu
  18.04's glibc 2.27. This must be checked on the actual UBI machine before
  Increment 2 begins. If incompatible, fallback options are limited: no
  containerization workaround is available without an explicit exception,
  since `AGENTS.md` and `docs/developed.md`'s Compatibility section both
  disallow introducing Docker. Building Ollama from source or using an
  older release are the remaining options if the current release doesn't
  run.
- **VRAM contention on the single RTX A4000 (16GB).** Section 3 keeps one
  engine active per backend *instance*, but nothing prevents an operator
  from running a Transformers-serving instance and an Ollama-serving
  instance side by side on the same physical GPU. Their combined VRAM
  footprint is a deployment-level risk this design does not solve;
  operators should not co-locate both without explicit VRAM budget
  planning until GPU selection/multi-GPU scheduling (later roadmap item)
  exists.
- **Chat templating differences.** Ollama applies its own model-specific
  template (from the Modelfile / GGUF metadata), which is not guaranteed to
  match the HF tokenizer's `apply_chat_template` output
  `TransformersEngine._prompt_from_messages()` uses for what is nominally
  "the same" model family — GGUF conversion pipelines can diverge from the
  original HF template. Switching a given model name between engines is
  therefore not guaranteed to produce identical prompts or outputs; Core
  and other consumers should not assume behavioral parity across engines
  for "the same" model, only contract parity (same API shape).
- **Ollama daemon connection configuration is undecided.** This design
  assumes the default `http://127.0.0.1:11434`, but does not specify
  whether that needs its own `config.yaml` field (e.g.
  `backend.ollama_host`) and env override, analogous to `HOST`/`PORT` for
  the backend itself. Open question for Increment 1.
- **Health-check cost.** Hitting `GET /api/tags` on every `/health` call
  adds a network round trip that `TransformersEngine.health()`'s in-process
  check doesn't have. Whether to cache/rate-limit this is an open question
  for Increment 2, not decided here.
- **Timeout behavior.** Network calls to the daemon can hang or time out
  in ways an in-process Transformers call cannot; whether `OllamaEngine`
  needs its own configurable timeout (distinct from any Transformers
  timeout behavior) is open for Increment 3.
