# Repository Audit — HEAD 87a844d

Read-only audit of `nemoclaw-backend` at commit `87a844d` ("fix: separate
reasoning models' thinking from the answer"), 53 commits after the last
release tag `v0.3.0`.

Scope: architecture compliance against `AGENTS.md`, contract/doc accuracy,
dead code, test architecture, and operational readiness. Live state on the
UBI Node was checked, not assumed.

## Findings

### HIGH — `model_runtime.py` is dead code that would load a second model

`model_runtime.py` is imported by nothing (`grep` over the whole tree
finds no `import model_runtime`). It is not merely unused: line 4 calls
`create_inference_service()` **at import time**, exactly as `api.py:20`
does. Anything that imports it — a future module, a test, a REPL session
— would construct a *second* `InferenceService`, which runs a second GPU
busy-check and a second `engine.load_model()`.

Under `engine: ollama` that is a wasted daemon round-trip; under
`engine: transformers` it is a second model load into VRAM. The facade
adds nothing: every function is a one-line delegation to
`InferenceService`, and callers already use the service or the API.

`docs/architecture.md:253` and `docs/developed.md:23` still describe it as
a live "thin compatibility facade", so the docs actively point future work
at a hazard.

**Action:** delete `model_runtime.py` and both doc references.

### HIGH — Docs and contract still describe `/admin/model/*` as `501` stubs

Phase 5 Increment 3 made these endpoints real, but three documents were
not updated in the same increment, which `AGENTS.md` requires:

- `docs/api-contract.md:30` — "its request/response shapes may … (currently
  501 stubs)"
- `docs/api-contract.md:59` — "`POST /admin/model/load|unload|switch`
  (implemented as 501 stubs)"
- `docs/developed.md:87,91` — "each returns HTTP `501` with a fixed JSON
  body", "`unload` takes no body and always returns `501`"
- `docs/architecture.md:165` — "admin operations are unstable and evolving
  (currently stub `501` …)"

All four statements are now false. `501` today means only "this engine does
not support runtime lifecycle" (`TransformersEngine`).

**Action:** correct all four. `docs/audit-2dabb09.md` may keep its `501`
references — it is a dated historical record.

### HIGH — `/health` is marked `partial` on a rationale that is no longer true

`openapi/backend-node.openapi.yaml` still carries:

> Today's only implementation (TransformersEngine, always-ready lifecycle)
> ever returns `"status": "ok"` — there is no code path that produces
> "degraded" or "unavailable" yet.

Both paths exist and are exercised. Verified live this session:
`"status":"unavailable"` while `lifecycle_state` is `unloaded`, and
`degraded` on `EngineUnavailableError`. `health_status_for_lifecycle_state()`
projects the whole enum and is unit-tested.

**Action:** move `/health` to `x-implementation-status: implemented` and
replace the stale `x-current-behavior`.

### MEDIUM — No release tag in 53 commits

Last tag is `v0.3.0` ("Modular backend architecture"). Since then, all
live-validated on real hardware: `OllamaEngine` Increments 1–4,
Ollama-on-UBI deployment, Phase 5 Increment 3 (real load/unload/switch),
shared-GPU ownership and safety, dynamic GPU pinning, `/v1/models`
discovery, and reasoning separation.

`AGENTS.md` says minor bumps mark backend capability or architecture
milestones, and that tags should mark validated runtime milestones. Several
have passed untagged.

**Action:** tag a minor release. `v0.4.0` is the honest floor; the
lifecycle and GPU-safety work arguably justify `v0.5.0`.

### MEDIUM — API layer is tested by reading its own source text

`tests/test_inference_service.py` contains four `Path("api.py").read_text()`
assertions that grep for route decorators and helper names instead of
issuing requests. Renaming a private helper breaks tests without any
behaviour changing, and no test covers actual routing, status codes, or
response bodies.

The root cause is structural and unchanged since `docs/audit-2dabb09.md`
flagged it: `api.py:20` builds `inference_service = create_inference_service()`
at import time, so importing `api.py` in a test loads a model. There is no
seam to inject a fake service.

Also blocking: `httpx` is absent on UBI, so `fastapi.testclient.TestClient`
cannot even be constructed there today (`RuntimeError: … requires the httpx
package`).

**Action:** introduce a dependency-injection seam (FastAPI `Depends`, or a
module-level setter used by `app.py`), add `httpx` to `requirements.txt`,
and replace the source-text assertions with real request tests. This is the
single largest structural weakness in the test suite.

### MEDIUM — `cli.py` is drifting toward owning policy

`cli.py` is 1381 lines — 32% of all source, three times the next largest
module — with 44 private helpers. `AGENTS.md` is explicit: "CLI commands
and FastAPI routes are delivery surfaces, never owners."

Most of it is legitimate formatting, but `_check_external_runtime_gpus()`
and `_check_gpu_before_start()` encode real policy: when to warn, when to
refuse, when to prompt, and what counts as a safe alternative. That is a
GPU-safety capability, and `GPUManager` is its natural owner; the CLI
should render the verdict, not compute it.

**Action:** extract the decision into a service (returning a structured
verdict the CLI formats). Not urgent — the logic is correct and tested —
but it will keep accreting if left.

### MEDIUM — Streaming is the only functional gap left for a chat UI

`POST /v1/chat/completions` with `"stream": true` returns `400`, which is
why it remains `x-implementation-status: partial`. Consequences:

- A 146-token reply arrives after one long pause instead of token by token.
- `BenchmarkService.first_token_latency` reports itself unavailable — the
  one place the codebase already refuses to fake a number.
- `docs/model-lifecycle-design.md` requires that active streams count as
  active requests for lifecycle draining, so it interacts directly with
  the request accounting added in Increment 3.

**Action:** implement SSE. This is the largest remaining piece of work and
the only one that changes what the product can do.

### LOW — Local and UBI environments diverge

`fastapi`, `torch` and `typer` are absent locally, so the suite runs 246
tests here and 255 on UBI, and `test_transformers_engine` always errors
locally with `ModuleNotFoundError: torch`. That error is easy to normalise
as "expected" and mask a genuine failure.

**Action:** skip rather than error when an optional dependency is missing
(`unittest.skipUnless`), so a red result always means something real.

## Solid

Re-verified, still true:

- No `shell=True` anywhere; no committed secrets.
- `requirements.txt` matches actual imports; no `pip freeze` drift. (Only
  `httpx` would need adding, and only for the test work above.)
- Ownership boundaries hold in the service/engine layers: `ModelManager`
  owns the catalog, engines own runtime facts, `InferenceService` composes.
  The `/v1/models` and reasoning-split work both respected this.
- No `TODO`/`FIXME`/`HACK` markers anywhere in the tree.
- The `ModelManager._replace_selected_model_line` ambiguity flagged in the
  previous audit is fixed — it now raises on ambiguous `id:` entries.
- Every capability shipped this session was live-validated on real
  hardware, and three real bugs were caught that way that unit tests had
  passed over (pre-flight degradation, lost `serve.log`, GPU UUID
  resolution).
- 255 tests pass on UBI.

## Recommended order

1. **Housekeeping** (small, removes active hazards): delete
   `model_runtime.py`, fix the four stale `501` claims, correct `/health`
   to `implemented`, skip-instead-of-error on missing optional deps.
2. **Tag a release** once (1) lands, so the tag marks a clean tree.
3. **Streaming (SSE)** — the only change that alters product capability.
   Do it before the test refactor, since it will add request-path tests of
   its own.
4. **API test seam** — dependency injection plus `httpx`, replacing the
   source-text assertions.
5. **Extract GPU policy out of `cli.py`** — lowest urgency; correctness is
   not in question, only ownership.

Deferred by prior explicit decision, not oversight: worker supervision
(`TransformersEngine` lifecycle, side-by-side switch), Backend Registry,
UBI OS/driver upgrade, and pinning the Ollama daemon's GPUs persistently.
