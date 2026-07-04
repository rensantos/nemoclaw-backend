# Repository Audit — HEAD 2dabb09

Read-only audit of `nemoclaw-backend` at commit `2dabb09` ("Add
/admin/model/* lifecycle stub endpoints and CLI commands (Phase 5
Increment 2)").

## Findings

### MEDIUM — `/admin/model/load` and `/admin/model/switch` can return 422, not always 501

`api.py:68-80` binds `admin_model_load` and `admin_model_switch` to a
required `ModelLifecycleRequest` body (`model_id: str`, `schemas.py:27-28`).
FastAPI validates the request body before the handler body runs, so a
missing or malformed body (no `model_id`, wrong type, invalid JSON, no
body at all) short-circuits to FastAPI's standard `422 Unprocessable
Entity` — never reaching `lifecycle_stub_response()`. This contradicts
the documented claim that these endpoints "always return HTTP 501".

`/admin/model/unload` (`api.py:73-75`) takes no request body and is
unaffected: it always returns `501` regardless of what is sent.

Fixed in this audit by correcting the documentation
(`docs/model-lifecycle-design.md`, `AGENTS.md`) to state the actual
behavior, per instructions. `api.py` was not changed.

### MEDIUM — `ModelManager._replace_selected_model_line` can silently mutate the wrong `id:` line

`services/model.py:128-161` looks for the first `id:` line inside the
`model:` block using indentation comparison only (`indent <=
model_indent` ends the block). It does not distinguish `model.id` from
an `id:` key belonging to `model.available[0]`.

If `available:` is declared before `id:` in the `model:` block, and the
first `available` entry's own `id:` line sits on a continuation line
whose indentation is still greater than `model_indent` (which any
mapping entry indented under `available:` will be), that `id:` line is
matched first. `select_model()` then rewrites
`model.available[0].id` instead of `model.id`, while `validate_model()`
and the caller both believe the selected model id changed. No exception
is raised — the call reports success silently with the wrong model
selected.

Example config that triggers it:

```yaml
model:
  available:
    - id: TinyLlama/TinyLlama-1.1B-Chat-v1.0
      path: /models/tinyllama
  id: TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Calling `select_model("some-other-model")` rewrites the `available[0].id`
line, not `model.id`.

### LOW — API-layer tests are source-text checks, not behavioral request tests

The existing API test coverage greps `api.py` for route decorators and
structure rather than issuing requests through FastAPI's `TestClient`
and asserting on status codes/bodies. This is a pre-existing pattern,
not introduced by this commit — `api.py` eagerly constructs
`inference_service = create_inference_service()` at import time
(`api.py:13`), which loads the configured Transformers model, so
importing `api.py` in a test process without a GPU/model available is
expensive or unsafe. Behavioral request tests would need a way to
inject a fake `InferenceService` before import, which does not exist
yet.

### Doc drift

`docs/architecture.md` and `docs/developed.md` did not mention Phase 5
Increments 1-2 (`LifecycleState`, `services/lifecycle.py`,
`lifecycle_state` in `/health`, `/admin/model/*` stub endpoints).
`docs/future-tasks.md` still listed lifecycle load/unload/switch as
purely future work, with no mention that the stub command/endpoint
surface already exists. Fixed in this audit; see below.

## Solid

- No `shell=True` subprocess calls found anywhere in the codebase.
- No secrets, credentials, or tokens committed.
- `requirements.txt` matches actual imports exactly; no unused
  dependencies, no `pip freeze` drift.
- No unused imports found.
- `.gitignore` correctly excludes runtime artifacts (`run/`, `logs/`,
  virtualenvs, etc.).
- Full test suite: 53/53 passing, no warnings.
- Stub design (Increment 2) is internally coherent: fixed response
  body, no engine/CUDA calls, `lifecycle_state` read-only.

## Caveat

This audit was performed in a sandbox without `fastapi`, `torch`, or a
GPU available. Static reading of `api.py`, `schemas.py`, and
`services/model.py` is the basis for the findings above; live
request/response behavior (the 422 vs. 501 split, and the YAML
mutation bug under a real config file) remains verified only by
inspection here, not by execution. Confirm both against a running
instance on UBI before relying on them operationally.
