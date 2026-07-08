# API Contract

This is a human-readable companion to
[`openapi/backend-node.openapi.yaml`](../openapi/backend-node.openapi.yaml),
the authoritative contract for one Nemoclaw Backend instance. It does not
duplicate endpoint schemas — read the YAML for request/response shapes. This
file explains the shape of the contract and the decisions behind it.

## The two-tier model

- **Tier 1 — OpenAI-compatible surface** (`/v1/*`): lets any
  OpenAI-compatible client (Nemoclaw Core, third-party tooling, the
  benchmark harness) talk to a backend instance without knowing anything
  Nemoclaw-specific. This is the interoperability surface.
- **Tier 2 — Native management surface** (`/health`, `/lifecycle`,
  `/admin/model/*`, `/capabilities`, `/engines`, `/metrics`, `/gpu`,
  `/benchmarks`, `/generate`): Nemoclaw-specific power — runtime status,
  lifecycle control, discovery, and observability that no OpenAI client
  needs and no OpenAI contract defines.

## Stability guarantees per tier

- `/v1/*` is stable. No breaking changes without explicit architectural
  approval (AGENTS.md, API Stability).
- Native **read** endpoints (`/health`, `/lifecycle`, `/capabilities`,
  `/engines`, `/metrics`, `/gpu`, `/benchmarks`) are stable once
  implemented — they are status/discovery surfaces, not control surfaces,
  so their response shapes should not need to change shape once shipped.
- `/admin/*` carries **no stability guarantee**. It is an unstable control
  surface by design (currently 501 stubs); its request/response shapes may
  change as real lifecycle behavior lands.
- `/generate` is implemented but deprecated; it exists only for old local
  callers and should not gain new consumers.

## The architectural rule

> Core never talks directly to inference runtimes. Core talks to Backend.
> Backend talks to inference runtimes.

Every endpoint in this contract exists to make that rule enforceable:
Nemoclaw Core (and any future Backend Registry or engine adapter) has a
complete, self-describing surface to code against without reaching past
Backend into Transformers, CUDA, or any other runtime internals.

## The independence test

A Backend instance must remain fully usable without Core: the OpenAI API
(Tier 1), the CLI (`./backend ...`), and the native API (Tier 2) must each
work standalone. If a feature only makes sense when Core is present, it does
not belong in this contract — it belongs in Core.

## `/lifecycle` vs `/admin/model/*` — decision record

Both surfaces touch lifecycle state, so the split is deliberate:

- `GET /lifecycle` (planned) is **observability/status only**. It returns
  the current `LifecycleState` and related read-only metadata. It performs
  no transitions and never will.
- `POST /admin/model/load|unload|switch` (implemented as 501 stubs) is the
  **authoritative lifecycle control surface** — the only place transitions
  happen once real behavior lands. This supersedes the Nemoclaw system
  spec's flat `/models/load`, `/models/unload`, `/engines/switch` paths
  (spec §6.5); see commit `61526a9` and `docs/architecture.md`'s "Admin
  surface vs spec" subsection for the full rationale.

Do not add write/transition semantics to `/lifecycle`. Do not add a second
read-only status endpoint under `/admin/`.

## `x-implementation-status` legend

Every path in `openapi/backend-node.openapi.yaml` carries one of:

| Status | Meaning |
| --- | --- |
| `implemented` | Ships today with the documented behavior. |
| `partial` | Ships today but with a documented gap between contract and current behavior (e.g. `stream` on `/v1/chat/completions`). |
| `stub` | Endpoint exists and returns a fixed response (e.g. 501), but performs no real work. Carries no stability guarantee. |
| `planned` | Committed direction; no route exists yet. Calling it returns 404 today. |
| `unscheduled` | Described for forward-compatibility only; not on the roadmap. |

`docs/registration-schema.json` (Backend Registry self-registration, spec
§7) is a forward contract in the same sense as `planned`/`unscheduled`
endpoints: nothing in this repo implements it yet.
