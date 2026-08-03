# Completion Plan

Written 2026-08-03, at `nemoclaw-backend` v0.6.0. This is the plan to
**finish the Nemoclaw project**, not just the backend.

The headline: **the backend is feature-complete for inference. Almost all
remaining work is in the frontend**, which uses none of what was built on
2026-08-03. Do not add backend surface without first checking the frontend
needs it.

## Where things actually stand

| | State |
|---|---|
| `nemoclaw-backend` | v0.6.0, 302 tests, no `partial` endpoints, live on UBI serving `qwen3:30b` |
| `nemoclaw-research-assistant` | commit `08b5ee4` — still non-streaming, ignores the new model flags, node choice still two hand-synced `.env` settings |

The frontend's `OpenAICompatibleProvider.chat()` explicitly **pops `stream`
off its kwargs and discards it**, and `models()` returns bare ids,
throwing away the `loaded`/`pulled`/`size_mib`/`fits` fields added for a
picker. Both verified by reading the source, not assumed.

One thing already works with no frontend change: the reasoning fix. The
provider reads `choices[0].message.content`, which the backend now
guarantees is the answer alone.

---

## Step 1 — Frontend streaming

**Why first:** the largest user-visible improvement. A 400-token answer
currently arrives after one long pause.

**Where:** `nemoclaw-research-assistant/scripts/providers/`.

**The catch:** `base.py`'s `LLMProvider` has no streaming method. Adding
one to `openai_compatible_provider.py` alone breaks the abstraction —
`ollama_provider.py` must implement it too (Ollama's native NDJSON, which
the backend already wraps; mirror `OllamaEngine._post_stream`).

**Acceptance:**
- A `chat_stream()` (or equivalent) on the base class, implemented by both
  providers.
- The OpenAI-compatible one parses `data: {...}` SSE and stops at
  `data: [DONE]`.
- `delta.reasoning` is kept separate from `delta.content` — do not
  concatenate them, or the reasoning fix is undone at the client.
- Token counts read off the final chunk's `usage`.
- The Telegram path streams or, if editing messages live is impractical,
  at minimum uses streaming to show progress.
- `scripts/test_llm_provider.py` extended to cover it.

## Step 2 — Model picker from `/v1/models`

**Why:** the backend already returns everything needed; the frontend
discards it. Small change, immediately useful.

**Acceptance:**
- `models()` returns the full objects, not just ids.
- The UI (Telegram command) lists models and marks unusable ones —
  `pulled: false` cannot be selected, `fits: false` warns.
- Selecting one calls `POST /admin/model/switch` with `"persist": true`.
  **Persist matters:** the backend sizes `gpu pin-free` from
  `config.model.id`, so an unpersisted switch leaves that stale.
- A `409 model_unavailable` or `404 model_not_configured` is shown to the
  user, not swallowed.

## Step 3 — Node selection (Local vs UBI)

**Why:** the largest item and the longest-standing want. Currently two
static `.env` choices that must be kept consistent by hand.

**The hard part** (see `NEMOCLAW_SETUP.md`'s "Future development"
section): there are **two independent wiring paths**, and one choice must
drive both —

1. the chat/Telegram path via `scripts/providers/` (OpenAI-compatible HTTP)
2. every subprocess script (`/deepweb`, `/evidence`, `/literature`) which
   talks **native Ollama** via `PROJECT_OLLAMA_HOST`, bypassing the
   backend entirely

**Constraint:** the backend has no auth and binds `127.0.0.1`, so UBI is
reached through `ssh -L 8000:127.0.0.1:8000 ubi-a4000`. Supporting both
nodes at once means **two tunnels on different local ports**, not one.

**Acceptance:**
- One user-facing choice (per session or per request) that both paths obey.
- Per-node model and `num_ctx` defaults, since the nodes have different
  models and context windows.
- Clear failure when the chosen node's tunnel is down — not a hang.
- Decide explicitly whether this needs the backend's deferred **Backend
  Registry**, or whether frontend-side config is enough. Registry work
  should not start until that question is answered.

## Step 4 — Optional: reasoning panel

Pure upside. Surface the `reasoning` field behind a toggle so the user can
see a model's thinking without it polluting the answer. Skip if time is
short; the frontend is already correct without it.

## Step 5 — Backend odds and ends, only if wanted

None of these is required for a working assistant:

- **The one unmet Streaming Assumptions clause**: past the drain timeout an
  open stream is left to finish rather than closed with a controlled
  error. Fine while transitions are fast and single-operator.
- **Seven `planned` endpoints**: `/lifecycle`, `/capabilities`, `/gpu`,
  `/engines`, `/metrics`, `/benchmarks`, `/v1/completions`. Build only on
  a real caller.
- **Worker supervision**, which would unlock `TransformersEngine`
  lifecycle and side-by-side switching. Unnecessary while Ollama is the
  deployment.

---

## Standing constraints — do not violate these

- **UBI disk is ~8 GB free, 100% used.** `df -h` before pulling anything;
  single model at a time.
- **The GPUs are shared with other researchers.** `docs/problems.md` and
  the GPU safety rules in `services/gpu_safety.py` are non-negotiable.
  Check `nvidia-smi` before and after.
- **`AGENTS.md`'s rule**: tests pass, docs updated *in the same
  increment*, commit and push. Work is not done until pushed.
- **Live-validate on real hardware.** Every session so far has found bugs
  that unit tests passed over — pre-flight degradation, a lost
  `serve.log`, GPU UUIDs, a streaming rejection that could never have
  produced its status code.

## Not on the critical path

The **UBI OS/driver upgrade** (Ubuntu 24.04) is agreed but unscheduled. It
would remove the Qwen3.5/gemma4 Ollama version gate, the
`transformers`/Qwen3 dead end and broken `bitsandbytes` at once — but
nothing above depends on it.
