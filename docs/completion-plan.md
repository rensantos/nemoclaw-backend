# Completion Plan

Originally written 2026-08-03 at v0.6.0. **Updated end of 2026-08-03**
after Steps 1-3 were built and live-validated; the sections below are
rewritten to match reality rather than left describing a finished plan as
pending.

The headline still holds: **the backend is feature-complete for
inference, and the remaining work is mostly frontend and operational.**
Do not add backend surface without first checking the frontend needs it.

## Where things actually stand

| | State |
|---|---|
| `nemoclaw-backend` | 341 tests, no `partial` endpoints, live on UBI serving `qwen3:30b` |
| `nemoclaw-research-assistant` | streaming, node selection, model picker, `/doctor`, `/pull` - all live-validated |

Three nodes are configured and reachable: `local` (its own backend on
:8001 over local Ollama), `ubi` (over VPN + one SSH tunnel), and `claude`
(native Anthropic provider - authenticates, but the account has no
credits, so it has never completed a real paid request).

## Steps 1-3 - DONE (2026-08-03)

- **Streaming** end to end, with a reasoning model's hidden thinking shown
  live in its own Telegram message and never persisted to history or the
  markdown export.
- **Model picker**: `/models` lists what is genuinely installed (numbered,
  selectable by number), `/pull` lists what could be downloaded and
  refuses on insufficient disk.
- **Node selection**: one `/node` choice, `NEMOCLAW_LOCAL_OLLAMA_HOST`
  pinning helper models locally so only ONE tunnel is needed for UBI.

Built beyond the original plan: `/doctor`, `scripts/setup.sh`, a committed
SearXNG setup, `ubi_connect.sh`/`ubi_disconnect.sh`, a native
`AnthropicProvider`, `GET /resources`, and `POST /admin/model/pull` with
hard disk-safety refusal.

## What is next

The frontend's `docs/NEXT_SESSION.md` is the working task list. In short,
highest value first:

1. **Use the assistant for real and fix what breaks.** Every bug found on
   2026-08-03 came from live use, none from the test suites. `/rag`,
   `/evidence`, `/literature`, `/hybrid`, PDF upload and `/metrics` have
   had zero live exercise.
2. **Make the bot and local backend survive a reboot** - both are plain
   `nohup` processes today, with no cron or systemd entry.
3. **Project-scoped chat memory** (`/project <name>`), designed in
   `NEMOCLAW_SETUP.md`.
4. **Claude credits**, so that provider gets its first real verification.

### Backend-side, only on a real caller

- **`/v1/embeddings`** is the one thing that would let the research
  subprocess scripts stop calling Ollama directly - the last
  Core-talks-to-runtime violation. Tracked in `docs/future-tasks.md`.
  Not urgent: embeddings are pinned local and work.
- **The unmet Streaming Assumptions clause**: past the drain timeout an
  open stream is left to finish rather than closed with a controlled
  error. Fine while transitions are fast and single-operator.
- **Six remaining `planned` endpoints**: `/lifecycle`, `/capabilities`,
  `/engines`, `/metrics`, `/benchmarks`, `/v1/completions`. Build only on
  a real caller. (`/gpu` is effectively superseded by `/resources`.)
- **Worker supervision**, which would unlock `TransformersEngine`
  lifecycle and side-by-side switching. Unnecessary while Ollama is the
  deployment.

### Versioning

`v0.7.0` is the current backend tag. The original 1.0 trigger - "the
frontend actually consumes streaming and the `/v1/models` flags" - is now
met, so 1.0 is unblocked on its own terms. Deliberately **not** tagged
yet: 2026-08-03 surfaced a long run of bugs that only real use exposed,
and `AGENTS.md`'s own rule is that tags mark validated runtime
milestones. Revisit after task 1 above.

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
  produced its status code. 2026-08-03 added six more, all found by a
  person using the bot and none by the suites: a long-running process
  ignoring `/node` because config was frozen at import; `/deepweb` and
  `/web` validating a stale per-role model against the wrong daemon;
  `/web` skipping its own fallback because the CLI was absent rather than
  failing; `/models` listing six absent models while hiding two installed
  ones; `/model` failing only on the persist path, because a lenient test
  fake hid a second catalog check; and `register_model()` deleting 114
  comment lines from `config.yaml`. Assume the untested paths hold more.

## Not on the critical path

The **UBI OS/driver upgrade** (Ubuntu 24.04) is agreed but unscheduled. It
would remove the Qwen3.5/gemma4 Ollama version gate, the
`transformers`/Qwen3 dead end and broken `bitsandbytes` at once — but
nothing above depends on it.
