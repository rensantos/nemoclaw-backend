# Ollama-on-UBI Deployment Design

This is a design document for a runtime-state/process-management change, per
`AGENTS.md`'s rule that such phases need a design doc before implementation.
It covers deploying an Ollama daemon directly on the UBI Backend Node
(alongside, not replacing, the existing `TransformersEngine` capability) and
switching UBI's live-served model to it.

## Why

`docs/problems.md`'s multi-family compatibility sweep and the follow-on Qwen
investigation established a hard ceiling for `TransformersEngine` on UBI:
the pinned driver (470.86, CUDA 11.4 max, no sudo to upgrade) caps
`transformers` at `4.49.0` before `torch.compiler` calls break under the
pinned `torch==2.0.1+cu117`, and Qwen3 support only landed in `transformers`
at `4.51.0` — a confirmed dead end, not a "not yet tried". (Qwen2/Qwen2.5
*do* work at `4.37.0` with a `bfloat16` load-dtype fix, but that's a smaller
win than actual Qwen3 support.)

Separately validated (throwaway scratch testing in `~/ollama-test/` on UBI,
not part of this repo): Ollama runs with genuine GPU acceleration directly
on UBI under the same driver, because it's a structurally different runtime
(llama.cpp/GGML, not PyTorch) with its own bundled CUDA libraries and much
looser driver-version coupling. This sidesteps the `transformers`/`torch`
wall entirely. `OllamaEngine` (`engines/ollama_engine.py`) already
implements every `InferenceEngine` method and was previously only ever
validated against the user's own laptop (the "Local Node" per
`docs/architecture.md`) — nothing in the code assumes that; it just talks to
whatever `backend.ollama_host` points at.

## What's being deployed

- **Ollama `v0.9.2`** — the newest release satisfying two independently
  binary-searched constraints on UBI: glibc (Ubuntu 18.04 has `2.27`;
  releases from `v0.10.0` on need `2.28`) and the bundled `cuda_v11` CUDA
  runtime (driver 470.86 maxes at CUDA 11.4; `v0.9.3` onward ship CUDA
  12-only libraries). Installed from the standalone `ollama-linux-amd64.tgz`
  release tarball — **not** the standard `curl | sh` installer, which needs
  root (installs a systemd unit) and isn't available on the `d3894` account.
- **Model: `qwen3:8b`.** Chosen over `qwen3:1.7b` (already validated
  end-to-end during the scratch testing, coherent output, real GPU usage
  confirmed via `nvidia-smi`) for meaningfully better quality, and over
  larger Qwen3 sizes (`14b`, `30b-a3b`) to stay conservative on UBI's disk —
  this is a shared box already at ~98% capacity system-wide with only
  ~26GB free at time of writing. `qwen3:8b`'s Q4 GGUF weights are
  ~5-6GB, leaving real headroom.
- **No root required anywhere in this deployment.**

## Install layout

```
~/ollama/
  bin/ollama          # v0.9.2 binary
  lib/ollama/...       # bundled cuda_v11 + cuda_v12 + cpu backends
  models/               # OLLAMA_MODELS - pulled model blobs live here
```

Moved from the scratch `~/ollama-test/v092/` validation directory to this
permanent location.

## Startup / process management

No systemd (no root). Two supported ways to start it, both documented here
rather than wrapped in a committed script, matching this repo's existing
convention of documenting exact UBI commands in prose (see
`docs/problems.md`'s pip install commands) rather than shipping host-specific
shell scripts into the Python package:

**Manual** (mirrors `./backend start`'s own manual-start convention — this
backend has no auto-start-on-boot today either):

```bash
cd ~/ollama && CUDA_VISIBLE_DEVICES=2,3 OLLAMA_MODELS=~/ollama/models \
  nohup ./bin/ollama serve > ~/ollama/serve.log 2>&1 < /dev/null &
```

`CUDA_VISIBLE_DEVICES=2,3` matches `config.yaml`'s existing `backend.gpu`
convention of only touching the two GPUs that aren't another user's shared
job — check `nvidia-smi` before trusting this is still current, per
`docs/problems.md`.

> **Known drift, found 2026-08-02 — the deployed daemon does not match
> this.** The actual UBI daemon (and its `@reboot` crontab entry) runs
> with `CUDA_VISIBLE_DEVICES=0,1,2,3`, i.e. it can reach *all four* GPUs
> including another user's. `backend.gpu: "2,3"` does **not** constrain
> it: under `engine: ollama` the backend never loads a model, so the
> daemon's own visible-device set is the only thing that decides where
> weights land. Ollama had been choosing GPU 2/3 because its scheduler
> picks by free VRAM — luck, not enforcement.
>
> `./backend start` now detects this (reads the daemon's
> `/proc/<pid>/environ`) and refuses to start whenever the daemon can
> reach a GPU someone else is using, printing the pinned restart command.
> The backend does not restart the daemon itself — that stays an operator
> action, and it must be fixed in **both** places: the running process and
> the `@reboot` crontab line.

**Reboot survival**: a `crontab -e` `@reboot` entry running the same command
survives an actual UBI reboot without needing root (`cron` itself is a
user-level facility). An idle Ollama daemon holds no GPU memory until a
request actually loads a model, so this is safe to leave in place. Note the
deployed entry currently carries the `0,1,2,3` drift described above.

Stopping it: find the PID (`ps -eo pid,cmd | grep '[o]llama serve'`) and
`kill <pid>` directly. **Do not use `pkill -f` for this** — confirmed during
this session's validation that `pkill -f` against an Ollama server process
launched from a prior SSH session unexpectedly killed the current SSH
session too (root cause not fully diagnosed; avoid the pattern rather than
rely on understanding it).

## Config change

`config/config.yaml`:
- `backend.engine: transformers` -> `ollama` (single-active-engine design,
  per `docs/ollama-engine-design.md` Section 3 — this replaces, not
  supplements, the currently-configured live engine for this backend
  instance)
- `backend.ollama_host` stays default (`http://127.0.0.1:11434`) — the
  daemon runs on UBI itself now, same host as the backend process, so no
  tunnel or remote host needed
- `model.id: qwen3:8b`
- `model.available` gets `qwen3:8b` (and optionally `qwen3:1.7b` as a
  lighter validated fallback) with `engine: ollama`

`TransformersEngine`'s config (quantization, revision, the existing
`available` entries) is left intact, not deleted — switching back to
`engine: transformers` remains a one-line revert if needed.

## Rollback

Set `backend.engine` back to `transformers` in `config/config.yaml` and
restart the backend. The Ollama daemon can keep running (harmless if
unused) or be killed by PID as above.

## Out of scope for this increment

- No CLI wiring for Ollama-daemon lifecycle (`./backend` does not gain new
  commands to start/stop/status the daemon) — `OllamaEngine` already
  documents that this backend doesn't own Ollama's CUDA context
  (`AGENTS.md`'s OllamaEngine Increment 2 notes), and that boundary holds
  whether the daemon is on the Local Node or on UBI.
- No multi-model / model-switching UI work.

## Update 2026-07-31 (same day): bigger Qwen3 sizes, live default changed

The "future increment if `qwen3:8b` proves out" above happened the same
day. `qwen3:30b-a3b` (MoE, ~18GB) and `qwen3:32b` (dense, ~20GB) were both
pulled and validated — Ollama tensor-splits either one across GPU 2+3
automatically (no config change beyond the `CUDA_VISIBLE_DEVICES=2,3`
already set for the daemon): `qwen3:32b` landed at ~12.3GB/~12.3GB,
comfortable under 16GB per card; `qwen3:30b-a3b` at ~10.6GB/~10.2GB.

**`model.id` is now `qwen3:32b`** — the best-quality option validated so
far, chosen as the live default. `30b-a3b` and `32b` do not both fit
UBI's disk at once (~38GB combined against a shared box that started
this whole investigation at ~26GB free); `30b-a3b` was deleted
(`ollama rm qwen3:30b-a3b`) before pulling `32b`. **After pulling
`32b`, UBI had only ~2GB free** — tighter than is comfortable on a
shared box. Anyone picking this up again: run `df -h` before pulling
anything else, and know that `30b-a3b` is validated-but-not-currently-
pulled, listed in `config/config.yaml`'s `model.available` for exactly
that reason (repull with a plain `ollama pull qwen3:30b-a3b` if wanted,
but something else likely needs to be deleted first to make room).

## Update 2026-07-31 (same day): model family compatibility + native context windows

`qwen3`'s native context window (`qwen3.context_length` in its own GGUF
metadata, checked via `ollama show`/`/api/show`) is **40960** tokens -
not the huge number `config.yaml`'s `model.think_default` investigation
briefly tried before finding out. This is an architectural ceiling, not
a memory/config one: requesting more (`n_ctx_per_seq > n_ctx_train`)
produces a real warning in the daemon log and degraded output, not just
higher memory cost - see `NEMOCLAW_SETUP.md` in
`nemoclaw-research-assistant` for the fuller memory-cost investigation
this same session that this finding grew out of.

**Which model families does this pinned Ollama (`v0.9.2`) actually
support?** Tested by attempting a real pull for each (the version-gate
check happens at the manifest stage, before any weight download, so
this is cheap even without fully downloading each one):

| Family | Pull result |
|---|---|
| `qwen3` | works (already deployed) |
| `gemma3` | works - pulled `gemma3:4b` fully, loaded, generated coherently |
| `llama3.1`, `llama3.2`, `mistral`, `deepseek-r1`, `phi4`, `codellama`, `qwen2.5`, `command-r`, `hermes3` | passed the manifest/version check (real download started); not individually pulled fully or load-tested this session |
| `gemma4` | **blocked** - `412: requires a newer version of Ollama`. Its manifest shows a vision projector layer (`mmproj-gemma-4-12B-it-bf16.gguf`) - it's multimodal, needing Ollama's newer engine this pinned version doesn't have. Same category of wall as the `transformers`/Qwen3 story (`docs/problems.md`), just on Ollama's side this time. |

**Also confirmed working this session:** `ollama pull hf.co/<repo>:<quant>`
pulls GGUF files directly from Hugging Face, not just Ollama's own
curated library - tested against `bartowski/Qwen2.5-7B-Instruct-GGUF`,
real download at ~100MB/s, no version block. Opens up any GGUF
quantization published on HF, not just the ~100 models Ollama curates,
as long as the architecture doesn't need the newer multimodal engine.

**Native context windows** - `qwen3` and `gemma3:4b` verified directly
via UBI's own pulled copies (`/api/show`); the rest are the base
architecture's published HuggingFace `config.json` value (same
underlying models Ollama packages, not independently re-verified
against UBI's specific GGUF file):

| Model | Native context | Verified how |
|---|---|---|
| `qwen3` (32b/8b/1.7b) | 40,960 | on UBI |
| `gemma3` (4b) | 131,072 | on UBI |
| `llama3.2` (3b) | 131,072 | on UBI |
| `llama3.2-vision` (11b) | 131,072 | on UBI |
| `llama3.1` | 131,072 | HF config |
| `deepseek-r1` (Llama-8B distill) | 131,072 | HF config |
| `hermes3` | 131,072 | HF config (Llama-3.1 base) |
| `mistral` (v0.3) | 32,768 | HF config |
| `qwen2.5` | 32,768 | HF config |
| `phi4` | 16,384 | HF config |
| `codellama` | 16,384 | HF config |
| `command-r` | ~128,000 | publicly documented; HF config is gated, couldn't fetch |
| `gemma4` | unknown | couldn't pull at all |

## Update 2026-08-01: live default switched to llama3.2-vision:11b

`qwen3:32b`'s native context (40,960) is far short of the `~131K` several
other confirmed-working families offer, so it was replaced. `llama3.2`
also has a separate multimodal sibling, `llama3.2-vision` (11b, 90b) -
`90b` is 54.6GB, infeasible on UBI's disk regardless of anything else,
but **`11b` (7.8GB) pulled successfully and works** - not blocked the
way `gemma4` was, likely because it uses Meta's own `mllama`
architecture rather than whatever newer generic multimodal engine
`gemma4` needs. Text-only use (the only thing tested/needed here) works
cleanly: coherent output, **all 41/41 layers on a single GPU** (~12.1GB,
notably lighter than `qwen3:32b` needing 2+ GPUs), 131,072 context
confirmed via its own metadata.

`model.id` is now `llama3.2-vision:11b`, live-validated via `/health`,
`/v1/models`, and a real `/v1/chat/completions` call. All previously
pulled models (`qwen3:32b`, `gemma3:4b`, `llama3.2:3b`) were deleted to
make room during this exploration and are not currently pulled - all
remain validated options, listed in `config/config.yaml`'s
`model.available` for exactly that reason.

**Disk state after this exploration:** UBI had ~18GB free after
`llama3.2-vision:11b` was pulled (started this update from ~24GB free
after deleting every other model) - check `df -h` before pulling
anything else on this shared box.

## Update 2026-08-01 (same day): live default switched to qwen3:30b; qwen3.5 blocked

Tried `qwen3.5:35b` (24GB, multimodal) first, aiming for its 256K
context - **blocked**, `412: requires a newer version of Ollama`,
confirmed via a direct pull attempt (fails at the manifest stage, no
weight download). Same wall as `gemma4` above; this generalizes the
finding from "a `gemma4` quirk" to "any architecture needing Ollama's
newer generic multimodal engine is blocked on this pinned `v0.9.2`."
Root cause is UBI's glibc 2.27 (Ubuntu 18.04) + driver 470.86 (CUDA
11.4 max) - the same ceiling already blocking Qwen3 under
`TransformersEngine` (`docs/problems.md`) and `bitsandbytes`. A real
fix needs an OS/driver upgrade (agreed in principle with the user,
Ubuntu 24.04 + accompanying driver reinstall - RTX A4000/Ampere has no
hardware blocker against a current driver) - not yet scheduled.

Pivoted to `qwen3:30b` (MoE, `qwen3moe` architecture) instead: **262,144
native context**, confirmed via `ollama show qwen3:30b` directly on
UBI - much larger than dense `qwen3` (32b/8b/1.7b, 40,960; these are
genuinely different context ceilings despite the shared model name).
`llama3.2-vision:11b` was deleted to make room. `model.id` is now
`qwen3:30b`, live-validated via `/health`, `/v1/models`, a real
`/v1/chat/completions` call, and the full test suite.

**Found and documented, not fixed:** `qwen3:30b` does not honor
`"think": false` - verified by calling the raw Ollama daemon directly
(bypassing this backend entirely), reasoning trace still appears
regardless. Likely this MoE variant's chat template lacks the
enable/disable-thinking conditional dense `qwen3` has. Not a bug in
`OllamaEngine._apply_think()` - there is no request-side workaround.
`think_default: false` is kept (harmless, correct for other models),
but on this tag every response carries a ~48-token reasoning preamble
regardless. `max_tokens_default: 256` absorbs it fine, so nothing
truncates - unlike the original `qwen3:32b` bug this field was built to
prevent.

**Disk state:** with only `qwen3:30b` pulled (~18GB), UBI is at **~7.6GB
free, 100% used** - the tightest this project's disk has been. User has
adopted a single-model-at-a-time policy on UBI as a direct consequence;
do not pull a second model without deleting this one first, and check
`df -h` regardless.
