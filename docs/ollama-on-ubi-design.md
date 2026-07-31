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

**Reboot survival**: a `crontab -e` `@reboot` entry running the same command
survives an actual UBI reboot without needing root (`cron` itself is a
user-level facility). An idle Ollama daemon holds no GPU memory until a
request actually loads a model, so this is safe to leave in place.

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
