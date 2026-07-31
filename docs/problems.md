# Known Problems

## UBI's old NVIDIA driver caps `torch`/`transformers` versions — do not `pip install -U`

Discovered 2026-07-30 during live model-loading validation. **Do not run
`pip install -U torch`, `pip install -U transformers`, or `pip install -U
accelerate` on UBI without reading this first** — it will break GPU
inference, and the failure mode is confusing (several unrelated-looking
errors in sequence, not one clear "wrong version" message).

### Root cause

UBI's installed NVIDIA driver is `470.86`, which supports CUDA **up to
11.4** only (`nvidia-smi` reports "CUDA Version: 11.4"). This account has
no `sudo`/admin access, so the driver cannot be upgraded from here — a
real driver upgrade would need whoever administers the UBI machine.

- PyTorch's official wheel index stopped offering CUDA <= 11.7 builds at
  `torch` 2.1.0; every `torch >= 2.1` release only ships CUDA 11.8+
  builds, and CUDA 11.8 itself requires driver >= 520.61 — well above
  470.86. So **no `torch >= 2.1` build can run on this GPU**, full stop.
- `transformers >= 4.51.0` (needed for Qwen3) requires `torch >= 2.1`.
  Worse: even `transformers` 4.51.0's *shared* modeling code (used by
  every architecture, not just Qwen3 — confirmed by reproducing the same
  failure loading plain `TinyLlama`/Llama) already calls
  `torch.compiler.*`, an API that doesn't exist before `torch` 2.1. So
  the ceiling isn't "avoid Qwen3" — it's "any sufficiently recent
  `transformers` release is incompatible with this driver, regardless of
  which model you try to load."

### The working combination (confirmed live, 2026-07-30)

```
torch==2.0.1+cu117   # pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
transformers==4.36.0
```

`accelerate` and `bitsandbytes` were not independently version-pinned or
re-validated against this exact combination — if either gets upgraded
later and breaks something, suspect the same `torch` ceiling first.

Two more, unrelated compatibility issues surfaced and were fixed in the
same session, worth knowing about even though they're not the `torch`
ceiling:

- **`torchvision`**: a stray, already-broken install (mismatched build,
  couldn't even load its own compiled extension) was crashing
  `transformers`' import chain via `PIL`. Fixed by uninstalling it
  entirely — nothing in this codebase needs it (text-only inference); it
  was never a declared dependency, just an environment leftover.
- **`Pillow`**: versions >= ~9.2 bundle a `libLerc` shared library built
  against a newer C++ toolchain than Ubuntu 18's system `libstdc++`
  provides (`GLIBCXX_3.4.29` required, `GLIBCXX_3.4.25` is the actual
  ceiling — check with `strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6
  | grep GLIBCXX | sort -V | tail`). Fixed by pinning `Pillow==9.0.1`.

### Practical consequence for model choice

Because `transformers` is capped at ~4.36.0 on UBI, **only architectures
already supported by transformers as of that release work** — proven:
Llama-family (`TinyLlama`, confirmed working end to end). Not supported:
Qwen3 (needs 4.51+), Gemma/Gemma 2/3/4 (needs 4.38+), and likely most
architectures added after ~late 2023. When picking a model for UBI,
check when transformers added support for its architecture before
assuming it'll load — this bit us twice in one session (Gemma 4, then
Qwen3) before TinyLlama (Llama architecture, always worked) confirmed the
real constraint.

**Update (2026-07-30, same session):** architecture support isn't the
only moving target. `mistralai/Mistral-7B-Instruct-v0.2` — an
architecture `transformers` 4.36.0 does support — still failed to load:

```
Exception: data did not match any variant of untagged enum PyPreTokenizerTypeWrapper at line 40 column 3
```

The model repo's `tokenizer.json` on the Hub was re-saved at some point
by a newer `tokenizers` library than the old, pinned one here can parse
— a file-format compatibility problem, independent of whether the model
*architecture* is old enough. Hub files can be silently updated over
time even for long-established models.

**Resolved (2026-07-31):** confirmed via `huggingface_hub.HfApi.
list_repo_commits()` that this repo's commit `9925900` ("[AUTO] CVST
Tokenizer Badger", 2024-06-24) is almost certainly the re-save. Pinning
`revision=dca6e4b60aca009ed25ffa70c9bb65e46960a573` (the original "Add
v0.2" commit, 2023-12-11, predating it) loads cleanly under the pinned
stack — verified both as a standalone `from_pretrained()` probe and
live end-to-end through `./backend start` (`/health`, `/v1/models`,
real `/v1/chat/completions` output with token usage). New `model.
revision` config field (`MODEL_REVISION` env override) threads this
through `TransformersEngine.load_model()`; `config/config.yaml` now
pins it for `mistralai/Mistral-7B-Instruct-v0.2`. General lesson: when
a `transformers`-4.36-era-supported model still fails to load, check
whether it's a Hub file-format drift issue before assuming the
architecture itself is unsupported — an older revision is often a
fix, not just a Mistral-specific workaround.

Also confirmed working the same session: `NousResearch/Llama-2-7b-chat-hf`
(ungated Llama-2-7B-chat mirror) loads and generates cleanly at `main`,
no revision pinning needed — not deployed (Mistral was chosen instead
to close the previously-open Mistral item), but validates the Llama
family still works at larger sizes than TinyLlama.

**bitsandbytes is currently broken on UBI, independent of model
choice:** `import bitsandbytes` itself fails —
`AttributeError: module 'torch.library' has no attribute
'impl_abstract'`. The installed `bitsandbytes` build expects a `torch`
API that doesn't exist in the pinned `torch==2.0.1+cu117`. This means
`model.quantization: 4bit`/`8bit` cannot actually run on UBI today
despite being implemented and unit-tested (`docs/quantization-design.md`)
— confirmed by reproducing the failure directly, not inferred. Not
fixed: would need a `bitsandbytes` version compatible with `torch`
2.0.1, not yet identified. `config/config.yaml` stays on `quantization:
none` until this is resolved.

**Practical consequence:** fp16 is the only working precision on UBI
right now, which means model choice is also VRAM-choice: a 7B model in
fp16 needs ~14.8GB, i.e. essentially all of one RTX A4000's 16GB.
GPU 0/1 are frequently occupied by another user's concurrent job
(~6.6GB used each, observed repeatedly this session and in the prior
session), which doesn't leave enough headroom for an unquantized 7B.
`config/config.yaml`'s `backend.gpu` was changed from `0` to `2` (the
idle card) for this reason — a single-index config change, not a
multi-GPU setup (see `docs/future-tasks.md`'s Multi-GPU entry, still
unstarted).

**Open strategic question from 2026-07-30, now answered by this
session's results:** given how narrow and moving UBI's compatibility
window looked, it was unclear whether continuing to chase
`TransformersEngine` compatibility here was worth it versus leaning on
the Local Node (Ollama). This session's outcome: the window is narrow
but tractable — Llama-2-7B, Mistral-7B (revision-pinned), and
Llama-3-8B (see below) all load and serve correctly. Worth continuing
to invest here, at least until the bitsandbytes/VRAM constraint above
becomes limiting again.

**Multi-GPU sharding confirmed working, and needed for Llama-3
(2026-07-31):** `NousResearch/Meta-Llama-3-8B-Instruct` (ungated
mirror, April 2024, no revision pin needed) loads under the pinned
`transformers==4.36.0` — but not on one GPU: its fp16 weights come
within a few hundred MiB of a full 16GB card because Llama-3's
vocabulary (128k tokens) is ~4x Llama-2/Mistral's (32k), inflating the
embedding/lm_head layers well past what "same param count" would
suggest. A single-GPU pinned probe hit `CUDA out of memory` (tried to
allocate 112MiB with 14.85GiB already allocated on a 15.74GiB card).
Retrying with `device_map="auto"` across two visible GPUs (via
`CUDA_VISIBLE_DEVICES=2,3`) worked cleanly — `accelerate` split it
~7.5GB/~9.2GB across the two cards, no CPU offload, no errors. This
is the first real (non-mocked) exercise of the multi-GPU path
described in `docs/future-tasks.md`'s Multi-GPU entry; no code change
was needed; `backend.gpu: "2,3"` in `config/config.yaml` was
sufficient. **Caveat confirmed live, not just theoretical:**
`GPUManager` (`services/gpu.py`) matches `backend.gpu` against a
single `nvidia-smi` index, so with a comma-separated value `./backend
status`/`gpu current` now show VRAM and temperature as unavailable —
cosmetic only, `/health` (which reads `torch.cuda.get_device_name(0)`
directly, not through `GPUManager`) and actual inference are both
unaffected.

`NousResearch/Meta-Llama-3-8B-Instruct` (2 GPUs) is now the live
default on UBI, confirmed via `/health`, `/v1/models`, and a real
`/v1/chat/completions` call. All testing and deployment this session
stayed on GPU 2/3 only — GPU 0/1 (the other user's concurrent job)
were never touched, verified via `nvidia-smi` before and after each
step.

**Ceiling for "bigger," independent of VRAM:** disk. This box's single
908GB volume is ~97-99% used by other users' data throughout this
session (13-28GB free depending on what's cached locally at any given
moment). A 70B-class model needs ~140GB just for fp16 weights — that
does not fit regardless of how many of the 4 GPUs' combined 64GB VRAM
are available. The practical "go bigger" ceiling on this machine is
roughly the 13B-34B range (Llama-2-13B ≈ 26GB, still Llama-family),
not larger, until disk changes.

### Multi-family model compatibility sweep (2026-07-31)

Systematic pass/fail sweep across model families beyond Llama/Mistral,
using the standalone `probe_model.py` diagnostic (tokenizer load +
model load + one `generate()` call, on GPU 2/3 only — GPU 0/1, the
other user's job, checked via `nvidia-smi` before/after and never
touched). Each candidate's cache was deleted before the next download,
given the ~30GB disk ceiling. Small pure-Python deps needed by some
repos' `trust_remote_code` (`tiktoken==0.5.1` — newer needs a Rust
compiler this box's `pip` can't build; `einops`; `transformers_stream_
generator`) were installed into the `llm` env; no core dependency
(`torch`/`transformers`/`accelerate`) was touched.

**Background fact that shapes this whole sweep:** `transformers==
4.36.0`'s `CONFIG_MAPPING_NAMES` has **zero** entries for `qwen` or
`gemma`, at any version — confirmed by inspecting it directly, not
inferred. So for those two families, the only possible path is a
repo that ships its own `trust_remote_code` modeling files predating
each family's full upstream integration into `transformers`.

| Model | Result | Why |
|---|---|---|
| `Qwen/Qwen-7B-Chat` (Qwen1, Aug 2023) | **PASS** | `trust_remote_code=True` + `tiktoken`/`einops`/`transformers_stream_generator`; ships its own modeling code, sidesteps the missing native class entirely. |
| `Qwen/Qwen1.5-7B-Chat` | FAIL | Needs `Qwen2Tokenizer`, which doesn't exist in 4.36.0 - and unlike Qwen1, ships no `trust_remote_code` fallback. Fails before any download. |
| `Qwen/Qwen2-7B-Instruct` | FAIL | Same as Qwen1.5 - identical `Qwen2Tokenizer` error. |
| `Qwen/Qwen3-1.7B` | FAIL | Same as Qwen1.5/2 - confirms Qwen3's block (docs/problems.md's original entry) is the same root cause as every post-Qwen1 release, not something specific to Qwen3. |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | FAIL | Loads as plain `LlamaForCausalLM`, but its `rope_scaling` uses the Llama-3.1-style dict (`{factor, low_freq_factor, high_freq_factor, original_max_position_embeddings, rope_type: "llama3"}`), which 4.36.0's `LlamaConfig` validation rejects - it only accepts the older `{type, factor}` shape. Confirms this distill is built on a Llama-3.1 base, not the plain Llama-3 that already works here. |
| `deepseek-ai/deepseek-llm-7b-chat` (Dec 2023) | **PASS** | Plain `LlamaForCausalLM`, older rope config shape. |
| `deepseek-ai/deepseek-coder-6.7b-instruct` | **PASS** | Same as above. |
| `mistralai/Mistral-7B-Instruct-v0.1` | **PASS** (revision-pinned) | Same "CVST Tokenizer Badger" Hub re-save pattern as v0.2 (docs/problems.md's earlier entry) - full commit history checked, confirmed the same auto-generated commit exists here too; pinning to the commit before it fixes it. |
| `mistralai/Mistral-7B-Instruct-v0.3` | FAIL, no fix available | Same tokenizer.json parse error, but its *entire* commit history (34 commits) dates from its initial 2024-05-22 release - it shipped with the modern format from day one, so there is no older revision to pin to. Would need a newer `tokenizers` library - a core-stack change, out of scope. |
| `mistralai/Mistral-Nemo-Instruct-2407` (12B) | FAIL | Weight-shape mismatch (`[1024,5120]` vs `[1280,5120]`) - Mistral-Nemo introduced an explicit `head_dim` config field that 4.36.0's `MistralConfig`/modeling code doesn't read, so it infers a different (wrong) head dimension than the checkpoint was actually saved with. Architecture-level gap, not fixable by installing a package. |
| `alpindale/gemma-7b-it` (ungated Gemma-1 mirror) | FAIL | Needs `GemmaTokenizer`, absent in 4.36.0; confirmed no `trust_remote_code` fallback either (fails identically both ways). Fails before any download either way - the official `google/gemma-*` repos are additionally gated (401, needs a human to accept Google's license on huggingface.co), but that's moot here since the architecture itself isn't loadable regardless. |
| `tiiuae/falcon-7b-instruct` | **PASS** | Via `trust_remote_code=True` - even though Falcon is now fully native in current `transformers`, the repo still carries its original custom modeling files, which the dynamic-module loader happily serves to an old `transformers` that doesn't have native Falcon support in this exact form. |
| `microsoft/phi-2` | **FAIL - silent, not a crash** | Loaded with **no exception**, but logged that essentially every `self_attn` weight in the checkpoint was "not used" and every `query_key_value` weight was "newly initialized" (random). The Hub checkpoint now uses split `q_proj`/`k_proj`/`v_proj` naming; 4.36.0's built-in `PhiForCausalLM` expects an older fused `query_key_value` naming. `generate()` "succeeded" and produced pure noise (`"'s.\n:\n:\n\"\n\"\n\"..."`). **This is the most dangerous failure mode found this session**: a naive test that only checks "did it throw" would wrongly call this a pass. Also separately hit (before this): `device_map="auto"` isn't supported by 4.36.0's `PhiForCausalLM` at all (`_no_split_modules` not implemented) - irrelevant for a model this small, worked once pinned to a single GPU, but is its own real gap. |

**Net result:** confirmed working (beyond the Llama-3-8B already
deployed): `Qwen/Qwen-7B-Chat`, `deepseek-ai/deepseek-llm-7b-chat`,
`deepseek-ai/deepseek-coder-6.7b-instruct`,
`mistralai/Mistral-7B-Instruct-v0.1`, `tiiuae/falcon-7b-instruct`.
None deployed - `NousResearch/Meta-Llama-3-8B-Instruct` remains the
live default (most recent architecture confirmed working). Recorded
here as validated options for a future model switch without needing
to re-run this compatibility research.

**Methodology note:** for `trust_remote_code=True` repos, "did the
process exit 0" is not sufficient evidence of a working model - the
Phi-2 case shows a load can silently substitute random weights for a
naming mismatch and still complete. Always inspect the actual
`generate()` output for coherence, not just the absence of a
traceback.

### If reproducing/verifying this

```bash
nvidia-smi  # confirms "CUDA Version: 11.4" and driver 470.86
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from transformers.utils import is_torch_available; print(is_torch_available())"
```

## Runtime Not Fully Verified In Sandbox

The code has passed syntax checks and config-loading checks in the current
workspace, but the live server has not been fully exercised here because this
sandbox does not expose the expected `~/miniforge3` Conda installation.

Before treating a deployment as healthy, run:

```bash
./scripts/start.sh
./scripts/status.sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

And test chat completion:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Say hello from Nemoclaw in one sentence."}
    ],
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

## PyYAML Is Now Required

`config.py` uses PyYAML. The `llm` Conda environment must include `yaml`
import support:

```bash
python -c "import yaml; print(yaml.__version__)"
```

## Typer Is Required For CLI Operations

The `backend` command uses Typer. The `llm` Conda environment must include
`typer` import support:

```bash
python -c "import typer; print(typer.__version__)"
```

## Streaming Is Not Implemented

`stream: true` requests return a `400` response. The endpoint accepts the field
for OpenAI-style request compatibility, but streaming output is future work.

## Benchmarks Require A Running Backend

`backend benchmark ...` commands call the local OpenAI-compatible HTTP endpoint.
They can be unit-tested without a GPU, but live benchmark numbers require the
backend to be running with a model loaded:

```bash
./backend start
./backend benchmark latency
./backend benchmark throughput
./backend benchmark vram
```
