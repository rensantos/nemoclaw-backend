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

**Read the above as a dated observation, not a standing rule.** No GPU on
this box is assigned to anyone: allocation is opportunistic, whoever
needs a free card takes one, and which indices are busy changes over
time. "GPU 0/1 are the other user's, 2/3 are ours" is a
misreading — repeatedly made — of what is only a snapshot of who happened
to be running what. Always check what is actually free (`./backend gpu`)
before assuming an index is available, and record a machine's choice in
the untracked `config.local.yaml` rather than the shared file.

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
| `google/gemma-4-12B-it` / `google/gemma-4-E4B-it` | FAIL | Real current-generation Gemma (24GB/16GB, both ungated - unlike `google/gemma-*-it`, which are gated). Uses a brand-new `gemma4`/`gemma4_unified` architecture (`Gemma4ForConditionalGeneration` etc.), but fails with the identical `GemmaTokenizer does not exist` error as the Gemma-1 mirror below - confirms Gemma is blocked at the transformers-native-support level for every generation, not an artifact of testing an old release. |
| `alpindale/gemma-7b-it` (ungated Gemma-1 mirror) | FAIL | Needs `GemmaTokenizer`, absent in 4.36.0; confirmed no `trust_remote_code` fallback either (fails identically both ways). Fails before any download either way - the official `google/gemma-*` repos are additionally gated (401, needs a human to accept Google's license on huggingface.co), but that's moot here since the architecture itself isn't loadable regardless. |
| `tiiuae/falcon-7b-instruct` | **PASS** | Via `trust_remote_code=True` - even though Falcon is now fully native in current `transformers`, the repo still carries its original custom modeling files, which the dynamic-module loader happily serves to an old `transformers` that doesn't have native Falcon support in this exact form. |
| `microsoft/phi-2` | **FAIL - silent, not a crash** | Loaded with **no exception**, but logged that essentially every `self_attn` weight in the checkpoint was "not used" and every `query_key_value` weight was "newly initialized" (random). The Hub checkpoint now uses split `q_proj`/`k_proj`/`v_proj` naming; 4.36.0's built-in `PhiForCausalLM` expects an older fused `query_key_value` naming. `generate()` "succeeded" and produced pure noise (`"'s.\n:\n:\n\"\n\"\n\"..."`). **This is the most dangerous failure mode found this session**: a naive test that only checks "did it throw" would wrongly call this a pass. Also separately hit (before this): `device_map="auto"` isn't supported by 4.36.0's `PhiForCausalLM` at all (`_no_split_modules` not implemented) - irrelevant for a model this small, worked once pinned to a single GPU, but is its own real gap. |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | FAIL | A second, smaller DeepSeek-R1 distillation (16.4GB, this one on a `Qwen3ForCausalLM` base rather than Llama). `KeyError: 'qwen3'` - inherits the exact Qwen3 architecture block already found above, before any download. |

**On the real DeepSeek flagships (R1, V3, V4):** not attempted, and
not worth attempting on this box. `deepseek-ai/DeepSeek-R1` is
**~688GB**, `deepseek-ai/DeepSeek-V4-Flash` is **~160GB**,
`deepseek-ai/DeepSeek-V4-Pro` is **~865GB** (all real, current Hub
listings, checked via `HfApi.model_info(files_metadata=True)`, not
estimated). These are large MoE models; UBI's disk ceiling has topped
out around ~30GB free all session. This isn't a VRAM/GPU-count
problem - no combination of the 4 GPUs' 64GB combined VRAM changes
whether ~700GB of weights fits on a disk that has ~30GB truly free at
its best moment. Only the small Llama/Qwen-based *distillations*
DeepSeek publishes alongside these (already covered above) are even
disk-feasible here.

**Net result:** confirmed working (beyond the Llama-3-8B already
deployed): `Qwen/Qwen-7B-Chat`, `deepseek-ai/deepseek-llm-7b-chat`,
`deepseek-ai/deepseek-coder-6.7b-instruct`,
`mistralai/Mistral-7B-Instruct-v0.1`, `tiiuae/falcon-7b-instruct`.
None of these were deployed to replace `NousResearch/Meta-Llama-3-8B-
Instruct` as `TransformersEngine`'s configured model at the time. Recorded
here as validated options for a future `TransformersEngine` model switch
without needing to re-run this compatibility research. (UBI's live
default has since moved to `OllamaEngine`/`qwen3:8b` - see
`docs/ollama-on-ubi-design.md` - but `TransformersEngine`'s config and
this research both remain intact for a one-line revert.)

**Methodology note:** for `trust_remote_code=True` repos, "did the
process exit 0" is not sufficient evidence of a working model - the
Phi-2 case shows a load can silently substitute random weights for a
naming mismatch and still complete. Always inspect the actual
`generate()` output for coherence, not just the absence of a
traceback.

### Compatibility sweep follow-up (2026-07-31, same session continued)

Additional candidates tested after the table above, using the same
`probe_model.py` methodology:

| Model | Result | Why |
|---|---|---|
| `Qwen/Qwen-14B-Chat` | **PASS** | Same Qwen1 `trust_remote_code` path as `Qwen-7B-Chat` above, ~28GB. This is the practical ceiling for the Qwen1 line specifically - Qwen1.5+ stays blocked per the table above, so 14B is "the biggest Qwen this stack can run" until the Qwen2/2.5 fix below. |
| `Qwen/Qwen-72B-Chat` | Not attempted (size-checked only) | ~144.6GB - same disk wall as the DeepSeek flagships above, no GPU/VRAM combination changes this. |
| `upstage/SOLAR-10.7B-Instruct-v1.0` | **PASS** | Depth-upscaled Llama architecture (plain `llama` model type) - no `trust_remote_code`, no revision pin, loads at `main`. |
| `01-ai/Yi-1.5-9B-Chat` | **SUSPECT - not a clean pass** | Loads with no exception, but `generate()` output has no whitespace between words at all. Same failure category as the Phi-2 case above (silent, not a crash) - likely a tokenizer-decode quirk in Yi's SentencePiece config under this old `transformers`, not verified further. Do not treat as validated without more investigation. |
| `HuggingFaceH4/zephyr-7b-beta` | **PASS** | `mistral` architecture, no `trust_remote_code`, no revision pin. |

**Superseded by the Qwen2/Qwen2.5 finding below:** the "Qwen1.5+ is
blocked" conclusion in the table above was later narrowed, not
overturned - see the Qwen version investigation section below for what
actually changes it and what doesn't (Qwen3 stays blocked; Qwen2/2.5
don't, with a one-minor-version `transformers` bump).

Queued but never tested (architecture-checked via `config.json` only, all
need `trust_remote_code=True` and 4.36.0's native registry lacks the
class either way - untested whether any ships a working fallback the way
Falcon/Qwen1 did): `NousResearch/Nous-Hermes-2-Mistral-7B-DPO` (plain
`mistral`, should behave like the other Mistral finetunes above),
`THUDM/chatglm3-6b`, `internlm/internlm2-chat-7b`,
`baichuan-inc/Baichuan2-13B-Chat`, `stabilityai/stablelm-2-12b-chat`,
`bigcode/starcoder2-15b-instruct-v0.1`.

### Qwen version investigation: Qwen2/2.5 work, Qwen3 is a confirmed dead end (2026-07-31)

Separate from the "no native Qwen support at all in 4.36.0" finding
above: is there a `transformers` version between the pinned `4.36.0` and
the known-broken `4.51.0`+ (`docs/problems.md`'s driver section) that
adds Qwen support without hitting the `torch.compiler`/`torch>=2.1`
break? Tested in an isolated `llm-qwen-test` conda env, never touching
the production `llm` env `./backend` depends on (**note:** `conda create
--clone` is not reliable for this - it silently reconstructed a stale
conda-recorded `torch==1.12.1` instead of the actual pip-installed
`torch==2.0.1+cu117` running in production; build test envs with an
explicit `pip install torch==2.0.1+cu117 --index-url
https://download.pytorch.org/whl/cu117` instead).

- **`qwen2` enters `CONFIG_MAPPING_NAMES` at exactly `transformers==
  4.37.0`** - one minor version above the pinned `4.36.0`. Confirmed
  working end-to-end: `Qwen/Qwen2-0.5B-Instruct` and
  `Qwen/Qwen2.5-1.5B-Instruct` both load and `generate()` coherently
  under `4.37.0` + the unchanged `torch==2.0.1+cu117`. No driver or
  torch change needed for Qwen2/2.5 at all.
- **Real catch:** `Qwen/Qwen2.5-1.5B-Instruct` produces silent garbage
  (`"!!!!!!..."`) when force-loaded in `float16` - which is what
  `engines/transformers_engine.py`'s `_load_kwargs()` hardcodes for
  every model today, regardless of the checkpoint's native dtype.
  Switching to `torch_dtype=torch.bfloat16` (the dtype in the model's
  own `config.json`) fixes it completely. `Qwen2-0.5B-Instruct`
  tolerated `float16` fine, so this is likely a scale/checkpoint-specific
  overflow, not universal - same "no crash isn't proof of a working
  model" lesson as the Phi-2/Yi-1.5 cases above, just for dtype instead
  of weight-naming. **Promoting any Qwen2/2.5 model into
  `TransformersEngine` needs a `_load_kwargs()` code change, not just a
  `requirements.txt` bump.**
- **`qwen3` is a confirmed dead end, not "not yet tried."** `qwen3` only
  enters `CONFIG_MAPPING_NAMES` at exactly `transformers==4.51.0`. The
  `torch.compiler`/`torch>=2.1` break (`AttributeError: module 'torch'
  has no attribute 'compiler'`, raised while importing
  `transformers.models.llama.modeling_llama`, which Qwen3 also depends
  on) was binary-searched and confirmed already present at `4.50.0`, with
  `4.49.0` the last version that still works - strictly before Qwen3
  support ever existed at any version. No `transformers` version has
  both. Would need a driver upgrade (no sudo on UBI) to ever change this.
- **Regression-checked on `4.37.0`:** `TinyLlama`,
  `mistralai/Mistral-7B-Instruct-v0.2` (revision-pinned), and
  `NousResearch/Meta-Llama-3-8B-Instruct` (then-live) all still load and
  generate coherently - no regression from the `4.36.0` -> `4.37.0` bump
  found. **Not promoted to `requirements.txt`** - the Ollama-on-UBI
  deployment (`docs/ollama-on-ubi-design.md`) reaches actual Qwen3 more
  directly and became the live choice instead; this `4.37.0` finding
  remains available if `TransformersEngine`-served Qwen2/2.5 is wanted
  later.

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

## Streaming (resolved 2026-08-03)

`stream: true` now returns Server-Sent Events per OpenAI convention. It
previously returned `400` unconditionally.

Still true: `400` is returned when the **active engine** cannot stream.
`TransformersEngine` generates a whole response in one call, so it has no
incremental path and says so rather than faking one.

Known limitation: for a model whose chat template leaks reasoning inline
(`qwen3:30b`), the reasoning prefix cannot stream incrementally — it is
buffered until the `</think>` marker proves where reasoning ends, because a
stream cannot retract tokens it has already sent. The answer after the
marker streams normally.

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
