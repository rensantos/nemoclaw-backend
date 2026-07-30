# TransformersEngine Quantization Design

This is a design document for adding optional 4-bit/8-bit quantized model
loading to `TransformersEngine`, so the UBI Node's RTX A4000 (16GB VRAM)
can serve a larger, better-quality model than fp16 alone allows.

## 1. Problem

`TransformersEngine.load_model()` currently loads every model with
`torch_dtype=torch.float16` and no quantization. An 8B-parameter model's
weights alone are ~16GB in fp16 — the entire card, with nothing left for
activations or KV cache. In practice this caps `engine: transformers`
deployments at ~7B-class models today, which is well below what a 16GB
GPU can serve if quantized.

## 2. DECISION: config surface

Add `model.quantization` to `config/config.yaml`, alongside the existing
`model.id`:

- Valid values: `"none"` (default), `"4bit"`, `"8bit"`.
- `MODEL_QUANTIZATION` environment override, following the exact
  `ENGINE`/`VALID_ENGINES` pattern already in `config.py`: fail fast with a
  clear error at `load_config()` time if the value isn't one of the three.
- `config.py`: add `quantization: str` to `ModelConfig`, add
  `"quantization": "none"` to `DEFAULTS["model"]`, add `VALID_QUANTIZATIONS
  = ("none", "4bit", "8bit")`.
- `"none"` behavior is byte-for-byte unchanged from today: existing
  deployments that don't set `model.quantization` see no difference.
- `./backend config` prints the configured value (new
  `Quantization: <value>` line in `cli.py`'s `_print_config()`), matching
  how the other `model.*`/`backend.*` fields are already surfaced.

## 3. DECISION: quantization backend and defaults

Use `bitsandbytes` via `transformers.BitsAndBytesConfig`, the standard,
well-established HF integration — not a custom quantization path.

- **New dependency**: `bitsandbytes` added to `requirements.txt`.
  Operator prerequisite: install it inside UBI's `llm` conda env
  (Python 3.9) before enabling quantization; verify with
  `python -c "import bitsandbytes"` after `pip install bitsandbytes`.
  Compatibility with the installed CUDA/torch build on UBI is not
  verified by this document — that is an operator validation step, the
  same posture `docs/ollama-engine-design.md` took for Ollama's
  OS/glibc compatibility on its target machine.
- **`"4bit"`** maps to:
  ```python
  BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True,
  )
  ```
  NF4 + double quantization is the standard recommended default for
  inference (not a training/QLoRA-specific choice) — best quality-per-bit
  among bitsandbytes' 4-bit modes, per HF's own guidance. Compute dtype
  stays float16, consistent with this engine's existing fp16 posture.
- **`"8bit"`** maps to `BitsAndBytesConfig(load_in_8bit=True)` — the
  simpler LLM.int8() path, offered for models or setups where 4-bit
  quality loss is unacceptable but full fp16 doesn't fit.
- When `quantization` is `"4bit"` or `"8bit"`, `AutoModelForCausalLM.
  from_pretrained()` is called with `quantization_config=<the above>` and
  **without** `torch_dtype=torch.float16` (the quantization config's
  `bnb_*_compute_dtype` governs compute precision instead — passing both
  is redundant and a known source of confusing warnings from
  `transformers`). When `quantization` is `"none"`, behavior is exactly
  today's: `torch_dtype=torch.float16`, no `quantization_config`.
- `device_map="auto"` is unchanged and used in all three cases — it was
  already required for multi-shard fp16 loads and is also required by
  bitsandbytes.

## 4. Non-goals

- **No dynamic/per-request quantization switching.** One fixed
  `model.quantization` value per backend instance, exactly like
  `model.id` and `backend.engine` today — consistent with this project's
  established "one engine, one model, one config per instance" posture
  (`docs/ollama-engine-design.md` Section 3 made the same call for engine
  selection).
- **No automatic VRAM-based model/quantization selection.** The operator
  picks `model.id` and `model.quantization` explicitly; the backend does
  not probe VRAM and choose for them.
- **No CPU quantization path.** This is specifically the CUDA
  bitsandbytes path; `engine: transformers` remains a GPU-serving engine.
- **No change to `OllamaEngine`** or any other engine — quantization here
  is a `TransformersEngine`-specific loading detail. Ollama has its own
  quantization story (GGUF quant levels, chosen at `ollama pull` time),
  entirely outside this backend's control, already noted in
  `docs/ollama-engine-design.md`.

## 5. Risks

- **bitsandbytes/CUDA/Python 3.9 compatibility on UBI is unverified by
  this document.** Must be checked on the actual machine before relying
  on it (see Section 3's operator prerequisite).
- **Quality tradeoff.** 4-bit quantization measurably degrades output
  quality versus fp16, though NF4 double-quantization is close for most
  chat/instruct use cases. 8-bit is closer to fp16 quality at roughly
  double the 4-bit memory footprint. This is a real tradeoff for the
  operator to weigh per model, not something this design resolves
  generically.
- **Load-time failures surface the same way they do today**: an
  incompatible model/quantization combination raises during
  `load_model()`, which already crashes startup with a clear error
  (`InferenceService.__init__` calls `load_model()` eagerly, uncaught) —
  no new silent-failure mode is introduced.

## 6. Testing

`TransformersEngine` has no existing unit tests (it requires a real GPU
and model download to exercise for real). This increment adds the first:
mock `transformers.AutoModelForCausalLM.from_pretrained` and
`AutoTokenizer.from_pretrained` to assert the exact kwargs
(`quantization_config` present/absent, `torch_dtype` present/absent) for
each of `"none"`/`"4bit"`/`"8bit"`, plus `config.py` precedence/fail-fast
tests for `model.quantization` mirroring the existing `backend.engine`
tests. No real model load, no GPU, no bitsandbytes install required to
run these tests.

Live validation happens only on UBI, with `bitsandbytes` actually
installed and a real model pulled: confirm `./backend start` with
`MODEL_QUANTIZATION=4bit` loads successfully, `/health` still reports
`cuda`/`gpu` correctly, and a real `/v1/chat/completions` call succeeds.
