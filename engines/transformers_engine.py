import time
from typing import Dict, List, Optional

import torch
import torch.fx
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import Config
from engines.base import InferenceEngine


def _quantization_load_kwargs(quantization: str) -> dict:
    """Maps model.quantization to AutoModelForCausalLM.from_pretrained()
    kwargs (docs/quantization-design.md Section 3).

    "none" keeps today's exact fp16 behavior. "4bit"/"8bit" pass a
    BitsAndBytesConfig instead of torch_dtype - the quantization config's
    own compute dtype governs precision, and passing both is redundant.
    """
    if quantization == "4bit":
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        }
    if quantization == "8bit":
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    return {"torch_dtype": torch.float16}


def _device_map(config: Config):
    """Chooses the from_pretrained() device_map.

    Single-GPU quantized loads pin directly to the one visible device
    (device_map={"": 0}) instead of "auto": observed live on UBI,
    accelerate's automatic device-map planning sized against the model's
    *unquantized* footprint, decided it didn't fit the physically free
    VRAM, and offloaded some layers to CPU - which bitsandbytes' 4-bit
    loader then refuses ("Some modules are dispatched on the CPU or the
    disk", docs/quantization-design.md). CUDA_VISIBLE_DEVICES already
    restricts this process to exactly config.backend.gpu's device(s), so
    a single configured GPU is always local index 0 regardless of its
    real system index - pinning there sidesteps the bad estimate
    entirely, since a quantized model that needs pinning at all is, by
    definition, meant to fit on one GPU.

    "none" quantization and multi-GPU configurations keep "auto": fp16
    may still need it to shard a large model across multiple visible
    GPUs, and multi-GPU quantized sharding is unimplemented/future work
    (docs/future-tasks.md's Multi-GPU entry).
    """
    gpu_indices = str(config.backend.gpu).split(",")
    if config.model.quantization != "none" and len(gpu_indices) == 1:
        return {"": 0}
    return "auto"


def scan_local_cache() -> Dict[str, object]:
    """Read-only scan of the local Hugging Face cache: which model repos
    are actually downloaded on this machine right now.

    Returns {repo_id: huggingface_hub.CachedRepoInfo}, model repos only.
    Never downloads, deletes, or otherwise mutates the cache - purely
    reports current local disk state, the TransformersEngine-specific
    counterpart to OllamaEngine's GET /api/tags check
    (docs/ollama-engine-design.md's "no config.yaml model.available
    equivalent for Ollama" decision applies in reverse here: this is the
    equivalent Transformers never had).
    """
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        cache_info = scan_cache_dir()
    except CacheNotFound:
        return {}

    return {repo.repo_id: repo for repo in cache_info.repos if repo.repo_type == "model"}


class TransformersEngine(InferenceEngine):
    """Hugging Face Transformers implementation of the inference engine."""

    def __init__(self, config: Config):
        self.config = config
        self.model_id = config.model_id
        self.tokenizer = None
        self.model = None

    def load_model(self, model_id: Optional[str] = None) -> None:
        # supports_runtime_lifecycle is False, so InferenceService never
        # reaches here with a model_id; the parameter exists for contract
        # parity and is honoured rather than silently ignored.
        if model_id is not None:
            self.model_id = model_id
        if self.model is not None and self.tokenizer is not None:
            return

        revision = self.config.model.revision or None
        print("Loading model: {} (quantization={}, revision={})".format(
            self.model_id, self.config.model.quantization, revision or "main"
        ))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=revision,
            device_map=_device_map(self.config),
            **_quantization_load_kwargs(self.config.model.quantization),
        )
        self.model.eval()

    def unload_model(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def health(self):
        self.load_model()
        return {
            "status": "ok",
            "model": self.model_id,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    def list_models(self):
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "local",
                }
            ],
        }

    def chat(
        self,
        messages: List,
        max_tokens: Optional[int],
        temperature: Optional[float],
        requested_model: Optional[str] = None,
        think: Optional[bool] = None,
    ):
        # requested_model and think are accepted for interface parity with
        # OllamaEngine but intentionally unused: fixing this engine's
        # echo-and-serve quirk is out of scope (docs/ollama-engine-design.md
        # Section 1), and there's no reasoning-mode concept here at all.
        self.load_model()
        max_new_tokens = (
            self.config.max_tokens_default if max_tokens is None else max_tokens
        )
        temp = self.config.temperature_default if temperature is None else temperature
        prompt = self._prompt_from_messages(messages)
        inputs = self._tokenize_prompt(prompt)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                do_sample=temp > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output[0][prompt_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completion_tokens = int(generated_ids.shape[-1])

        return {
            "content": text,
            # Interface parity with OllamaEngine; no reasoning-mode concept
            # here, matching how this engine already ignores `think`.
            "reasoning": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        think: Optional[bool] = None,
    ):
        # think is accepted for interface parity with OllamaEngine but
        # intentionally unused - no reasoning-mode concept here.
        self.load_model()
        inputs = self._tokenize_prompt(prompt)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return {"model": self.model_id, "response": text}

    def _model_device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    def _message_dicts(self, messages: List) -> List[Dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

    def _prompt_from_messages(self, messages: List) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                self._message_dicts(messages),
                tokenize=False,
                add_generation_prompt=True,
            )

        lines = []
        for message in messages:
            lines.append("{}: {}".format(message.role, message.content))
        lines.append("assistant:")
        return "\n".join(lines)

    def _tokenize_prompt(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs.to(self._model_device())
