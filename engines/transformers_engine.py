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

    def load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        print("Loading model: {} (quantization={})".format(
            self.model_id, self.config.model.quantization
        ))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
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
    ):
        # requested_model is accepted for interface parity with OllamaEngine
        # but intentionally unused: fixing this engine's echo-and-serve
        # quirk is out of scope (docs/ollama-engine-design.md Section 1).
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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def generate_text(self, prompt: str, max_new_tokens: int, temperature: float):
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
