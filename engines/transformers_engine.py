import time
from typing import Dict, List, Optional

import torch
import torch.fx
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from engines.base import InferenceEngine


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

        print("Loading model: {}".format(self.model_id))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
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

    def chat(self, messages: List, max_tokens: Optional[int], temperature: Optional[float]):
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
