import time
from typing import Dict, List, Optional

from config import settings

import torch
import torch.fx
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = settings.model_id

print("Loading model: {}".format(MODEL_ID))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()


def _model_device():
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _message_dicts(messages: List) -> List[Dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _prompt_from_messages(messages: List) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            _message_dicts(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

    lines = []
    for message in messages:
        lines.append("{}: {}".format(message.role, message.content))
    lines.append("assistant:")
    return "\n".join(lines)


def _tokenize_prompt(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs.to(_model_device())


def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": created,
                "owned_by": "local",
            }
        ],
    }


def generate_chat(messages: List, max_tokens: Optional[int], temperature: Optional[float]):
    max_new_tokens = (
        settings.max_tokens_default if max_tokens is None else max_tokens
    )
    temp = settings.temperature_default if temperature is None else temperature
    prompt = _prompt_from_messages(messages)
    inputs = _tokenize_prompt(prompt)
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            do_sample=temp > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0][prompt_tokens:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    completion_tokens = int(generated_ids.shape[-1])

    return {
        "content": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def generate_text(prompt: str, max_new_tokens: int, temperature: float):
    inputs = _tokenize_prompt(prompt)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"model": MODEL_ID, "response": text}
