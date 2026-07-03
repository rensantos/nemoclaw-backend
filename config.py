import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_id: str
    gpu: str
    host: str
    port: int
    max_tokens_default: int
    temperature_default: float


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


settings = Settings(
    model_id=os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    gpu=os.environ.get("GPU", "0"),
    host=os.environ.get("HOST", "127.0.0.1"),
    port=_int_env("PORT", 8000),
    max_tokens_default=_int_env("MAX_TOKENS_DEFAULT", 256),
    temperature_default=_float_env("TEMPERATURE_DEFAULT", 0.7),
)
