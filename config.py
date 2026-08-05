import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"

DEFAULTS = {
    "backend": {
        "host": "127.0.0.1",
        "port": 8000,
        "gpu": 0,
        "engine": "transformers",
        "ollama_host": "http://127.0.0.1:11434",
        # Empty means "use this machine's hostname". Every backend instance
        # otherwise answers /health with the same shape, so with several
        # reachable at once (a local one plus an SSH-tunnelled remote)
        # there is no way to tell which machine replied.
        "instance": "",
    },
    "model": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_tokens_default": 256,
        "temperature_default": 0.7,
        "quantization": "none",
        "revision": "",
        "think_default": None,
    },
}

VALID_ENGINES = ("transformers", "ollama")
VALID_QUANTIZATIONS = ("none", "4bit", "8bit")


@dataclass(frozen=True)
class BackendConfig:
    host: str
    port: int
    gpu: str
    engine: str
    ollama_host: str
    instance: str


@dataclass(frozen=True)
class ModelConfig:
    id: str
    max_tokens_default: int
    temperature_default: float
    quantization: str
    revision: str
    think_default: Optional[bool]


@dataclass(frozen=True)
class Config:
    backend: BackendConfig
    model: ModelConfig

    @property
    def host(self) -> str:
        return self.backend.host

    @property
    def port(self) -> int:
        return self.backend.port

    @property
    def gpu(self) -> str:
        return self.backend.gpu

    @property
    def model_id(self) -> str:
        return self.model.id

    @property
    def max_tokens_default(self) -> int:
        return self.model.max_tokens_default

    @property
    def temperature_default(self) -> float:
        return self.model.temperature_default


def _load_yaml_config():
    return load_yaml_config()


def load_yaml_config(path: Path = CONFIG_PATH):
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("{} must contain a YAML mapping".format(path))

    return loaded


def _section_value(raw_config, section, key):
    section_data = raw_config.get(section, {})
    if not isinstance(section_data, dict):
        return DEFAULTS[section][key]
    return section_data.get(key, DEFAULTS[section][key])


def _env_value(name: str, fallback):
    value = os.environ.get(name)
    if value is None or value == "":
        return fallback
    return value


def _int_env(name: str, fallback) -> int:
    value = _env_value(name, fallback)
    return int(value)


def _float_env(name: str, fallback) -> float:
    value = _env_value(name, fallback)
    return float(value)


def _optional_bool_env(name: str, fallback):
    value = os.environ.get(name)
    if value is None or value == "":
        return fallback
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise ValueError(
        "Invalid {} '{}'; expected a boolean (true/false)".format(name, value)
    )


def load_config() -> Config:
    raw_config = _load_yaml_config()

    host = _env_value("HOST", _section_value(raw_config, "backend", "host"))
    port = _int_env("PORT", _section_value(raw_config, "backend", "port"))
    gpu = str(_env_value("GPU", _section_value(raw_config, "backend", "gpu")))
    engine = _env_value("ENGINE", _section_value(raw_config, "backend", "engine"))
    if engine not in VALID_ENGINES:
        raise ValueError(
            "Invalid backend.engine '{}'; valid values: {}".format(
                engine, ", ".join(VALID_ENGINES)
            )
        )
    ollama_host = _env_value(
        "OLLAMA_HOST", _section_value(raw_config, "backend", "ollama_host")
    )
    instance = str(
        _env_value("INSTANCE", _section_value(raw_config, "backend", "instance")) or ""
    ).strip() or socket.gethostname()
    model_id = _env_value("MODEL_ID", _section_value(raw_config, "model", "id"))
    max_tokens_default = _int_env(
        "MAX_TOKENS_DEFAULT",
        _section_value(raw_config, "model", "max_tokens_default"),
    )
    temperature_default = _float_env(
        "TEMPERATURE_DEFAULT",
        _section_value(raw_config, "model", "temperature_default"),
    )
    quantization = _env_value(
        "MODEL_QUANTIZATION", _section_value(raw_config, "model", "quantization")
    )
    if quantization not in VALID_QUANTIZATIONS:
        raise ValueError(
            "Invalid model.quantization '{}'; valid values: {}".format(
                quantization, ", ".join(VALID_QUANTIZATIONS)
            )
        )
    revision = _env_value(
        "MODEL_REVISION", _section_value(raw_config, "model", "revision")
    )
    think_default = _optional_bool_env(
        "MODEL_THINK_DEFAULT", _section_value(raw_config, "model", "think_default")
    )

    return Config(
        backend=BackendConfig(
            host=host,
            port=port,
            gpu=gpu,
            engine=engine,
            ollama_host=ollama_host,
            instance=instance,
        ),
        model=ModelConfig(
            id=model_id,
            max_tokens_default=max_tokens_default,
            temperature_default=temperature_default,
            quantization=quantization,
            revision=revision,
            think_default=think_default,
        ),
    )


def configured_models(raw_config=None):
    """Compatibility wrapper. New code should use services.model.ModelManager."""
    from services.model import ModelManager

    manager = ModelManager()
    return manager._configured_models(raw_config)


def selected_model_id(raw_config=None) -> str:
    """Compatibility wrapper. New code should use services.model.ModelManager."""
    from services.model import ModelManager

    return ModelManager().selected_model_id(raw_config)


def configured_model(model_id: str, raw_config=None):
    """Compatibility wrapper. New code should use services.model.ModelManager."""
    from services.model import ModelManager

    return ModelManager()._configured_model(model_id, raw_config)


def update_selected_model(model_id: str, path: Path = CONFIG_PATH) -> None:
    """Compatibility wrapper. New code should use services.model.ModelManager."""
    from services.model import ModelManager

    ModelManager(path).select_model(model_id)


config = load_config()
settings = config

os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.backend.gpu)


if __name__ == "__main__":
    print("Host: {}".format(config.backend.host))
    print("Port: {}".format(config.backend.port))
    print("GPU: {}".format(config.backend.gpu))
    print("Engine: {}".format(config.backend.engine))
    print("Model: {}".format(config.model.id))
