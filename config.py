import os
from dataclasses import dataclass
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"

DEFAULTS = {
    "backend": {
        "host": "127.0.0.1",
        "port": 8000,
        "gpu": 0,
        "engine": "transformers",
    },
    "model": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_tokens_default": 256,
        "temperature_default": 0.7,
    },
}

VALID_ENGINES = ("transformers", "ollama")


@dataclass(frozen=True)
class BackendConfig:
    host: str
    port: int
    gpu: str
    engine: str


@dataclass(frozen=True)
class ModelConfig:
    id: str
    max_tokens_default: int
    temperature_default: float


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
    model_id = _env_value("MODEL_ID", _section_value(raw_config, "model", "id"))
    max_tokens_default = _int_env(
        "MAX_TOKENS_DEFAULT",
        _section_value(raw_config, "model", "max_tokens_default"),
    )
    temperature_default = _float_env(
        "TEMPERATURE_DEFAULT",
        _section_value(raw_config, "model", "temperature_default"),
    )

    return Config(
        backend=BackendConfig(host=host, port=port, gpu=gpu, engine=engine),
        model=ModelConfig(
            id=model_id,
            max_tokens_default=max_tokens_default,
            temperature_default=temperature_default,
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
