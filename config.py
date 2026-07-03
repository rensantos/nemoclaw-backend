import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"

DEFAULTS = {
    "backend": {
        "host": "127.0.0.1",
        "port": 8000,
        "gpu": 0,
    },
    "model": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_tokens_default": 256,
        "temperature_default": 0.7,
    },
}


@dataclass(frozen=True)
class BackendConfig:
    host: str
    port: int
    gpu: str


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
        backend=BackendConfig(host=host, port=port, gpu=gpu),
        model=ModelConfig(
            id=model_id,
            max_tokens_default=max_tokens_default,
            temperature_default=temperature_default,
        ),
    )


def _model_entry_from_id(model_id: str) -> Dict[str, object]:
    return {
        "id": model_id,
        "name": model_id,
        "path": model_id,
        "engine": "transformers",
        "device": "cuda",
    }


def _normalise_model_entry(model_id: str, value) -> Dict[str, object]:
    if isinstance(value, dict):
        entry = dict(value)
        entry.setdefault("id", model_id)
    else:
        entry = _model_entry_from_id(model_id)

    entry.setdefault("name", entry["id"])
    entry.setdefault("path", entry["id"])
    entry.setdefault("engine", "transformers")
    entry.setdefault("device", "cuda")
    return entry


def configured_models(raw_config: Optional[dict] = None) -> List[Dict[str, object]]:
    raw_config = load_yaml_config() if raw_config is None else raw_config
    model_section = raw_config.get("model", {})
    if not isinstance(model_section, dict):
        model_section = {}

    selected_model_id = model_section.get("id", DEFAULTS["model"]["id"])
    available = model_section.get("available")
    if available is None:
        available = raw_config.get("models")

    models = []
    if isinstance(available, list):
        for item in available:
            if isinstance(item, dict):
                model_id = item.get("id") or item.get("path") or item.get("name")
                if model_id:
                    models.append(_normalise_model_entry(str(model_id), item))
            elif item:
                models.append(_model_entry_from_id(str(item)))
    elif isinstance(available, dict):
        for model_id, item in available.items():
            models.append(_normalise_model_entry(str(model_id), item))

    if not models:
        models.append(_model_entry_from_id(str(selected_model_id)))

    known_ids = {str(model["id"]) for model in models}
    if str(selected_model_id) not in known_ids:
        models.insert(0, _model_entry_from_id(str(selected_model_id)))

    return models


def selected_model_id(raw_config: Optional[dict] = None) -> str:
    raw_config = load_yaml_config() if raw_config is None else raw_config
    model_section = raw_config.get("model", {})
    if not isinstance(model_section, dict):
        return DEFAULTS["model"]["id"]
    return str(model_section.get("id", DEFAULTS["model"]["id"]))


def configured_model(model_id: str, raw_config: Optional[dict] = None):
    for model in configured_models(raw_config):
        if str(model["id"]) == model_id:
            return model
    return None


def _yaml_scalar(value: str) -> str:
    return yaml.safe_dump(
        value,
        default_flow_style=True,
        width=1000000,
    ).splitlines()[0]


def _replace_selected_model_line(path: Path, model_id: str) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines(True)
    in_model_section = False
    model_indent = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        if stripped.startswith("model:"):
            in_model_section = True
            model_indent = indent
            continue

        if in_model_section and indent <= model_indent:
            in_model_section = False

        if in_model_section and stripped.startswith("id:"):
            prefix = line[:indent]
            line_body = line.rstrip("\n")
            comment = ""
            if "#" in line_body:
                comment = "  #" + line_body.split("#", 1)[1]
            lines[index] = "{}id: {}{}\n".format(
                prefix,
                _yaml_scalar(model_id),
                comment,
            )
            path.write_text("".join(lines), encoding="utf-8")
            return True

    return False


def update_selected_model(model_id: str, path: Path = CONFIG_PATH) -> None:
    raw_config = load_yaml_config(path)
    if configured_model(model_id, raw_config) is None:
        raise ValueError("Model is not configured: {}".format(model_id))

    if _replace_selected_model_line(path, model_id):
        return

    model_section = raw_config.setdefault("model", {})
    if not isinstance(model_section, dict):
        raise ValueError("model section must be a YAML mapping")
    model_section["id"] = model_id

    with path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(raw_config, config_file, sort_keys=False)


config = load_config()
settings = config

os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.backend.gpu)


if __name__ == "__main__":
    print("Host: {}".format(config.backend.host))
    print("Port: {}".format(config.backend.port))
    print("GPU: {}".format(config.backend.gpu))
    print("Model: {}".format(config.model.id))
