import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


CONFIG_DIR = Path(__file__).resolve().parent / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
# Untracked per-machine overrides layered over CONFIG_PATH. See
# load_layered_config() for why, and for what it deliberately does not cover.
CONFIG_LOCAL_PATH = CONFIG_DIR / "config.local.yaml"

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


def _merge_sections(base: dict, overlay: dict) -> dict:
    """Overlay wins per key, one level into each section.

    Deliberately not a deep recursive merge: the config is two levels
    (section -> key) plus model.available, which is a *list*. Merging a
    list element-wise would be guesswork, so an overlay that sets
    model.available replaces it wholesale, and one that does not leaves
    the base catalog untouched.
    """
    merged = {section: dict(values) if isinstance(values, dict) else values
              for section, values in base.items()}
    for section, values in overlay.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section] = {**merged[section], **values}
        else:
            merged[section] = values
    return merged


def instance_config_candidates(instance: str) -> list[Path]:
    """Per-machine config filenames to try, best match first.

    A hostname like "a4000.ipa.test" yields both
    config/config.a4000.ipa.test.yaml and the friendlier
    config/config.a4000.yaml, so a machine can be named by its short
    hostname without setting anything.
    """
    name = re.sub(r"[^A-Za-z0-9._-]", "-", (instance or "").strip()).strip(".-")
    if not name:
        return []
    candidates = [CONFIG_DIR / "config.{}.yaml".format(name)]
    head = name.split(".", 1)[0]
    if head and head != name:
        candidates.append(CONFIG_DIR / "config.{}.yaml".format(head))
    return candidates


def _resolve_instance(raw_config) -> str:
    """The machine's name, before the per-machine file is loaded.

    Chicken-and-egg: the instance name selects which config file to load,
    but is itself a config value. Resolved here from the sources that are
    available without that file - the INSTANCE env var, the shared
    config's own backend.instance, then the hostname.
    """
    return str(
        _env_value("INSTANCE", _section_value(raw_config, "backend", "instance")) or ""
    ).strip() or socket.gethostname()


def load_layered_config():
    """The shared config with per-machine layers on top.

    Three layers, lowest first - all optional except the base:

    1. `config/config.yaml` - **tracked**. Defaults shared by every
       machine.
    2. `config/config.<instance>.yaml` - **tracked**. This machine's
       differences (GPU, port, model). Tracked deliberately: every
       machine's real configuration is then versioned and readable from
       any other machine, which an untracked file would not give. Each
       machine only ever edits its own file, so they cannot conflict with
       each other the way one shared `config.local.yaml` would - that is
       the whole reason the filename carries the instance name.
    3. `config/config.local.yaml` - **untracked**. Anything that must not
       be committed, or a scratch override. Highest priority.

    Env vars still beat all three (see load_config).

    KNOWN GAP: this is the read path. ModelManager still *writes* into
    config.yaml (persisted model switches, catalog entries auto-added
    after a pull), because it reads-mutates-writes the whole document
    with comment-preserving line editing. So a persisted switch dirties
    the shared file rather than this machine's own - and a model.id set
    in a layer above overrides it on the next load. Per machine, use one
    or the other, not both.
    """
    merged = base = load_yaml_config()
    # Look up by the resolved instance AND by the bare hostname. A machine
    # launched with INSTANCE=zerob must find config.zerob.yaml, but the same
    # machine invoked without that env (./backend status, a cron job) must
    # still find its own file - otherwise it would silently fall back to the
    # shared defaults and report a different GPU or model than it serves.
    # Naming the file after the hostname therefore always works, and such a
    # file can set backend.instance itself to get a friendlier name.
    seen: set[Path] = set()
    for path in [*instance_config_candidates(_resolve_instance(base)),
                 *instance_config_candidates(socket.gethostname())]:
        if path in seen:
            continue
        seen.add(path)
        per_machine = load_yaml_config(path)
        if per_machine:
            merged = _merge_sections(merged, per_machine)
            break
    local = load_yaml_config(CONFIG_LOCAL_PATH)
    if local:
        merged = _merge_sections(merged, local)
    return merged


def active_instance_config_path() -> Optional[Path]:
    """This machine's own per-machine config file, if it has one.

    The same lookup load_layered_config() uses to decide which overlay to
    READ, exposed so writes can target the same file instead of the shared
    one. Returns None when this machine has no per-machine file, in which
    case the shared config.yaml is genuinely the right place to write.

    Only an existing file counts: creating one as a side effect of a model
    switch would invent a tracked, committed file the operator never asked
    for.
    """
    base = load_yaml_config()
    seen: set[Path] = set()
    for path in [*instance_config_candidates(_resolve_instance(base)),
                 *instance_config_candidates(socket.gethostname())]:
        if path in seen:
            continue
        seen.add(path)
        if path.exists() and load_yaml_config(path):
            return path
    return None


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
    raw_config = load_layered_config()

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
