from pathlib import Path
from typing import Dict, List, Optional

import yaml

from config import CONFIG_PATH, DEFAULTS, load_yaml_config


class ModelManager:
    """Owns configured model metadata and selected-model configuration."""

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path

    def list_models(self) -> List[Dict[str, object]]:
        return self.available_models()

    def available_models(self) -> List[Dict[str, object]]:
        return self._configured_models(self._load())

    def current_model(self):
        raw_config = self._load()
        return self.model_info(self.selected_model_id(raw_config))

    def model_info(self, model_id: str):
        model = self._configured_model(model_id, self._load())
        if model is None:
            raise ValueError("Model is not configured: {}".format(model_id))
        return model

    def select_model(self, model_id: str) -> None:
        raw_config = self._load()
        self.validate_model(model_id, raw_config)

        if self._replace_selected_model_line(model_id):
            return

        model_section = raw_config.setdefault("model", {})
        if not isinstance(model_section, dict):
            raise ValueError("model section must be a YAML mapping")
        model_section["id"] = model_id

        with self.config_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(raw_config, config_file, sort_keys=False)

    def validate_model(self, model_id: str, raw_config: Optional[dict] = None) -> None:
        if self._configured_model(model_id, raw_config or self._load()) is None:
            raise ValueError("Model is not configured: {}".format(model_id))

    def selected_model_id(self, raw_config: Optional[dict] = None) -> str:
        raw_config = self._load() if raw_config is None else raw_config
        model_section = raw_config.get("model", {})
        if not isinstance(model_section, dict):
            return DEFAULTS["model"]["id"]
        return str(model_section.get("id", DEFAULTS["model"]["id"]))

    def _load(self):
        return load_yaml_config(self.config_path)

    def _configured_model(self, model_id: str, raw_config: Optional[dict] = None):
        for model in self._configured_models(raw_config or self._load()):
            if str(model["id"]) == model_id:
                return model
        return None

    def _configured_models(self, raw_config: Optional[dict] = None) -> List[Dict[str, object]]:
        raw_config = self._load() if raw_config is None else raw_config
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
                        models.append(self._normalise_model_entry(str(model_id), item))
                elif item:
                    models.append(self._model_entry_from_id(str(item)))
        elif isinstance(available, dict):
            for model_id, item in available.items():
                models.append(self._normalise_model_entry(str(model_id), item))

        if not models:
            models.append(self._model_entry_from_id(str(selected_model_id)))

        known_ids = {str(model["id"]) for model in models}
        if str(selected_model_id) not in known_ids:
            models.insert(0, self._model_entry_from_id(str(selected_model_id)))

        return models

    def _model_entry_from_id(self, model_id: str) -> Dict[str, object]:
        return {
            "id": model_id,
            "name": model_id,
            "path": model_id,
            "engine": "transformers",
            "device": "cuda",
        }

    def _normalise_model_entry(self, model_id: str, value) -> Dict[str, object]:
        if isinstance(value, dict):
            entry = dict(value)
            entry.setdefault("id", model_id)
        else:
            entry = self._model_entry_from_id(model_id)

        entry.setdefault("name", entry["id"])
        entry.setdefault("path", entry["id"])
        entry.setdefault("engine", "transformers")
        entry.setdefault("device", "cuda")
        return entry

    def _yaml_scalar(self, value: str) -> str:
        return yaml.safe_dump(
            value,
            default_flow_style=True,
            width=1000000,
        ).splitlines()[0]

    def _replace_selected_model_line(self, model_id: str) -> bool:
        lines = self.config_path.read_text(encoding="utf-8").splitlines(True)

        in_model_section = False
        model_indent = None
        child_indent = None
        match_index = None
        ambiguous = False

        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip(" "))

            if not in_model_section:
                if stripped.startswith("model:"):
                    in_model_section = True
                    model_indent = indent
                continue

            if indent <= model_indent:
                # Left the model: mapping entirely.
                break

            if child_indent is None:
                # The first key encountered directly under model: fixes the
                # indentation depth of model's own keys (id, available, ...).
                child_indent = indent

            if indent != child_indent:
                # Nested content of a direct child's value (e.g. an entry of
                # available:), never the top-level model.id.
                continue

            if stripped.startswith("id:"):
                if match_index is not None:
                    ambiguous = True
                else:
                    match_index = index

        if ambiguous:
            raise ValueError(
                "Ambiguous 'id:' entries found directly under the model: "
                "section of {}; refusing to guess which one selects the "
                "active model.".format(self.config_path)
            )

        if match_index is None:
            return False

        line = lines[match_index]
        indent = len(line) - len(line.lstrip(" "))
        prefix = line[:indent]
        line_body = line.rstrip("\n")
        comment = ""
        if "#" in line_body:
            comment = "  #" + line_body.split("#", 1)[1]
        lines[match_index] = "{}id: {}{}\n".format(
            prefix,
            self._yaml_scalar(model_id),
            comment,
        )
        self.config_path.write_text("".join(lines), encoding="utf-8")
        return True
