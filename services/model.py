from pathlib import Path
from typing import Dict, List, Optional

import yaml

from config import CONFIG_PATH, DEFAULTS, load_layered_config, load_yaml_config


class ModelManager:
    """Owns configured model metadata and selected-model configuration.

    Reads are layered (see _load); writes still target config.yaml. That
    asymmetry is deliberate for now and is the remaining half of the
    per-machine config work: redirecting writes means splitting this
    class's read-mutate-write cycle, whose comment-preserving line
    editing is delicate enough that breaking it breaks model
    registration. Consequence today: a persisted `model switch` or a
    model auto-registered after a pull still lands in the SHARED
    config.yaml rather than this machine's own file.
    """

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

    def register_model(self, model_id: str) -> bool:
        """Adds a model to the selectable catalog, if not already there.

        A freshly downloaded tag is useless without this: /v1/models
        enumerates model.available, and validate_model() rejects anything
        absent from it, so the model would be invisible and unselectable.

        Returns True when the catalog was changed, False when the model was
        already listed (so a repeat download is not an error).
        """
        raw_config = self._load()
        if self._configured_model(model_id, raw_config) is not None:
            return False

        model_section = raw_config.get("model")
        if model_section is not None and not isinstance(model_section, dict):
            raise ValueError("model section must be a YAML mapping")

        if self._append_available_entry(model_id):
            return True

        # Fall back to a full rewrite only when the surgical append could
        # not find the list to extend. This loses comments, so it is a last
        # resort rather than the normal path (see _append_available_entry).
        model_section = raw_config.setdefault("model", {})
        available = model_section.get("available")
        if not isinstance(available, list):
            available = []
        available.append(
            {"id": model_id, "engine": "ollama", "notes": "added by /admin/model/pull"}
        )
        model_section["available"] = available

        with self.config_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(raw_config, config_file, sort_keys=False)
        return True

    def _append_available_entry(self, model_id: str) -> bool:
        """Appends to model.available by editing lines, not by re-dumping.

        yaml.safe_dump() rewrites the whole document, which strips every
        comment and re-quotes scalars - observed live: a single registered
        model deleted 114 lines of hand-written operational notes and
        silently turned `gpu: "2,3"` into unquoted `gpu: 2,3`. This config
        carries hard-won deployment knowledge, so a routine append must not
        destroy it.

        Mirrors _replace_selected_model_line()'s approach. Returns False if
        the expected structure is not found, leaving the caller to decide.
        """
        lines = self.config_path.read_text(encoding="utf-8").splitlines(True)

        in_model_section = False
        model_indent = None
        available_indent = None
        insert_at = None
        item_indent = "    "

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
                break  # left the model: mapping

            if available_indent is None:
                if stripped.startswith("available:"):
                    available_indent = indent
                continue

            if indent <= available_indent:
                break  # left the available: list
            if stripped.startswith("- "):
                item_indent = " " * indent
            insert_at = index + 1

        if available_indent is None or insert_at is None:
            return False

        entry = (
            f"{item_indent}- id: {model_id}\n"
            f"{item_indent}  path: {model_id}\n"
            f"{item_indent}  engine: ollama\n"
            f"{item_indent}  notes: added automatically when this model was installed\n"
        )
        lines.insert(insert_at, entry)
        self.config_path.write_text("".join(lines), encoding="utf-8")
        return True

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
        """Reads see the LAYERED config (shared + this machine's file +
        config.local.yaml), not just config.yaml.

        Without this, a machine whose model catalog lives in its own
        per-machine file would serve a model that /v1/models does not
        list and validate_model() rejects - observed exactly that after
        moving this desktop's drift out of the shared file.

        Only the default path is layered: a ModelManager constructed
        against an explicit path (tests, tooling) reads that file alone,
        so it stays isolated.

        NOTE the asymmetry - writes still go to self.config_path. See the
        class docstring.
        """
        if self.config_path == CONFIG_PATH:
            return load_layered_config()
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
