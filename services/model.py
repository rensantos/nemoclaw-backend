import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from config import (
    CONFIG_PATH,
    DEFAULTS,
    active_instance_config_path,
    load_layered_config,
    load_yaml_config,
    settings,
)

# Matches engines/ollama_engine.py's own _TAGS_TIMEOUT_SECONDS. Short on
# purpose: this runs inside /v1/models and every model validation, so a
# hung daemon must degrade to the configured catalog quickly rather than
# stall an API response.
_TAGS_TIMEOUT_SECONDS = 5
# Re-querying the daemon on every validate/list call would mean several
# HTTP round trips per switch. A few seconds is short enough that a model
# pulled by hand shows up almost immediately, long enough that one API
# call does not fan out into repeated requests.
_TAGS_CACHE_TTL_SECONDS = 5.0


def ollama_installed_model_ids(base_url: str) -> List[str]:
    """Tags the daemon actually has, or [] when it cannot be asked.

    Never raises: an unreachable daemon must degrade to whatever the YAML
    catalog says, not break listing and validation outright.
    """
    try:
        request = urllib.request.Request("{}/api/tags".format(base_url.rstrip("/")))
        with urllib.request.urlopen(request, timeout=_TAGS_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return []

    names = []
    for model in (payload or {}).get("models") or []:
        if isinstance(model, dict):
            name = model.get("name") or model.get("model")
            if name:
                names.append(str(name))
    return sorted(names)


class ModelManager:
    """Owns configured model metadata and selected-model configuration.

    For an Ollama backend the selectable catalog is derived from the
    DAEMON, not from a hand-maintained YAML list: whatever is pulled is
    selectable, and nothing else needs registering. The YAML catalog is
    still read and still supplies per-entry metadata, but it is no longer
    the source of truth for *which* models exist.

    That is a deliberate reversal. The old allowlist had to be kept in
    sync by hand and drifted in both directions - live on this desktop,
    16 tags were installed while the catalog named 4, so `/model
    gemma4:12b` 404'd "Model is not configured" for a model sitting on
    disk. Worse, the two halves disagreed with each other: register_model()
    appends to the SHARED config.yaml, but a per-machine
    config.<instance>.yaml REPLACES model.available wholesale during
    layering (see config._merge_sections), so the append was invisible to
    the very next read - a switch would succeed against the engine and
    then fail when persisting it.

    Reads are layered (see _load) and writes now follow them: a persisted
    change lands in THIS machine's own config.<instance>.yaml when it has
    one, and only in the shared config.yaml when it does not (see
    _write_path). Previously reads were layered while writes always hit
    the shared file, which was both ineffective (the per-machine layer
    overrode the write on the very next load) and actively disruptive (it
    dirtied a tracked file every other machine pulls - a real `git pull`
    on the laptop aborted because of it).
    """

    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        installed_models_provider=None,
    ):
        self.config_path = config_path
        # Injected by tests so they never touch a real daemon (project
        # rule). None means "decide from settings" - see
        # _installed_model_ids().
        self._installed_models_provider = installed_models_provider
        self._installed_cache: Optional[List[str]] = None
        self._installed_cached_at = 0.0

    def _write_path(self) -> Path:
        """Where a persisted change belongs: THIS machine's own config
        file when it has one, otherwise the shared config.yaml.

        Writes used to always target the shared file while reads were
        layered, which was actively harmful rather than merely untidy:

        - The change was usually INEFFECTIVE. A per-machine
          config.<instance>.yaml sets model.id, and that layer wins on the
          next load, so a persisted switch written to the shared file was
          overridden immediately and silently.
        - It dirtied a file every machine pulls. Observed live on
          2026-08-06: `git pull` on the laptop aborted with "Your local
          changes to the following files would be overwritten by merge:
          config/config.yaml", purely because model switches on other
          machines had been auto-writing their own model.id there.

        Only an existing per-machine file is used; this never creates one,
        so a machine that deliberately runs on the shared config keeps
        doing exactly that.
        """
        if self.config_path != CONFIG_PATH:
            # Explicit-path construction (tests, tooling) stays isolated,
            # mirroring _load() and _installed_model_ids().
            return self.config_path
        return active_instance_config_path() or self.config_path

    def _installed_model_ids(self) -> List[str]:
        """Model ids present on this machine's runtime, [] if unknowable.

        Only meaningful for Ollama: a Transformers repo is not "installed"
        in any way this can enumerate, so that engine keeps the YAML
        catalog as its sole source.

        A ModelManager built against an explicit config_path (tests,
        tooling) stays fully offline, mirroring _load()'s isolation of
        the same case.
        """
        if self._installed_models_provider is not None:
            return list(self._installed_models_provider())
        if self.config_path != CONFIG_PATH:
            return []
        if settings.backend.engine != "ollama":
            return []

        now = time.monotonic()
        if (
            self._installed_cache is not None
            and now - self._installed_cached_at < _TAGS_CACHE_TTL_SECONDS
        ):
            return self._installed_cache

        self._installed_cache = ollama_installed_model_ids(settings.backend.ollama_host)
        self._installed_cached_at = now
        return self._installed_cache

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

        write_path = self._write_path()
        if write_path != self.config_path:
            # A per-machine file exists but has no model.id line to
            # replace. Rewriting THAT file alone is safe; dumping
            # raw_config (the fully MERGED layered config) into it would
            # copy every shared default into this machine's overlay,
            # freezing them and defeating the point of layering.
            self._write_own_section(write_path, {"id": model_id})
            return

        model_section = raw_config.setdefault("model", {})
        if not isinstance(model_section, dict):
            raise ValueError("model section must be a YAML mapping")
        model_section["id"] = model_id

        with self.config_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(raw_config, config_file, sort_keys=False)

    def _write_own_section(self, path: Path, model_updates: Dict[str, object]) -> None:
        """Merge `model_updates` into `path`'s own model: section.

        Reads that ONE file (never the layered merge), so only keys this
        machine already declares plus the update survive.
        """
        own = load_yaml_config(path) if path.exists() else {}
        if not isinstance(own, dict):
            own = {}
        model_section = own.setdefault("model", {})
        if not isinstance(model_section, dict):
            raise ValueError("model section must be a YAML mapping")
        model_section.update(model_updates)
        with path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(own, config_file, sort_keys=False)

    def register_model(self, model_id: str) -> bool:
        """Adds a model to the selectable catalog, if not already there.

        Largely vestigial for an Ollama backend since live discovery
        landed: a freshly pulled tag is already selectable the moment the
        daemon reports it, so _configured_model() finds it here and this
        returns False without writing anything. That is the desired
        outcome, not a regression - it also stops every pull from
        dirtying the SHARED config.yaml on a machine whose real catalog
        lives in its own per-machine file.

        Still meaningful for a Transformers backend, which has no daemon
        to enumerate, and as an explicit way to pin an entry with real
        metadata.

        Returns True when the catalog was changed, False when the model was
        already selectable (so a repeat download is not an error).
        """
        raw_config = self._load()
        if self._configured_model(model_id, raw_config) is not None:
            return False

        model_section = raw_config.get("model")
        if model_section is not None and not isinstance(model_section, dict):
            raise ValueError("model section must be a YAML mapping")

        if self._append_available_entry(model_id):
            return True

        entry = {"id": model_id, "engine": "ollama", "notes": "added by /admin/model/pull"}

        write_path = self._write_path()
        if write_path != self.config_path:
            # Same reasoning as select_model()'s fallback: extend only
            # this machine's OWN available list, never a dump of the
            # merged config.
            own = load_yaml_config(write_path) if write_path.exists() else {}
            own_available = ((own or {}).get("model") or {}).get("available")
            own_available = list(own_available) if isinstance(own_available, list) else []
            own_available.append(entry)
            self._write_own_section(write_path, {"available": own_available})
            return True

        # Fall back to a full rewrite only when the surgical append could
        # not find the list to extend. This loses comments, so it is a last
        # resort rather than the normal path (see _append_available_entry).
        model_section = raw_config.setdefault("model", {})
        available = model_section.get("available")
        if not isinstance(available, list):
            available = []
        available.append(entry)
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
        write_path = self._write_path()
        if not write_path.exists():
            return False
        lines = write_path.read_text(encoding="utf-8").splitlines(True)

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
        write_path.write_text("".join(lines), encoding="utf-8")
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
            known_ids.add(str(selected_model_id))

        # Whatever the daemon actually has, catalogued or not. Appended
        # rather than prepended so a hand-written entry (which may carry
        # real metadata) keeps its position and its fields; this only ever
        # ADDS models that are genuinely present on disk.
        for model_id in self._installed_model_ids():
            if model_id not in known_ids:
                models.append(self._ollama_entry_from_id(model_id))
                known_ids.add(model_id)

        return models

    def _model_entry_from_id(self, model_id: str) -> Dict[str, object]:
        return {
            "id": model_id,
            "name": model_id,
            "path": model_id,
            "engine": "transformers",
            "device": "cuda",
        }

    def _ollama_entry_from_id(self, model_id: str) -> Dict[str, object]:
        """A live-discovered tag. engine must say "ollama" explicitly:
        InferenceService._servable_by_active_engine() drops entries whose
        engine does not match the running one, so borrowing
        _model_entry_from_id()'s "transformers" default would make every
        discovered model vanish from /v1/models again."""
        return {
            "id": model_id,
            "name": model_id,
            "path": model_id,
            "engine": "ollama",
            "notes": "installed on this machine (discovered from the Ollama daemon)",
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
        write_path = self._write_path()
        if not write_path.exists():
            return False
        lines = write_path.read_text(encoding="utf-8").splitlines(True)

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
                "active model.".format(write_path)
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
        write_path.write_text("".join(lines), encoding="utf-8")
        return True
