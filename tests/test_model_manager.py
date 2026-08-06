from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import yaml

from services.model import ModelManager


class ModelManagerTests(unittest.TestCase):
    def test_available_models_falls_back_to_legacy_selected_model(self):
        manager = ModelManager()
        raw_config = {"model": {"id": "legacy-model"}}

        models = manager._configured_models(raw_config)

        self.assertEqual(models[0]["id"], "legacy-model")
        self.assertEqual(models[0]["engine"], "transformers")

    def test_model_info_finds_available_entry(self):
        manager = ModelManager()
        raw_config = {
            "model": {
                "id": "tiny",
                "available": [
                    {"id": "tiny", "name": "Tiny"},
                    {"id": "other", "name": "Other"},
                ],
            }
        }

        model = manager._configured_model("other", raw_config)

        self.assertEqual(model["name"], "Other")
        self.assertEqual(model["engine"], "transformers")

    def test_selected_model_id_reads_model_section(self):
        manager = ModelManager()
        raw_config = {"model": {"id": "tiny"}}

        self.assertEqual(manager.selected_model_id(raw_config), "tiny")

    def test_select_model_updates_yaml_file(self):
        config_text = """# keep this comment
backend:
  host: 127.0.0.1
  port: 8000
  gpu: 0

model:
  id: tiny  # selected model
  available:
    - id: tiny
    - id: other
"""

        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(config_text, encoding="utf-8")
            manager = ModelManager(config_path)

            manager.select_model("other")

            updated_text = config_path.read_text(encoding="utf-8")
            updated = yaml.safe_load(updated_text)

        self.assertEqual(updated["model"]["id"], "other")
        self.assertEqual(len(updated["model"]["available"]), 2)
        self.assertIn("# keep this comment", updated_text)
        self.assertIn("# selected model", updated_text)

    def test_select_model_ignores_available_entry_id_on_continuation_line(self):
        # Regression test for docs/audit-2dabb09.md: available: declared
        # before id:, with the first entry's id: on a continuation line
        # (block sequence dash on its own line, keys indented below it).
        # The line scanner must match model.id, not available[0].id.
        config_text = """model:
  available:
    -
      id: tiny
      path: /models/tiny
    - id: other
  id: tiny
"""

        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(config_text, encoding="utf-8")
            manager = ModelManager(config_path)

            manager.select_model("other")

            updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(updated["model"]["id"], "other")
        self.assertEqual(updated["model"]["available"][0]["id"], "tiny")
        self.assertEqual(updated["model"]["available"][0]["path"], "/models/tiny")

    def test_select_model_raises_on_ambiguous_id_lines(self):
        config_text = """model:
  id: tiny
  id: tiny
  available:
    - id: tiny
"""

        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(config_text, encoding="utf-8")
            manager = ModelManager(config_path)

            with self.assertRaises(ValueError):
                manager.select_model("tiny")

    def test_select_model_rejects_unknown_model(self):
        raw_config = {
            "model": {
                "id": "tiny",
                "available": [{"id": "tiny"}],
            }
        }

        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
            manager = ModelManager(config_path)

            with self.assertRaises(ValueError):
                manager.select_model("missing")

    def test_current_model_returns_selected_model_metadata(self):
        raw_config = {
            "model": {
                "id": "tiny",
                "available": [{"id": "tiny", "name": "Tiny"}],
            }
        }

        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
            manager = ModelManager(config_path)

            model = manager.current_model()

        self.assertEqual(model["id"], "tiny")
        self.assertEqual(model["name"], "Tiny")


class RegisterModelTests(unittest.TestCase):
    """config.yaml carries hand-written deployment knowledge. Registering a
    model is a routine append and must not destroy it - observed live, a
    single register_model() call deleted 114 comment lines and silently
    rewrote `gpu: "2,3"` as unquoted `gpu: 2,3`."""

    COMMENTED_CONFIG = """backend:
  # GPU 0/1 belong to another user's job - do not touch.
  gpu: "2,3"
  engine: ollama
model:
  # The live default.
  id: tiny
  available:
    # Catalogued models follow.
    - id: tiny
      path: tiny
      engine: ollama
"""

    def _manager(self, tmp_dir):
        config_path = Path(tmp_dir) / "config.yaml"
        config_path.write_text(self.COMMENTED_CONFIG, encoding="utf-8")
        return ModelManager(config_path), config_path

    def test_registering_preserves_comments_and_quoting(self):
        with TemporaryDirectory() as tmp_dir:
            manager, config_path = self._manager(tmp_dir)

            self.assertTrue(manager.register_model("qwen3:4b"))
            after = config_path.read_text(encoding="utf-8")

        self.assertIn("# GPU 0/1 belong to another user's job", after)
        self.assertIn("# The live default.", after)
        self.assertIn("# Catalogued models follow.", after)
        # Unquoting this would change how the value parses.
        self.assertIn('gpu: "2,3"', after)

    def test_registering_adds_a_usable_catalog_entry(self):
        with TemporaryDirectory() as tmp_dir:
            manager, config_path = self._manager(tmp_dir)
            manager.register_model("qwen3:4b")

            parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            ids = [entry["id"] for entry in parsed["model"]["available"]]

            # Must be selectable afterwards, which is the whole point.
            manager.validate_model("qwen3:4b")

        self.assertIn("qwen3:4b", ids)
        self.assertIn("tiny", ids)

    def test_registering_a_known_model_changes_nothing(self):
        with TemporaryDirectory() as tmp_dir:
            manager, config_path = self._manager(tmp_dir)
            before = config_path.read_text(encoding="utf-8")

            self.assertFalse(manager.register_model("tiny"))

            self.assertEqual(config_path.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()


class LayeredReadTests(unittest.TestCase):
    """ModelManager reads must see the per-machine layers, or a machine
    serves a model that /v1/models does not list and validate_model()
    rejects - observed exactly that after moving a desktop's drift out of
    the shared config.yaml."""

    def test_default_path_reads_the_layered_config(self):
        layered = {"model": {"id": "deepseek-r1:14b",
                             "available": [{"id": "deepseek-r1:14b", "path": "deepseek-r1:14b",
                                            "engine": "ollama"}]}}

        # installed_models_provider stubbed empty so this stays a test of
        # LAYERING alone: without it, live daemon discovery (see
        # ModelManager._installed_model_ids) would union this machine's
        # real tags in and the assertion would depend on what happens to
        # be pulled - and would hit a real daemon, which tests must not.
        with mock.patch("services.model.load_layered_config", return_value=layered) as layered_loader:
            manager = ModelManager(installed_models_provider=lambda: [])
            ids = [m["id"] for m in manager.available_models()]

        layered_loader.assert_called()
        self.assertEqual(ids, ["deepseek-r1:14b"])

    def test_an_explicit_path_stays_isolated(self):
        """Tests and tooling that name a file must read that file alone."""
        with mock.patch("services.model.load_layered_config") as layered_loader:
            with mock.patch("services.model.load_yaml_config", return_value={"model": {"available": []}}):
                ModelManager(config_path=Path("/tmp/somewhere-else.yaml")).available_models()

        layered_loader.assert_not_called()


class InstalledModelDiscoveryTests(unittest.TestCase):
    """The selectable catalog for an Ollama backend comes from the DAEMON,
    not a hand-maintained YAML list.

    Live on 2026-08-06 this desktop had 16 tags pulled and 4 catalogued,
    so `/model gemma4:12b` returned 404 "Model is not configured" for a
    model sitting on disk ready to serve. Worse, the two halves of the
    old design disagreed with each other: register_model() appends to the
    SHARED config.yaml, but a per-machine config.<instance>.yaml REPLACES
    model.available wholesale when layered (config._merge_sections), so
    the append was invisible to the very next read - the switch succeeded
    against the engine and then failed while persisting.
    """

    def _layered(self, available_ids):
        return {
            "model": {
                "id": "catalogued-a",
                "available": [
                    {"id": model_id, "path": model_id, "engine": "ollama"}
                    for model_id in available_ids
                ],
            }
        }

    def _manager(self, available_ids, installed):
        manager = ModelManager(installed_models_provider=lambda: installed)
        return manager

    def test_installed_but_uncatalogued_models_are_selectable(self):
        with mock.patch(
            "services.model.load_layered_config", return_value=self._layered(["catalogued-a"])
        ):
            manager = self._manager(["catalogued-a"], ["catalogued-a", "pulled-by-hand"])
            ids = [m["id"] for m in manager.available_models()]

            self.assertIn("pulled-by-hand", ids)
            # The real failure this fixes: validation used to reject it.
            manager.validate_model("pulled-by-hand")

    def test_discovered_entries_are_tagged_as_ollama(self):
        """InferenceService._servable_by_active_engine() drops entries
        whose engine does not match the running one, so a discovered model
        inheriting the "transformers" default would vanish from
        /v1/models - the exact symptom being fixed."""
        with mock.patch(
            "services.model.load_layered_config", return_value=self._layered(["catalogued-a"])
        ):
            manager = self._manager(["catalogued-a"], ["pulled-by-hand"])
            entry = manager._configured_model("pulled-by-hand")

        self.assertEqual(entry["engine"], "ollama")

    def test_catalogued_entries_keep_their_own_metadata(self):
        """Discovery only ADDS; it must not overwrite a hand-written entry
        that carries real metadata."""
        layered = {
            "model": {
                "id": "catalogued-a",
                "available": [
                    {"id": "catalogued-a", "path": "custom/path", "engine": "ollama",
                     "notes": "hand written"}
                ],
            }
        }
        with mock.patch("services.model.load_layered_config", return_value=layered):
            manager = self._manager(["catalogued-a"], ["catalogued-a"])
            entry = manager._configured_model("catalogued-a")

        self.assertEqual(entry["path"], "custom/path")
        self.assertEqual(entry["notes"], "hand written")

    def test_no_duplicates_when_catalogued_and_installed(self):
        with mock.patch(
            "services.model.load_layered_config", return_value=self._layered(["both"])
        ):
            manager = self._manager(["both"], ["both"])
            ids = [m["id"] for m in manager.available_models()]

        self.assertEqual(ids.count("both"), 1)

    def test_an_unreachable_daemon_degrades_to_the_catalog(self):
        """A daemon that cannot be asked must not empty the catalog and
        make every model unselectable."""
        with mock.patch(
            "services.model.load_layered_config", return_value=self._layered(["catalogued-a"])
        ):
            manager = ModelManager(installed_models_provider=lambda: [])
            ids = [m["id"] for m in manager.available_models()]

        self.assertEqual(ids, ["catalogued-a"])

    def test_register_model_is_a_no_op_for_an_installed_model(self):
        """Nothing needs writing once the model is already discoverable -
        which is also what stops every pull from dirtying config.yaml."""
        with mock.patch(
            "services.model.load_layered_config", return_value=self._layered(["catalogued-a"])
        ):
            manager = self._manager(["catalogued-a"], ["catalogued-a", "pulled-by-hand"])
            self.assertFalse(manager.register_model("pulled-by-hand"))

    def test_tag_parsing_ignores_malformed_entries(self):
        from services.model import ollama_installed_model_ids

        payload = {"models": [{"name": "a:1"}, {"model": "b:2"}, {}, "junk", None]}
        fake = mock.MagicMock()
        fake.__enter__ = mock.Mock(return_value=fake)
        fake.__exit__ = mock.Mock(return_value=False)
        with mock.patch("services.model.urllib.request.urlopen", return_value=fake):
            with mock.patch("services.model.json.load", return_value=payload):
                self.assertEqual(ollama_installed_model_ids("http://x:11434"), ["a:1", "b:2"])

    def test_an_unreachable_daemon_returns_empty_rather_than_raising(self):
        from services.model import ollama_installed_model_ids

        with mock.patch(
            "services.model.urllib.request.urlopen", side_effect=OSError("refused")
        ):
            self.assertEqual(ollama_installed_model_ids("http://x:11434"), [])


class PerMachineWriteTests(unittest.TestCase):
    """Persisted changes belong in THIS machine's config.<instance>.yaml.

    Writing them to the shared config.yaml (the old behaviour) was both
    ineffective - the per-machine layer overrides model.id on the very
    next load - and disruptive: a real `git pull` on the laptop aborted
    on 2026-08-06 with "Your local changes to the following files would
    be overwritten by merge: config/config.yaml", purely because model
    switches on other machines had auto-written their own model.id there.
    """

    SHARED = """\
model:
  id: shared-default
  available:
    - id: shared-default
      path: shared-default
      engine: ollama
"""

    PER_MACHINE = """\
# This machine's own file.
model:
  # The live default here.
  id: machine-default
"""

    def test_selected_model_is_written_to_the_per_machine_file(self):
        with TemporaryDirectory() as tmp_dir:
            shared = Path(tmp_dir) / "config.yaml"
            per_machine = Path(tmp_dir) / "config.thismachine.yaml"
            shared.write_text(self.SHARED, encoding="utf-8")
            per_machine.write_text(self.PER_MACHINE, encoding="utf-8")

            manager = ModelManager(installed_models_provider=lambda: ["other-model"])
            with mock.patch.object(manager, "config_path", shared), \
                 mock.patch.object(manager, "_write_path", return_value=per_machine), \
                 mock.patch("services.model.load_layered_config",
                            return_value={"model": {"id": "machine-default"}}):
                manager.select_model("other-model")

            self.assertIn("other-model", per_machine.read_text(encoding="utf-8"))
            # The shared file must be untouched - this is the whole point.
            self.assertEqual(shared.read_text(encoding="utf-8"), self.SHARED)

    def test_the_per_machine_files_comments_survive(self):
        with TemporaryDirectory() as tmp_dir:
            shared = Path(tmp_dir) / "config.yaml"
            per_machine = Path(tmp_dir) / "config.thismachine.yaml"
            shared.write_text(self.SHARED, encoding="utf-8")
            per_machine.write_text(self.PER_MACHINE, encoding="utf-8")

            manager = ModelManager(installed_models_provider=lambda: ["other-model"])
            with mock.patch.object(manager, "config_path", shared), \
                 mock.patch.object(manager, "_write_path", return_value=per_machine), \
                 mock.patch("services.model.load_layered_config",
                            return_value={"model": {"id": "machine-default"}}):
                manager.select_model("other-model")

            written = per_machine.read_text(encoding="utf-8")
            self.assertIn("# This machine's own file.", written)
            self.assertIn("# The live default here.", written)

    def test_without_a_per_machine_file_the_shared_one_is_still_written(self):
        """A machine that deliberately runs on the shared config keeps
        doing exactly that; this never invents a per-machine file."""
        with TemporaryDirectory() as tmp_dir:
            shared = Path(tmp_dir) / "config.yaml"
            shared.write_text(self.SHARED, encoding="utf-8")

            manager = ModelManager(config_path=shared,
                                   installed_models_provider=lambda: ["shared-default"])
            manager.select_model("shared-default")

            self.assertIn("shared-default", shared.read_text(encoding="utf-8"))
