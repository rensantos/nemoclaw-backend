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

        with mock.patch("services.model.load_layered_config", return_value=layered) as layered_loader:
            ids = [m["id"] for m in ModelManager().available_models()]

        layered_loader.assert_called()
        self.assertEqual(ids, ["deepseek-r1:14b"])

    def test_an_explicit_path_stays_isolated(self):
        """Tests and tooling that name a file must read that file alone."""
        with mock.patch("services.model.load_layered_config") as layered_loader:
            with mock.patch("services.model.load_yaml_config", return_value={"model": {"available": []}}):
                ModelManager(config_path=Path("/tmp/somewhere-else.yaml")).available_models()

        layered_loader.assert_not_called()
