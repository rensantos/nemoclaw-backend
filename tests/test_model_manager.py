from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

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


if __name__ == "__main__":
    unittest.main()
