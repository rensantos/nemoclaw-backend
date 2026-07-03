from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from config import configured_model, configured_models, selected_model_id, update_selected_model


class ConfigModelTests(unittest.TestCase):
    def test_configured_models_falls_back_to_legacy_selected_model(self):
        raw_config = {"model": {"id": "legacy-model"}}

        models = configured_models(raw_config)

        self.assertEqual(models[0]["id"], "legacy-model")
        self.assertEqual(models[0]["engine"], "transformers")

    def test_configured_model_finds_available_entry(self):
        raw_config = {
            "model": {
                "id": "tiny",
                "available": [
                    {"id": "tiny", "name": "Tiny"},
                    {"id": "other", "name": "Other"},
                ],
            }
        }

        model = configured_model("other", raw_config)

        self.assertEqual(model["name"], "Other")
        self.assertEqual(model["engine"], "transformers")

    def test_selected_model_id_reads_model_section(self):
        raw_config = {"model": {"id": "tiny"}}

        self.assertEqual(selected_model_id(raw_config), "tiny")

    def test_update_selected_model_updates_yaml_file(self):
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

            update_selected_model("other", config_path)

            updated_text = config_path.read_text(encoding="utf-8")
            updated = yaml.safe_load(updated_text)

        self.assertEqual(updated["model"]["id"], "other")
        self.assertEqual(len(updated["model"]["available"]), 2)
        self.assertIn("# keep this comment", updated_text)
        self.assertIn("# selected model", updated_text)

    def test_update_selected_model_rejects_unknown_model(self):
        raw_config = {
            "model": {
                "id": "tiny",
                "available": [{"id": "tiny"}],
            }
        }

        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")

            with self.assertRaises(ValueError):
                update_selected_model("missing", config_path)


if __name__ == "__main__":
    unittest.main()
