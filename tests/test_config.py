import os
import unittest
from unittest import mock

import config as config_module


class ConfigEngineSelectionTests(unittest.TestCase):
    def _load(self, raw_config, env=None):
        with mock.patch("config._load_yaml_config", return_value=raw_config):
            with mock.patch.dict(os.environ, env or {}, clear=True):
                return config_module.load_config()

    def test_engine_defaults_to_transformers_when_unset(self):
        result = self._load({"backend": {"host": "127.0.0.1", "port": 8000, "gpu": 0}})

        self.assertEqual(result.backend.engine, "transformers")

    def test_engine_reads_yaml_value(self):
        result = self._load({"backend": {"engine": "ollama"}})

        self.assertEqual(result.backend.engine, "ollama")

    def test_engine_env_override_takes_precedence_over_yaml(self):
        result = self._load(
            {"backend": {"engine": "transformers"}},
            env={"ENGINE": "ollama"},
        )

        self.assertEqual(result.backend.engine, "ollama")

    def test_invalid_engine_value_in_yaml_raises(self):
        with self.assertRaises(ValueError):
            self._load({"backend": {"engine": "bogus"}})

    def test_invalid_engine_env_override_raises(self):
        with self.assertRaises(ValueError):
            self._load({}, env={"ENGINE": "bogus"})


if __name__ == "__main__":
    unittest.main()
