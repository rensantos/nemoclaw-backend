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

    def test_ollama_host_defaults_when_unset(self):
        result = self._load({})

        self.assertEqual(result.backend.ollama_host, "http://127.0.0.1:11434")

    def test_ollama_host_reads_yaml_value(self):
        result = self._load(
            {"backend": {"ollama_host": "http://10.0.0.5:11434"}}
        )

        self.assertEqual(result.backend.ollama_host, "http://10.0.0.5:11434")

    def test_ollama_host_env_override_takes_precedence_over_yaml(self):
        result = self._load(
            {"backend": {"ollama_host": "http://10.0.0.5:11434"}},
            env={"OLLAMA_HOST": "http://192.168.1.50:11434"},
        )

        self.assertEqual(result.backend.ollama_host, "http://192.168.1.50:11434")

    def test_quantization_defaults_to_none_when_unset(self):
        result = self._load({})

        self.assertEqual(result.model.quantization, "none")

    def test_quantization_reads_yaml_value(self):
        result = self._load({"model": {"quantization": "4bit"}})

        self.assertEqual(result.model.quantization, "4bit")

    def test_quantization_env_override_takes_precedence_over_yaml(self):
        result = self._load(
            {"model": {"quantization": "none"}},
            env={"MODEL_QUANTIZATION": "8bit"},
        )

        self.assertEqual(result.model.quantization, "8bit")

    def test_invalid_quantization_value_in_yaml_raises(self):
        with self.assertRaises(ValueError):
            self._load({"model": {"quantization": "2bit"}})

    def test_invalid_quantization_env_override_raises(self):
        with self.assertRaises(ValueError):
            self._load({}, env={"MODEL_QUANTIZATION": "2bit"})

    def test_revision_defaults_to_empty_when_unset(self):
        result = self._load({})

        self.assertEqual(result.model.revision, "")

    def test_revision_reads_yaml_value(self):
        result = self._load({"model": {"revision": "abc123"}})

        self.assertEqual(result.model.revision, "abc123")

    def test_revision_env_override_takes_precedence_over_yaml(self):
        result = self._load(
            {"model": {"revision": "abc123"}},
            env={"MODEL_REVISION": "def456"},
        )

        self.assertEqual(result.model.revision, "def456")


if __name__ == "__main__":
    unittest.main()
