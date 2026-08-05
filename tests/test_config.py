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

    def test_think_default_is_none_when_unset(self):
        result = self._load({})

        self.assertIsNone(result.model.think_default)

    def test_think_default_reads_yaml_false(self):
        result = self._load({"model": {"think_default": False}})

        self.assertFalse(result.model.think_default)

    def test_think_default_reads_yaml_true(self):
        result = self._load({"model": {"think_default": True}})

        self.assertTrue(result.model.think_default)

    def test_think_default_env_override_takes_precedence_over_yaml(self):
        result = self._load(
            {"model": {"think_default": True}},
            env={"MODEL_THINK_DEFAULT": "false"},
        )

        self.assertFalse(result.model.think_default)

    def test_think_default_invalid_env_value_raises(self):
        with self.assertRaises(ValueError):
            self._load({}, env={"MODEL_THINK_DEFAULT": "not-a-bool"})


if __name__ == "__main__":
    unittest.main()


class InstanceIdentityTests(unittest.TestCase):
    """Every backend instance otherwise returns an identical /health shape,
    so with several reachable at once there is no way to tell them apart."""

    def _load(self, raw_config, env=None):
        with mock.patch("config._load_yaml_config", return_value=raw_config):
            with mock.patch.dict(os.environ, env or {}, clear=True):
                return config_module.load_config()

    def test_defaults_to_the_machine_hostname(self):
        import socket

        result = self._load({"backend": {}})

        self.assertEqual(result.backend.instance, socket.gethostname())

    def test_config_value_overrides_the_hostname(self):
        result = self._load({"backend": {"instance": "ubi"}})

        self.assertEqual(result.backend.instance, "ubi")

    def test_env_var_wins_over_config(self):
        result = self._load({"backend": {"instance": "ubi"}}, env={"INSTANCE": "zerob"})

        self.assertEqual(result.backend.instance, "zerob")

    def test_blank_value_falls_back_to_the_hostname(self):
        import socket

        result = self._load({"backend": {"instance": "   "}})

        self.assertEqual(result.backend.instance, socket.gethostname())


class LocalConfigOverlayTests(unittest.TestCase):
    """config.yaml is tracked and shared; config.local.yaml is gitignored
    and holds only what differs on this machine."""

    def _load(self, base, local, env=None):
        def fake_load(path=config_module.CONFIG_PATH):
            return local if path == config_module.CONFIG_LOCAL_PATH else base

        with mock.patch("config.load_yaml_config", side_effect=fake_load):
            with mock.patch.dict(os.environ, env or {}, clear=True):
                return config_module.load_config()

    def test_overlay_value_wins_over_the_tracked_file(self):
        result = self._load({"backend": {"port": 8000}}, {"backend": {"port": 8001}})

        self.assertEqual(result.backend.port, 8001)

    def test_keys_absent_from_the_overlay_fall_through(self):
        result = self._load(
            {"backend": {"port": 8000, "gpu": "2,3", "engine": "ollama"}},
            {"backend": {"port": 8001}},
        )

        self.assertEqual(result.backend.port, 8001)
        self.assertEqual(result.backend.gpu, "2,3", "untouched keys keep the shared value")
        self.assertEqual(result.backend.engine, "ollama")

    def test_a_section_missing_from_the_overlay_is_untouched(self):
        result = self._load(
            {"backend": {"port": 8000}, "model": {"id": "qwen3:30b"}},
            {"backend": {"port": 8001}},
        )

        self.assertEqual(result.model.id, "qwen3:30b")

    def test_no_overlay_file_behaves_exactly_as_before(self):
        result = self._load({"backend": {"port": 8000}, "model": {"id": "qwen3:30b"}}, {})

        self.assertEqual(result.backend.port, 8000)
        self.assertEqual(result.model.id, "qwen3:30b")

    def test_env_vars_still_win_over_the_overlay(self):
        result = self._load(
            {"backend": {"port": 8000}}, {"backend": {"port": 8001}}, env={"PORT": "8002"}
        )

        self.assertEqual(result.backend.port, 8002)

    def test_overlay_can_name_the_instance_without_touching_the_tracked_file(self):
        result = self._load({"backend": {}}, {"backend": {"instance": "zetopi"}})

        self.assertEqual(result.backend.instance, "zetopi")
