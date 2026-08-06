import os
import unittest
from unittest import mock

import config as config_module


class ConfigEngineSelectionTests(unittest.TestCase):
    def _load(self, raw_config, env=None):
        with mock.patch("config.load_layered_config", return_value=raw_config):
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
        with mock.patch("config.load_layered_config", return_value=raw_config):
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


class InstanceConfigFileTests(unittest.TestCase):
    """Per-machine config lives in config.<instance>.yaml, TRACKED, so every
    machine's real configuration is versioned and readable from any other.
    The filename carries the machine name precisely so the machines cannot
    conflict with each other the way one shared filename would."""

    def _load(self, base, machine=None, local=None, env=None, machine_file="config.zerob.yaml"):
        def fake_load(path=config_module.CONFIG_PATH):
            if path == config_module.CONFIG_PATH:
                return base
            if path == config_module.CONFIG_LOCAL_PATH:
                return local or {}
            if path.name == machine_file:
                return machine or {}
            return {}

        with mock.patch("config.load_yaml_config", side_effect=fake_load):
            with mock.patch.dict(os.environ, env or {"INSTANCE": "zerob"}, clear=True):
                return config_module.load_config()

    def test_machine_file_overrides_the_shared_file(self):
        result = self._load({"backend": {"port": 8000, "gpu": "2,3"}}, machine={"backend": {"port": 8001}})

        self.assertEqual(result.backend.port, 8001)
        self.assertEqual(result.backend.gpu, "2,3", "untouched keys keep the shared value")

    def test_local_untracked_file_beats_the_machine_file(self):
        result = self._load({"backend": {"gpu": "2,3"}},
                            machine={"backend": {"gpu": "0"}}, local={"backend": {"gpu": "1"}})

        self.assertEqual(result.backend.gpu, "1")

    def test_env_still_beats_every_file(self):
        result = self._load({"backend": {"port": 8000}}, machine={"backend": {"port": 8001}},
                            env={"INSTANCE": "zerob", "PORT": "8002"})

        self.assertEqual(result.backend.port, 8002)

    def test_a_machine_with_no_file_of_its_own_gets_the_shared_config(self):
        result = self._load({"backend": {"port": 8000, "gpu": "2,3"}},
                            machine={"backend": {"port": 9999}}, env={"INSTANCE": "unknown-machine"})

        self.assertEqual(result.backend.port, 8000)
        self.assertEqual(result.backend.gpu, "2,3")

    def test_found_by_hostname_when_INSTANCE_is_not_set(self):
        """A machine invoked without INSTANCE (./backend status, cron) must
        still find its own file, or it would silently report a different GPU
        and model than it actually serves."""
        import socket

        result = self._load({"backend": {"port": 8000}}, machine={"backend": {"port": 8001}},
                            env={}, machine_file="config.{}.yaml".format(socket.gethostname()))

        self.assertEqual(result.backend.port, 8001)

    def test_candidate_filenames_include_the_short_hostname(self):
        names = [p.name for p in config_module.instance_config_candidates("a4000.ipa.test")]

        self.assertEqual(names, ["config.a4000.ipa.test.yaml", "config.a4000.yaml"])

    def test_candidate_filenames_reject_path_traversal(self):
        names = [p.name for p in config_module.instance_config_candidates("../../etc/passwd")]

        for name in names:
            self.assertNotIn("/", name)


class LocalConfigOverlayTests(unittest.TestCase):
    """config.local.yaml stays supported as the untracked, highest-priority
    layer for anything that must not be committed."""

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
