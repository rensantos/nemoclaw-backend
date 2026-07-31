import unittest
from unittest import mock

import torch

from config import BackendConfig, Config, ModelConfig
from engines.transformers_engine import TransformersEngine, scan_local_cache


def _make_config(quantization="none", model_id="test-model", gpu="0", revision=""):
    return Config(
        backend=BackendConfig(
            host="127.0.0.1",
            port=8000,
            gpu=gpu,
            engine="transformers",
            ollama_host="http://127.0.0.1:11434",
        ),
        model=ModelConfig(
            id=model_id,
            max_tokens_default=16,
            temperature_default=0.1,
            quantization=quantization,
            revision=revision,
        ),
    )


class TransformersEngineQuantizationTests(unittest.TestCase):
    """docs/quantization-design.md Section 6: verify the exact kwargs
    passed to from_pretrained() per quantization setting, without loading
    a real model or requiring bitsandbytes/a GPU."""

    def _load(self, quantization, gpu="0"):
        engine = TransformersEngine(_make_config(quantization=quantization, gpu=gpu))
        with mock.patch(
            "engines.transformers_engine.AutoTokenizer.from_pretrained"
        ), mock.patch(
            "engines.transformers_engine.AutoModelForCausalLM.from_pretrained"
        ) as mocked_model:
            mocked_model.return_value = mock.Mock()
            engine.load_model()
        return mocked_model

    def test_none_quantization_uses_auto_device_map(self):
        mocked_model = self._load("none")

        _, kwargs = mocked_model.call_args
        self.assertEqual(kwargs.get("torch_dtype"), torch.float16)
        self.assertNotIn("quantization_config", kwargs)
        self.assertEqual(kwargs.get("device_map"), "auto")

    def test_4bit_quantization_passes_nf4_config_not_torch_dtype(self):
        mocked_model = self._load("4bit")

        _, kwargs = mocked_model.call_args
        self.assertNotIn("torch_dtype", kwargs)
        quant_config = kwargs["quantization_config"]
        self.assertTrue(quant_config.load_in_4bit)
        self.assertEqual(quant_config.bnb_4bit_quant_type, "nf4")
        self.assertTrue(quant_config.bnb_4bit_use_double_quant)

    def test_8bit_quantization_passes_int8_config_not_torch_dtype(self):
        mocked_model = self._load("8bit")

        _, kwargs = mocked_model.call_args
        self.assertNotIn("torch_dtype", kwargs)
        quant_config = kwargs["quantization_config"]
        self.assertTrue(quant_config.load_in_8bit)

    def test_single_gpu_quantized_load_pins_device_map_not_auto(self):
        """Regression test: live on UBI, device_map="auto" mis-sized a
        4-bit load against the model's unquantized footprint and
        offloaded layers to CPU, which bitsandbytes then refused."""
        for quantization in ("4bit", "8bit"):
            with self.subTest(quantization=quantization):
                mocked_model = self._load(quantization, gpu="0")
                _, kwargs = mocked_model.call_args
                self.assertEqual(kwargs.get("device_map"), {"": 0})

    def test_multi_gpu_quantized_load_keeps_auto_device_map(self):
        mocked_model = self._load("4bit", gpu="0,1,2,3")

        _, kwargs = mocked_model.call_args
        self.assertEqual(kwargs.get("device_map"), "auto")

    def test_unset_revision_passes_none_to_from_pretrained(self):
        engine = TransformersEngine(_make_config(revision=""))
        with mock.patch(
            "engines.transformers_engine.AutoTokenizer.from_pretrained"
        ) as mocked_tokenizer, mock.patch(
            "engines.transformers_engine.AutoModelForCausalLM.from_pretrained"
        ) as mocked_model:
            mocked_model.return_value = mock.Mock()
            engine.load_model()

        self.assertIsNone(mocked_tokenizer.call_args.kwargs.get("revision"))
        self.assertIsNone(mocked_model.call_args.kwargs.get("revision"))

    def test_pinned_revision_is_passed_to_tokenizer_and_model(self):
        engine = TransformersEngine(_make_config(revision="abc123"))
        with mock.patch(
            "engines.transformers_engine.AutoTokenizer.from_pretrained"
        ) as mocked_tokenizer, mock.patch(
            "engines.transformers_engine.AutoModelForCausalLM.from_pretrained"
        ) as mocked_model:
            mocked_model.return_value = mock.Mock()
            engine.load_model()

        self.assertEqual(mocked_tokenizer.call_args.kwargs.get("revision"), "abc123")
        self.assertEqual(mocked_model.call_args.kwargs.get("revision"), "abc123")

    def test_load_model_is_a_no_op_once_already_loaded(self):
        engine = TransformersEngine(_make_config(quantization="none"))
        with mock.patch(
            "engines.transformers_engine.AutoTokenizer.from_pretrained"
        ), mock.patch(
            "engines.transformers_engine.AutoModelForCausalLM.from_pretrained"
        ) as mocked_model:
            mocked_model.return_value = mock.Mock()
            engine.load_model()
            engine.load_model()

        self.assertEqual(mocked_model.call_count, 1)


class ScanLocalCacheTests(unittest.TestCase):
    """Read-only local HF cache discovery, independent of config.yaml."""

    def test_returns_repo_id_keyed_dict_for_model_repos_only(self):
        model_repo = mock.Mock(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", repo_type="model")
        dataset_repo = mock.Mock(repo_id="some/dataset", repo_type="dataset")
        fake_cache_info = mock.Mock(repos=[model_repo, dataset_repo])

        with mock.patch("huggingface_hub.scan_cache_dir", return_value=fake_cache_info):
            result = scan_local_cache()

        self.assertEqual(set(result.keys()), {"TinyLlama/TinyLlama-1.1B-Chat-v1.0"})
        self.assertIs(result["TinyLlama/TinyLlama-1.1B-Chat-v1.0"], model_repo)

    def test_returns_empty_dict_when_no_cache_directory_exists(self):
        from huggingface_hub.errors import CacheNotFound

        with mock.patch(
            "huggingface_hub.scan_cache_dir",
            side_effect=CacheNotFound("no cache", cache_dir="/nonexistent"),
        ):
            result = scan_local_cache()

        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
