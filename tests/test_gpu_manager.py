import types
import unittest
from unittest import mock

from services.gpu import GPUInfo, GPUManager


class FakeBackendConfig:
    gpu = "0"


class FakeModelConfig:
    id = "tiny"


class FakeConfig:
    backend = FakeBackendConfig()
    model = FakeModelConfig()


class GPUManagerTests(unittest.TestCase):
    def test_detect_gpus_parses_nvidia_smi_output(self):
        output = "0, RTX A4000, 16384, 512, 15872, 45, 12, 535.0\n"
        result = types.SimpleNamespace(stdout=output, stderr="")
        manager = GPUManager(FakeConfig())

        with mock.patch("services.gpu.subprocess.run", return_value=result):
            gpus = manager.detect_gpus()

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].index, "0")
        self.assertEqual(gpus[0].name, "RTX A4000")
        self.assertEqual(gpus[0].memory_total_mib, 16384)
        self.assertEqual(gpus[0].memory_used_mib, 512)
        self.assertEqual(gpus[0].memory_free_mib, 15872)
        self.assertEqual(gpus[0].temperature_c, 45)
        self.assertEqual(gpus[0].utilization_percent, 12)
        self.assertEqual(gpus[0].driver_version, "535.0")

    def test_detect_gpus_returns_empty_when_nvidia_smi_missing(self):
        manager = GPUManager(FakeConfig())

        with mock.patch("services.gpu.subprocess.run", side_effect=OSError):
            self.assertEqual(manager.detect_gpus(), [])

    def test_current_reports_configured_gpu_and_model(self):
        manager = GPUManager(FakeConfig())
        gpu = GPUInfo(
            index="0",
            name="RTX A4000",
            memory_total_mib=16384,
            memory_used_mib=512,
            memory_free_mib=15872,
            temperature_c=45,
            utilization_percent=12,
            driver_version="535.0",
        )

        with mock.patch.object(manager, "detect_gpus", return_value=[gpu]), \
                mock.patch.object(manager, "_torch_cuda_state", return_value=(True, "0")):
            current = manager.current()

        self.assertEqual(current.selected_cuda_device, "0")
        self.assertEqual(current.backend_gpu, "0")
        self.assertEqual(current.current_model, "tiny")
        self.assertEqual(current.available_memory_mib, 15872)
        self.assertTrue(current.cuda_available)
        self.assertEqual(current.driver_version, "535.0")


if __name__ == "__main__":
    unittest.main()
