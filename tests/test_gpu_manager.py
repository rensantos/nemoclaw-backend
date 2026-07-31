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

    def test_gpu_name_returns_selected_gpu(self):
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

        with mock.patch.object(manager, "detect_gpus", return_value=[gpu]):
            self.assertEqual(manager.gpu_name(), "RTX A4000")

    def test_gpu_name_returns_none_when_no_gpu_detected(self):
        manager = GPUManager(FakeConfig())

        with mock.patch.object(manager, "detect_gpus", return_value=[]):
            self.assertIsNone(manager.gpu_name())

    def test_busy_gpus_returns_empty_when_configured_gpu_idle(self):
        manager = GPUManager(FakeConfig())
        idle_gpu = GPUInfo(
            index="0", name="RTX A4000", memory_total_mib=16384,
            memory_used_mib=1, memory_free_mib=16383,
            temperature_c=40, utilization_percent=0, driver_version="470.86",
        )

        with mock.patch.object(manager, "detect_gpus", return_value=[idle_gpu]):
            self.assertEqual(manager.busy_gpus(), [])

    def test_busy_gpus_flags_configured_gpu_above_threshold(self):
        manager = GPUManager(FakeConfig())
        busy_gpu = GPUInfo(
            index="0", name="RTX A4000", memory_total_mib=16384,
            memory_used_mib=7000, memory_free_mib=9384,
            temperature_c=60, utilization_percent=30, driver_version="470.86",
        )

        with mock.patch.object(manager, "detect_gpus", return_value=[busy_gpu]):
            result = manager.busy_gpus()

        self.assertEqual(result, [busy_gpu])

    def test_busy_gpus_checks_each_index_in_multi_gpu_config(self):
        class MultiGPUBackendConfig:
            gpu = "2,3"

        class MultiGPUConfig:
            backend = MultiGPUBackendConfig()
            model = FakeModelConfig()

        manager = GPUManager(MultiGPUConfig())
        idle = GPUInfo(
            index="2", name="RTX A4000", memory_total_mib=16384,
            memory_used_mib=3, memory_free_mib=16381,
            temperature_c=40, utilization_percent=0, driver_version="470.86",
        )
        busy = GPUInfo(
            index="3", name="RTX A4000", memory_total_mib=16384,
            memory_used_mib=6600, memory_free_mib=9784,
            temperature_c=65, utilization_percent=40, driver_version="470.86",
        )

        with mock.patch.object(manager, "detect_gpus", return_value=[idle, busy]):
            result = manager.busy_gpus()

        self.assertEqual(result, [busy])

    def test_busy_gpus_respects_custom_threshold(self):
        manager = GPUManager(FakeConfig())
        gpu = GPUInfo(
            index="0", name="RTX A4000", memory_total_mib=16384,
            memory_used_mib=100, memory_free_mib=16284,
            temperature_c=40, utilization_percent=0, driver_version="470.86",
        )

        with mock.patch.object(manager, "detect_gpus", return_value=[gpu]):
            self.assertEqual(manager.busy_gpus(threshold_mib=50), [gpu])
            self.assertEqual(manager.busy_gpus(threshold_mib=500), [])

    def test_idle_alternative_gpus_excludes_configured_index(self):
        # FakeConfig.gpu == "0" - GPU 0 itself should never appear as its
        # own "alternative", even though it's idle here.
        manager = GPUManager(FakeConfig())
        configured_idle = GPUInfo(
            index="0", name="RTX A4000", memory_total_mib=16384,
            memory_used_mib=1, memory_free_mib=16383,
            temperature_c=40, utilization_percent=0, driver_version="470.86",
        )
        other_idle = GPUInfo(
            index="1", name="RTX A4000", memory_total_mib=16384,
            memory_used_mib=3, memory_free_mib=16381,
            temperature_c=40, utilization_percent=0, driver_version="470.86",
        )

        with mock.patch.object(manager, "detect_gpus", return_value=[configured_idle, other_idle]):
            result = manager.idle_alternative_gpus()

        self.assertEqual(result, [other_idle])

    def test_idle_alternative_gpus_excludes_busy_non_configured_gpus(self):
        manager = GPUManager(FakeConfig())
        other_busy = GPUInfo(
            index="1", name="RTX A4000", memory_total_mib=16384,
            memory_used_mib=6600, memory_free_mib=9784,
            temperature_c=65, utilization_percent=40, driver_version="470.86",
        )

        with mock.patch.object(manager, "detect_gpus", return_value=[other_busy]):
            self.assertEqual(manager.idle_alternative_gpus(), [])

    def test_idle_alternative_gpus_handles_multi_gpu_config(self):
        class MultiGPUBackendConfig:
            gpu = "2,3"

        class MultiGPUConfig:
            backend = MultiGPUBackendConfig()
            model = FakeModelConfig()

        manager = GPUManager(MultiGPUConfig())
        gpu0 = GPUInfo(
            index="0", name="RTX A4000", memory_total_mib=16384,
            memory_used_mib=1, memory_free_mib=16383,
            temperature_c=40, utilization_percent=0, driver_version="470.86",
        )
        gpu2_configured = GPUInfo(
            index="2", name="RTX A4000", memory_total_mib=16384,
            memory_used_mib=1, memory_free_mib=16383,
            temperature_c=40, utilization_percent=0, driver_version="470.86",
        )

        with mock.patch.object(manager, "detect_gpus", return_value=[gpu0, gpu2_configured]):
            result = manager.idle_alternative_gpus()

        self.assertEqual(result, [gpu0])


if __name__ == "__main__":
    unittest.main()
