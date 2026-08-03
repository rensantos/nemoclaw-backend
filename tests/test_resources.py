import unittest
from unittest import mock

from services.resources import DiskInfo, HostResourceService, MemoryInfo


class FakeGPU:
    def __init__(self, index, total, used, free):
        self.index = index
        self.name = "NVIDIA RTX A4000"
        self.memory_total_mib = total
        self.memory_used_mib = used
        self.memory_free_mib = free


class FakeGPUManager:
    def __init__(self, gpus=None, error=None):
        self._gpus = gpus or []
        self._error = error

    def detect_gpus(self):
        if self._error:
            raise self._error
        return self._gpus


class FakeEngine:
    def __init__(self, path=None, error=None):
        self._path = path
        self._error = error

    def model_storage_path(self):
        if self._error:
            raise self._error
        return self._path


class DiskTests(unittest.TestCase):
    def test_measures_the_engine_model_directory(self):
        """The number must answer "can I download this?", so it has to
        measure the filesystem that would receive the download."""
        service = HostResourceService(engine=FakeEngine("/home/d3894/ollama/models"))
        usage = mock.Mock(total=1000 * 1024 * 1024, free=250 * 1024 * 1024)

        with mock.patch("services.resources.shutil.disk_usage", return_value=usage) as probe:
            disk = service.disk()

        probe.assert_called_once_with("/home/d3894/ollama/models")
        self.assertEqual(disk.free_mib, 250)
        self.assertEqual(disk.used_percent, 75.0)

    def test_unreadable_disk_is_unknown_not_zero(self):
        """A caller about to download 20GB must not read a failed probe as
        "no space" or, worse, as "plenty"."""
        service = HostResourceService(engine=FakeEngine("/nope"))

        with mock.patch("services.resources.shutil.disk_usage", side_effect=OSError):
            self.assertIsNone(service.disk())

    def test_engine_without_a_storage_path_falls_back_to_home(self):
        service = HostResourceService(engine=FakeEngine(None))
        usage = mock.Mock(total=10 * 1024 * 1024, free=5 * 1024 * 1024)

        with mock.patch("services.resources.shutil.disk_usage", return_value=usage):
            with mock.patch("services.resources.os.path.expanduser", return_value="/home/me"):
                self.assertEqual(service.disk().path, "/home/me")

    def test_a_failing_engine_lookup_does_not_propagate(self):
        service = HostResourceService(engine=FakeEngine(error=RuntimeError("boom")))
        self.assertIsNone(service.model_storage_path())

    def test_used_percent_is_unknown_without_a_total(self):
        self.assertIsNone(DiskInfo(path="/", total_mib=None, free_mib=10).used_percent)


class MemoryTests(unittest.TestCase):
    MEMINFO = (
        "MemTotal:       32573440 kB\n"
        "MemFree:         1000000 kB\n"
        "MemAvailable:   28763136 kB\n"
        "Buffers:          100000 kB\n"
    )

    def test_reads_total_and_available(self):
        service = HostResourceService()
        with mock.patch("builtins.open", mock.mock_open(read_data=self.MEMINFO)):
            memory = service.memory()

        self.assertEqual(memory.total_mib, 31810)
        # MemAvailable, not MemFree: MemFree ignores reclaimable cache and
        # badly understates what a new workload could get.
        self.assertEqual(memory.available_mib, 28089)

    def test_unreadable_meminfo_is_unknown(self):
        service = HostResourceService()
        with mock.patch("builtins.open", side_effect=OSError):
            self.assertIsNone(service.memory())


class GPUAndSnapshotTests(unittest.TestCase):
    def test_sums_vram_across_gpus(self):
        service = HostResourceService(
            gpu_manager=FakeGPUManager([
                FakeGPU("0", 16117, 3, 16114),
                FakeGPU("1", 16117, 10000, 6117),
            ])
        )
        snapshot = HostResourceService.snapshot(service)

        self.assertEqual(snapshot.total_vram_mib, 32234)
        self.assertEqual(snapshot.free_vram_mib, 22231)

    def test_no_gpu_manager_reports_no_gpus_rather_than_failing(self):
        self.assertEqual(HostResourceService().gpus(), [])

    def test_gpu_detection_failure_is_contained(self):
        service = HostResourceService(gpu_manager=FakeGPUManager(error=OSError("nvidia-smi missing")))
        self.assertEqual(service.gpus(), [])

    def test_vram_totals_are_unknown_when_nothing_is_reported(self):
        service = HostResourceService(gpu_manager=FakeGPUManager([]))
        snapshot = HostResourceService.snapshot(service)

        self.assertIsNone(snapshot.total_vram_mib)
        self.assertIsNone(snapshot.free_vram_mib)

    def test_zero_free_vram_is_reported_as_zero_not_unknown(self):
        """A full GPU is a fact; it must not be confused with a missing
        reading."""
        service = HostResourceService(gpu_manager=FakeGPUManager([FakeGPU("0", 16117, 16117, 0)]))
        snapshot = HostResourceService.snapshot(service)

        self.assertEqual(snapshot.free_vram_mib, 0)


class MemoryInfoTests(unittest.TestCase):
    def test_fields_default_to_unknown(self):
        info = MemoryInfo(total_mib=None, available_mib=None)
        self.assertIsNone(info.total_mib)
        self.assertIsNone(info.available_mib)


if __name__ == "__main__":
    unittest.main()
