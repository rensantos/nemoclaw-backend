"""Tests for model downloading (docs/model-pull-design.md).

The disk gates are the point of these tests. This backend runs on a shared
machine whose volume is ~99% full with other researchers' data, so a
download that "mostly works" is not acceptable: it must refuse.
"""

import unittest
from unittest import mock

from engines.base import InferenceEngine
from services.inference import DISK_RESERVE_MIB, MINIMUM_FREE_MIB, InferenceService
from services.lifecycle import InsufficientDiskError, PullNotSupportedError
from services.resources import DiskInfo, HostResources

MIB = 1024 * 1024


class FakeEngine(InferenceEngine):
    supports_pull = True

    def __init__(self, total_bytes=None, error=None):
        self.model_id = "qwen3:30b"
        self.total_bytes = total_bytes
        self.error = error
        self.pulled = []
        self.aborted = False

    def pull_model(self, model_id, size_guard=None):
        self.pulled.append(model_id)
        if self.error:
            raise self.error
        if self.total_bytes is not None and size_guard is not None:
            try:
                size_guard(self.total_bytes)
            except Exception:
                self.aborted = True
                raise
        return self.total_bytes

    def model_storage_path(self):
        return "/home/d3894/ollama/models"

    def load_model(self, model_id=None):
        pass

    def unload_model(self):
        pass

    def health(self):
        return {}

    def list_models(self):
        return {}

    def chat(self, *args, **kwargs):
        return {}

    def generate_text(self, *args, **kwargs):
        return {}


class NoPullEngine(FakeEngine):
    supports_pull = False


class FakeModelManager:
    def __init__(self, error=None, already_present=False):
        self.error = error
        self.already_present = already_present
        self.registered = []

    def register_model(self, model_id):
        if self.error:
            raise self.error
        self.registered.append(model_id)
        return not self.already_present


def service_with(engine, free_mib, model_manager=None):
    service = InferenceService(engine, model_manager=model_manager)
    disk = (
        None
        if free_mib is None
        else DiskInfo(path="/home/d3894/ollama/models", total_mib=929315, free_mib=free_mib)
    )
    service.host_resources = lambda: HostResources(disk=disk, memory=None, gpus=[])
    return service


class PullCapabilityTests(unittest.TestCase):
    def test_engine_without_pull_support_is_refused(self):
        service = service_with(NoPullEngine(), free_mib=500000)

        with self.assertRaises(PullNotSupportedError):
            service.pull_model("anything")

    def test_a_refused_engine_is_never_asked_to_download(self):
        engine = NoPullEngine()
        service = service_with(engine, free_mib=500000)

        with self.assertRaises(PullNotSupportedError):
            service.pull_model("anything")

        self.assertEqual(engine.pulled, [])


class PreflightDiskTests(unittest.TestCase):
    def test_refuses_before_touching_the_disk_when_nearly_full(self):
        engine = FakeEngine(total_bytes=1 * MIB)
        service = service_with(engine, free_mib=MINIMUM_FREE_MIB - 1)

        with self.assertRaises(InsufficientDiskError) as ctx:
            service.pull_model("tiny:1b")

        self.assertEqual(engine.pulled, [])  # never started
        self.assertIn("Refusing", str(ctx.exception))

    def test_allows_when_above_the_floor(self):
        engine = FakeEngine(total_bytes=100 * MIB)
        service = service_with(engine, free_mib=50000)

        result = service.pull_model("small:1b")

        self.assertEqual(engine.pulled, ["small:1b"])
        self.assertEqual(result["size_mib"], 100)


class InFlightDiskTests(unittest.TestCase):
    """The size is not knowable before starting, so the real check happens
    against the first size the daemon reports."""

    def test_aborts_when_the_reported_size_will_not_fit(self):
        free_mib = 8144  # UBI's actual free space
        engine = FakeEngine(total_bytes=20000 * MIB)
        service = service_with(engine, free_mib=free_mib)

        with self.assertRaises(InsufficientDiskError) as ctx:
            service.pull_model("qwen3:32b")

        self.assertTrue(engine.aborted)
        self.assertEqual(ctx.exception.required_mib, 20000)
        self.assertEqual(ctx.exception.free_mib, free_mib)

    def test_reserve_is_enforced_not_just_raw_fit(self):
        """A model that technically fits but would leave the filesystem at
        ~100% is still refused: a full disk breaks the whole machine."""
        free_mib = 10000
        engine = FakeEngine(total_bytes=(free_mib - 100) * MIB)
        service = service_with(engine, free_mib=free_mib)

        with self.assertRaises(InsufficientDiskError):
            service.pull_model("just-barely:70b")

    def test_a_model_that_fits_with_reserve_is_allowed(self):
        free_mib = 10000
        engine = FakeEngine(total_bytes=(free_mib - DISK_RESERVE_MIB - 100) * MIB)
        service = service_with(engine, free_mib=free_mib)

        result = service.pull_model("fits:7b")

        self.assertFalse(engine.aborted)
        self.assertEqual(result["free_mib_before"], free_mib)

    def test_unknown_free_space_does_not_pretend_to_judge(self):
        """Refusing on unknown would block every download on a host that
        cannot report disk; claiming it fits would be a lie. The download
        proceeds without a size verdict, which is what actually happened
        before any of this existed."""
        engine = FakeEngine(total_bytes=99999 * MIB)
        service = service_with(engine, free_mib=None)

        result = service.pull_model("unknown-host:7b")

        self.assertFalse(engine.aborted)
        self.assertIsNone(result["free_mib_before"])


class CatalogRegistrationTests(unittest.TestCase):
    def test_a_downloaded_model_is_added_to_the_catalog(self):
        """Without this the model is invisible to /v1/models and rejected
        by switch - the whole reason pull goes through the backend."""
        manager = FakeModelManager()
        service = service_with(FakeEngine(total_bytes=MIB), 50000, model_manager=manager)

        result = service.pull_model("new:7b")

        self.assertEqual(manager.registered, ["new:7b"])
        self.assertTrue(result["registered"])

    def test_re_downloading_a_known_model_is_not_an_error(self):
        manager = FakeModelManager(already_present=True)
        service = service_with(FakeEngine(total_bytes=MIB), 50000, model_manager=manager)

        self.assertFalse(service.pull_model("known:7b")["registered"])

    def test_a_failed_catalog_write_does_not_misreport_the_download(self):
        """The bytes are on disk either way; raising here would tell the
        operator nothing was downloaded when something was."""
        manager = FakeModelManager(error=OSError("read-only config"))
        service = service_with(FakeEngine(total_bytes=MIB), 50000, model_manager=manager)

        result = service.pull_model("new:7b")

        self.assertFalse(result["registered"])
        self.assertEqual(result["model_id"], "new:7b")

    def test_pull_does_not_switch_the_served_model(self):
        """Downloading and serving are separate decisions; a pull that
        evicted the running model would surprise an active conversation."""
        engine = FakeEngine(total_bytes=MIB)
        service = service_with(engine, 50000, model_manager=FakeModelManager())
        before = service.loaded_model_id

        service.pull_model("other:7b")

        self.assertEqual(service.loaded_model_id, before)


class SizeReportingTests(unittest.TestCase):
    def test_size_is_unknown_when_the_daemon_never_reported_one(self):
        service = service_with(FakeEngine(total_bytes=None), 50000)

        self.assertIsNone(service.pull_model("quiet:7b")["size_mib"])


if __name__ == "__main__":
    unittest.main()
