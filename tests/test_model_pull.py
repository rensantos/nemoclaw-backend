"""Tests for model downloading (docs/model-pull-design.md).

The disk gates are the point of these tests. This backend runs on a shared
machine whose volume is ~99% full with other researchers' data, so a
download that "mostly works" is not acceptable: it must refuse.
"""

import unittest
from unittest import mock

from engines.base import InferenceEngine
from engines.ollama_engine import OllamaEngine
from services.inference import DISK_RESERVE_MIB, MINIMUM_FREE_MIB, InferenceService
from services.lifecycle import InsufficientDiskError, PullNotSupportedError
from services.resources import DiskInfo, HostResources

MIB = 1024 * 1024


class FakeEngine(InferenceEngine):
    supports_pull = True

    def __init__(self, total_bytes=None, error=None, events=None):
        self.model_id = "qwen3:30b"
        self.total_bytes = total_bytes
        self.error = error
        self.events = events or []
        # Called after each event is forwarded, so a test can look at the
        # service mid-transfer - the only moment progress reporting is
        # for, since the call itself returns when it is already over.
        self.on_event = None
        self.pulled = []
        self.aborted = False

    def pull_model(self, model_id, size_guard=None, on_progress=None):
        self.pulled.append(model_id)
        for event in self.events:
            if on_progress is not None:
                on_progress(event)
            if self.on_event is not None:
                self.on_event()
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


class PullProgressTests(unittest.TestCase):
    """Progress has to be readable *during* the transfer.

    pull_model answers only when the download is over, so everything these
    tests assert is sampled mid-flight, through the same snapshot the
    status endpoint serves. A percentage that only appears at the end
    would be worth nothing: the whole complaint it answers is an hour of
    silence with no way to tell a running download from a hung one.
    """

    DIGEST = "sha256:6e9f90f02bb3"
    EVENTS = [
        {"status": "pulling manifest"},
        {"status": "pulling 6e9f90f02bb3", "digest": DIGEST, "total": 900 * MIB, "completed": 100 * MIB},
        {"status": "pulling 6e9f90f02bb3", "digest": DIGEST, "total": 900 * MIB, "completed": 450 * MIB},
    ]

    def _watched(self, engine, free_mib=50000):
        service = service_with(engine, free_mib=free_mib)
        snapshots = []
        engine.on_event = lambda: snapshots.append(service.pull_status())
        return service, snapshots

    def test_nothing_pulled_yet_reports_idle_rather_than_erroring(self):
        status = service_with(FakeEngine(), 50000).pull_status()

        self.assertFalse(status["active"])
        self.assertEqual(status["status"], "idle")
        self.assertIsNone(status["percent"])

    def test_bytes_and_percent_are_visible_mid_download(self):
        engine = FakeEngine(total_bytes=900 * MIB, events=self.EVENTS)
        service, snapshots = self._watched(engine)

        service.pull_model("deepseek-r1:14b")

        midway = snapshots[-1]
        self.assertTrue(midway["active"])
        self.assertEqual(midway["model_id"], "deepseek-r1:14b")
        self.assertEqual(midway["completed_bytes"], 450 * MIB)
        self.assertEqual(midway["total_bytes"], 900 * MIB)
        self.assertEqual(midway["percent"], 50.0)

    def test_a_phase_with_no_bytes_still_moves_the_status(self):
        """'pulling manifest' and 'verifying sha256 digest' carry no byte
        count; a caller shown only bytes would read them as a stall."""
        engine = FakeEngine(total_bytes=900 * MIB, events=self.EVENTS)
        service, snapshots = self._watched(engine)

        service.pull_model("deepseek-r1:14b")

        self.assertEqual(snapshots[0]["status"], "pulling manifest")
        self.assertIsNone(snapshots[0]["percent"])

    def test_cumulative_layer_counts_are_not_added_to_themselves(self):
        """Ollama re-reports each layer's running total, so summing what
        arrives would claim 550MiB downloaded when 450MiB has landed."""
        engine = FakeEngine(total_bytes=900 * MIB, events=self.EVENTS)
        service, snapshots = self._watched(engine)

        service.pull_model("deepseek-r1:14b")

        self.assertEqual(snapshots[-1]["completed_bytes"], 450 * MIB)

    def test_separate_layers_are_summed(self):
        engine = FakeEngine(
            total_bytes=100 * MIB,
            events=[
                {"status": "pulling a", "digest": "sha256:a", "total": 100 * MIB, "completed": 100 * MIB},
                {"status": "pulling b", "digest": "sha256:b", "total": 40 * MIB, "completed": 10 * MIB},
            ],
        )
        service, snapshots = self._watched(engine)

        service.pull_model("two-layer:7b")

        self.assertEqual(snapshots[-1]["completed_bytes"], 110 * MIB)
        self.assertEqual(snapshots[-1]["total_bytes"], 140 * MIB)
        self.assertEqual(snapshots[-1]["layers"], 2)

    def test_a_finished_download_reads_as_complete_and_inactive(self):
        """The last event before success is routinely a few KiB short of
        the total; leaving it there would report a finished download as
        stalled at 99%."""
        engine = FakeEngine(
            total_bytes=900 * MIB,
            events=[{"status": "pulling x", "digest": self.DIGEST, "total": 900 * MIB, "completed": 899 * MIB}],
        )
        service = service_with(engine, free_mib=50000)

        service.pull_model("deepseek-r1:14b")
        status = service.pull_status()

        self.assertFalse(status["active"])
        self.assertEqual(status["percent"], 100.0)
        self.assertEqual(status["status"], "success")
        self.assertIsNone(status["error"])
        self.assertIsNotNone(status["finished_at"])

    def test_a_failed_download_stops_being_active_and_keeps_the_reason(self):
        engine = FakeEngine(error=RuntimeError("manifest not found"), events=self.EVENTS)
        service = service_with(engine, free_mib=50000)

        with self.assertRaises(RuntimeError):
            service.pull_model("nope:1b")
        status = service.pull_status()

        self.assertFalse(status["active"])
        self.assertIn("manifest not found", status["error"])

    def test_an_in_flight_disk_refusal_is_recorded_as_a_failed_pull(self):
        engine = FakeEngine(total_bytes=20000 * MIB)
        service = service_with(engine, free_mib=8144)

        with self.assertRaises(InsufficientDiskError):
            service.pull_model("qwen3:32b")

        self.assertFalse(service.pull_status()["active"])
        self.assertIn("Refusing", service.pull_status()["error"])

    def test_a_preflight_refusal_is_not_published_as_a_download(self):
        """Nothing was transferred, so nothing should appear as one; the
        caller already has the 507 explaining why."""
        service = service_with(FakeEngine(total_bytes=MIB), free_mib=MINIMUM_FREE_MIB - 1)

        with self.assertRaises(InsufficientDiskError):
            service.pull_model("tiny:1b")

        self.assertEqual(service.pull_status()["status"], "idle")
        self.assertIsNone(service.pull_status()["model_id"])

    def test_a_second_pull_does_not_inherit_the_first_ones_bytes(self):
        engine = FakeEngine(total_bytes=900 * MIB, events=self.EVENTS)
        service, snapshots = self._watched(engine)
        service.pull_model("first:7b")

        service.pull_model("second:7b")

        first_event_of_second_pull = snapshots[3]
        self.assertEqual(first_event_of_second_pull["model_id"], "second:7b")
        self.assertEqual(first_event_of_second_pull["completed_bytes"], 0)


class EngineProgressTests(unittest.TestCase):
    """OllamaEngine's side of the same feature, exercised against the raw
    event stream rather than the service."""

    def _engine(self, events):
        engine = OllamaEngine.__new__(OllamaEngine)
        engine.base_url = "http://127.0.0.1:11434"
        engine.model_id = "qwen3:4b"
        engine._post_stream = lambda path, payload: iter(events)
        return engine

    def test_every_event_reaches_the_reporter(self):
        seen = []
        engine = self._engine(
            [{"status": "pulling manifest"}, {"status": "pulling x", "total": 10, "completed": 4}]
        )

        engine.pull_model("x:1b", on_progress=seen.append)

        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1]["completed"], 4)

    def test_a_reporter_that_raises_does_not_abort_the_transfer(self):
        """Nine gigabytes are not thrown away because a status line could
        not be written."""
        engine = self._engine([{"status": "pulling x", "total": 10, "completed": 4}] * 3)

        def broken(event):
            raise RuntimeError("telegram is down")

        with self.assertLogs("engines.ollama_engine", level="WARNING") as logs:
            total = engine.pull_model("x:1b", on_progress=broken)

        self.assertEqual(total, 10)
        self.assertEqual(len(logs.records), 1)  # dropped, not retried per event


if __name__ == "__main__":
    unittest.main()
