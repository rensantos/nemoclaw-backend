"""Start-time GPU safety policy, tested without a terminal.

These decisions used to live in cli.py, where they could only be
exercised through captured stdout. They are policy, so they belong to a
service and are asserted here as verdicts.
"""

import types
import unittest
from unittest import mock

from services.gpu import GPUInfo, GPUManager
from services.gpu_safety import (
    CONFIRM,
    PROCEED,
    REFUSE,
    GPUSafetyService,
    runtime_inspector_for,
)


def _gpu(index, used=1, util=0):
    return GPUInfo(
        index=index, name="RTX A4000", memory_total_mib=16117,
        memory_used_mib=used, memory_free_mib=16117 - used,
        temperature_c=45, utilization_percent=util, driver_version="470.86",
    )


def _config(engine="ollama", gpu="2,3"):
    return types.SimpleNamespace(
        backend=types.SimpleNamespace(engine=engine, gpu=gpu),
        model=types.SimpleNamespace(id="qwen3:30b"),
    )


class FakeRuntime:
    def __init__(self, daemon_pids=(23825,), runtime_pids=(17181, 23825)):
        self._daemon_pids = list(daemon_pids)
        self._runtime_pids = list(runtime_pids)

    def daemon_pids(self):
        return self._daemon_pids

    def runtime_pids(self):
        return self._runtime_pids


def _service(gpus, processes=(), runtime=None, config=None):
    manager = GPUManager(config or _config())
    manager.detect_gpus = lambda: list(gpus)
    manager.gpu_processes = lambda: list(processes)
    return GPUSafetyService(config or _config(), manager, runtime)


def _proc(index, pid, name="/home/d3894/ollama/bin/ollama"):
    from services.gpu import GPUProcess

    return GPUProcess(gpu_index=index, pid=pid, process_name=name, memory_mib=6000)


class StartDecisionTests(unittest.TestCase):
    def test_proceeds_when_everything_is_idle(self):
        service = _service([_gpu("2"), _gpu("3")], runtime=FakeRuntime(daemon_pids=[]))

        decision = service.evaluate_start()

        self.assertEqual(decision.outcome, PROCEED)
        self.assertTrue(decision.allowed)

    def test_our_own_model_never_blocks_a_start(self):
        """The regression that made the backend refuse because of the
        model it was itself serving."""
        gpus = [_gpu("2", used=10645), _gpu("3", used=10181)]
        processes = [_proc("2", 17181), _proc("3", 17181)]
        service = _service(
            gpus, processes, runtime=FakeRuntime(daemon_pids=[], runtime_pids=[17181])
        )

        decision = service.evaluate_start()

        self.assertEqual(decision.outcome, PROCEED)
        self.assertEqual(decision.configured_busy, [])

    def test_refuses_when_the_configured_gpu_is_busy_and_another_is_idle(self):
        gpus = [_gpu("0"), _gpu("1"), _gpu("2", used=7000), _gpu("3")]
        processes = [_proc("2", 999, name="python")]
        service = _service(gpus, processes, runtime=FakeRuntime(daemon_pids=[]))

        decision = service.evaluate_start()

        self.assertEqual(decision.outcome, REFUSE)
        self.assertEqual([g.index for g in decision.configured_busy], ["2"])
        self.assertIn("0", [g.index for g in decision.alternatives])

    def test_asks_when_the_configured_gpu_is_busy_and_nothing_is_idle(self):
        gpus = [_gpu("2", used=7000), _gpu("3", used=6500)]
        processes = [_proc("2", 999, name="python"), _proc("3", 998, name="python")]
        service = _service(gpus, processes, runtime=FakeRuntime(daemon_pids=[]))

        decision = service.evaluate_start()

        self.assertEqual(decision.outcome, CONFIRM)

    def test_force_overrides_a_busy_configured_gpu(self):
        gpus = [_gpu("0"), _gpu("2", used=7000)]
        processes = [_proc("2", 999, name="python")]
        service = _service(gpus, processes, runtime=FakeRuntime(daemon_pids=[]))

        decision = service.evaluate_start(force=True)

        self.assertEqual(decision.outcome, PROCEED)
        self.assertTrue(decision.forced)
        self.assertTrue(decision.configured_busy)


class RuntimeExposureTests(unittest.TestCase):
    """backend.gpu constrains this process only; an external model runtime
    places models using its own visible devices."""

    def _service_with_visibility(self, gpus, processes, visible):
        manager = GPUManager(_config())
        manager.detect_gpus = lambda: list(gpus)
        manager.gpu_processes = lambda: list(processes)
        manager.visible_gpu_indexes_for_process = lambda pid: list(visible)
        return GPUSafetyService(_config(), manager, FakeRuntime())

    def test_warns_but_proceeds_while_a_safe_placement_remains(self):
        """Refusing whenever a busy GPU is merely reachable would block
        almost every start on a shared box."""
        gpus = [_gpu("0", used=6658, util=80), _gpu("1"), _gpu("2"), _gpu("3")]
        processes = [_proc("0", 999, name="python")]
        service = self._service_with_visibility(
            gpus, processes, visible=["0", "1", "2", "3"]
        )

        decision = service.evaluate_start()

        self.assertEqual(decision.outcome, PROCEED)
        self.assertEqual(len(decision.exposures), 1)
        self.assertEqual([g.index for g in decision.exposures[0].unsafe], ["0"])
        self.assertTrue(decision.exposures[0].has_safe_placement)

    def test_refuses_when_every_reachable_gpu_is_busy(self):
        gpus = [_gpu("0", used=6658, util=80), _gpu("1", used=6658, util=87)]
        processes = [_proc("0", 999, name="python"), _proc("1", 998, name="python")]
        service = self._service_with_visibility(gpus, processes, visible=["0", "1"])

        decision = service.evaluate_start()

        self.assertEqual(decision.outcome, REFUSE)
        self.assertIsNotNone(decision.blocking_exposure)

    def test_force_overrides_even_with_no_safe_placement(self):
        gpus = [_gpu("0", used=6658, util=80)]
        processes = [_proc("0", 999, name="python")]
        service = self._service_with_visibility(gpus, processes, visible=["0"])

        decision = service.evaluate_start(force=True)

        self.assertEqual(decision.outcome, PROCEED)
        self.assertTrue(decision.forced)

    def test_unreadable_visibility_is_reported_not_guessed(self):
        manager = GPUManager(_config())
        manager.detect_gpus = lambda: [_gpu("0")]
        manager.gpu_processes = lambda: []
        manager.unsafe_gpus_for_process = lambda pid, own_pids=None: None
        service = GPUSafetyService(_config(), manager, FakeRuntime())

        decision = service.evaluate_start()

        self.assertEqual(decision.outcome, PROCEED)
        self.assertFalse(decision.exposures[0].verified)

    def test_no_runtime_means_no_exposures(self):
        service = _service([_gpu("0")], runtime=None)

        decision = service.evaluate_start()

        self.assertEqual(decision.exposures, [])


class RuntimeInspectorSelectionTests(unittest.TestCase):
    def test_ollama_engine_gets_an_inspector(self):
        self.assertIsNotNone(runtime_inspector_for(_config(engine="ollama")))

    def test_in_process_engines_have_nothing_separate_to_inspect(self):
        self.assertIsNone(runtime_inspector_for(_config(engine="transformers")))

    def test_inspector_delegates_to_the_engine_module(self):
        inspector = runtime_inspector_for(_config(engine="ollama"))
        with mock.patch("engines.ollama_engine.find_daemon_pids", return_value=[1]), \
                mock.patch("engines.ollama_engine.find_runtime_pids", return_value=[1, 2]):
            self.assertEqual(inspector.daemon_pids(), [1])
            self.assertEqual(inspector.runtime_pids(), [1, 2])


if __name__ == "__main__":
    unittest.main()
