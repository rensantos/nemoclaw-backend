import importlib
import dataclasses
import io
import json
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from services.gpu import CurrentGPUInfo, GPUInfo


def _install_typer_stub():
    if "typer" in sys.modules:
        return

    typer_stub = types.ModuleType("typer")

    class FakeTyper:
        def __init__(self, *args, **kwargs):
            pass

        def add_typer(self, *args, **kwargs):
            pass

        def command(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def callback(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    class FakeContext:
        invoked_subcommand = None

        def get_help(self):
            return "help"

    class FakeExit(Exception):
        def __init__(self, code=0):
            super().__init__(code)
            self.code = code

    def option(default, *args, **kwargs):
        return default

    def echo(message=""):
        print(message)

    typer_stub.Typer = FakeTyper
    typer_stub.Context = FakeContext
    typer_stub.Exit = FakeExit
    def confirm(*args, **kwargs):
        return False

    typer_stub.Option = option
    typer_stub.echo = echo
    typer_stub.confirm = confirm
    sys.modules["typer"] = typer_stub


_install_typer_stub()
cli = importlib.import_module("cli")


class CliHelperTests(unittest.TestCase):
    def test_server_command_uses_configured_host_and_port(self):
        command = cli._server_command()

        self.assertIn("-m", command)
        self.assertIn("uvicorn", command)
        self.assertIn("server:app", command)
        self.assertIn(str(cli.config.backend.host), command)
        self.assertIn(str(cli.config.backend.port), command)

    def test_health_status_reads_json_status(self):
        with mock.patch.object(cli, "_health_text", return_value='{"status": "ok"}'):
            self.assertEqual(cli._health_status(), "ok")

    def test_health_status_reports_unavailable(self):
        message = "unavailable (connection refused)"
        with mock.patch.object(cli, "_health_text", return_value=message):
            self.assertEqual(cli._health_status(), message)

    def test_lifecycle_state_reads_json_field(self):
        with mock.patch.object(
            cli, "_health_text", return_value='{"status": "ok", "lifecycle_state": "ready"}'
        ):
            self.assertEqual(cli._lifecycle_state(), "ready")

    def test_lifecycle_state_reports_unknown_when_unavailable(self):
        message = "unavailable (connection refused)"
        with mock.patch.object(cli, "_health_text", return_value=message):
            self.assertEqual(cli._lifecycle_state(), "unknown")

    def test_status_displays_lifecycle_state(self):
        state = cli.BackendState(
            pid=None,
            pid_running=False,
            pid_matches_backend=False,
            health="ok",
            health_ok=True,
            port_open=True,
            matching_processes=[],
            lifecycle_state="ready",
        )
        current_gpu = CurrentGPUInfo(
            selected_cuda_device="0",
            backend_gpu="0",
            current_model="tiny",
            available_memory_mib=1024,
            cuda_available=True,
            torch_current_device="0",
            driver_version="535.0",
        )

        with mock.patch.object(cli, "_backend_state", return_value=state), \
                mock.patch.object(cli.gpu_manager, "current", return_value=current_gpu), \
                mock.patch.object(cli.gpu_manager, "detect_gpus", return_value=[]):
            output = io.StringIO()
            with redirect_stdout(output):
                cli.status()

        text = output.getvalue()
        self.assertIn("Lifecycle: ready", text)

    def test_status_uses_gpu_manager_for_gpu_info(self):
        state = cli.BackendState(
            pid=None,
            pid_running=False,
            pid_matches_backend=False,
            health="unavailable",
            health_ok=False,
            port_open=False,
            matching_processes=[],
        )
        # cli.status() filters gpu_manager.detect_gpus() by the real,
        # module-global config.backend.gpu (not by current_gpu.backend_gpu
        # below) - match whatever that's actually configured to, rather
        # than hardcoding an index, so this test doesn't silently depend
        # on config/config.yaml's current value.
        configured_gpu_index = cli.config.backend.gpu
        current_gpu = CurrentGPUInfo(
            selected_cuda_device=configured_gpu_index,
            backend_gpu=configured_gpu_index,
            current_model="tiny",
            available_memory_mib=1024,
            cuda_available=True,
            torch_current_device=configured_gpu_index,
            driver_version="535.0",
        )
        detected_gpu = GPUInfo(
            index=configured_gpu_index,
            name="RTX A4000",
            memory_total_mib=16384,
            memory_used_mib=512,
            memory_free_mib=15872,
            temperature_c=45,
            utilization_percent=12,
            driver_version="535.0",
        )

        with mock.patch.object(cli, "_backend_state", return_value=state), \
                mock.patch.object(cli.gpu_manager, "current", return_value=current_gpu), \
                mock.patch.object(cli.gpu_manager, "detect_gpus", return_value=[detected_gpu]):
            output = io.StringIO()
            with redirect_stdout(output):
                cli.status()

        text = output.getvalue()
        self.assertIn("VRAM: 1024 MiB", text)
        self.assertIn("Temperature: 45 C", text)

    def test_read_pid_returns_none_for_invalid_pid_file(self):
        with TemporaryDirectory() as tmp_dir:
            pid_file = Path(tmp_dir) / "backend.pid"
            pid_file.write_text("not-a-pid", encoding="utf-8")

            with mock.patch.object(cli, "PID_FILE", pid_file):
                self.assertIsNone(cli._read_pid())

    def test_pid_matches_backend_for_uvicorn_server(self):
        command = "/usr/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000"

        with mock.patch.object(cli, "_pid_command", return_value=command):
            self.assertTrue(cli._pid_matches_backend(123))

    def test_pid_does_not_match_unrelated_python_process(self):
        command = "/usr/bin/python unrelated.py"

        with mock.patch.object(cli, "_pid_command", return_value=command):
            self.assertFalse(cli._pid_matches_backend(123))

    def test_backend_state_running_when_health_ok_without_pid(self):
        with mock.patch.object(cli, "_read_pid", return_value=None), \
                mock.patch.object(cli, "_health_result", return_value=("ok", True)), \
                mock.patch.object(cli, "_port_is_open", return_value=False), \
                mock.patch.object(cli, "_matching_backend_processes", return_value=[]):
            state = cli._backend_state()

        self.assertTrue(state.running)
        self.assertFalse(state.managed_by_cli)
        self.assertEqual(state.health, "ok")

    def test_status_reports_unmanaged_running_backend(self):
        state = cli.BackendState(
            pid=None,
            pid_running=False,
            pid_matches_backend=False,
            health="ok",
            health_ok=True,
            port_open=True,
            matching_processes=[],
        )

        current_gpu = CurrentGPUInfo(
            selected_cuda_device="0",
            backend_gpu="0",
            current_model="tiny",
            available_memory_mib=None,
            cuda_available=False,
            torch_current_device="unavailable",
            driver_version="unavailable",
        )

        with mock.patch.object(cli, "_backend_state", return_value=state), \
                mock.patch.object(cli.gpu_manager, "current", return_value=current_gpu), \
                mock.patch.object(cli.gpu_manager, "detect_gpus", return_value=[]):
            output = io.StringIO()
            with redirect_stdout(output):
                cli.status()

        text = output.getvalue()
        self.assertIn("Running: yes", text)
        self.assertIn("Managed by CLI: no", text)
        self.assertIn("Health: ok", text)

    def test_stop_refuses_to_kill_unmanaged_backend(self):
        state = cli.BackendState(
            pid=None,
            pid_running=False,
            pid_matches_backend=False,
            health="ok",
            health_ok=True,
            port_open=True,
            matching_processes=["123 uvicorn server:app --port 8000"],
        )

        with mock.patch.object(cli, "_backend_state", return_value=state), \
                mock.patch.object(cli, "_terminate_pid") as terminate:
            output = io.StringIO()
            with redirect_stdout(output):
                with self.assertRaises(cli.typer.Exit):
                    cli.stop()

        terminate.assert_not_called()

    def test_model_list_marks_current_model(self):
        models = [
            {"id": "tiny", "name": "Tiny", "engine": "transformers"},
            {"id": "other", "name": "Other", "engine": "transformers"},
        ]

        with mock.patch.object(cli.model_manager, "selected_model_id", return_value="tiny"), \
                mock.patch.object(cli.model_manager, "list_models", return_value=models):
            output = io.StringIO()
            with redirect_stdout(output):
                cli.model_list()

        text = output.getvalue()
        self.assertIn("Configured models", text)
        self.assertIn("Model: tiny (current)", text)
        self.assertIn("Model: other", text)

    def test_model_current_shows_selected_model(self):
        model = {"id": "tiny", "name": "Tiny", "path": "Tiny/Tiny", "engine": "transformers"}

        with mock.patch.object(cli.model_manager, "selected_model_id", return_value="tiny"), \
                mock.patch.object(cli.model_manager, "current_model", return_value=model):
            output = io.StringIO()
            with redirect_stdout(output):
                cli.model_current()

        text = output.getvalue()
        self.assertIn("Selected/default model", text)
        self.assertIn("Model: tiny (current)", text)
        self.assertIn("Loaded model: determined by the running backend process", text)

    def test_model_use_rejects_invalid_model_id(self):
        with mock.patch.object(cli.model_manager, "validate_model", side_effect=ValueError):
            output = io.StringIO()
            with redirect_stdout(output):
                with self.assertRaises(cli.typer.Exit):
                    cli.model_use("missing")

        self.assertIn("Model is not configured: missing", output.getvalue())

    def test_model_use_updates_config_and_warns_when_running(self):
        state = cli.BackendState(
            pid=None,
            pid_running=False,
            pid_matches_backend=False,
            health="ok",
            health_ok=True,
            port_open=True,
            matching_processes=[],
        )

        with mock.patch.object(cli.model_manager, "validate_model") as validate_model, \
                mock.patch.object(cli.model_manager, "selected_model_id", return_value="tiny"), \
                mock.patch.object(cli.model_manager, "select_model") as select_model, \
                mock.patch.object(cli, "_backend_state", return_value=state):
            output = io.StringIO()
            with redirect_stdout(output):
                cli.model_use("other")

        validate_model.assert_called_once_with("other")
        select_model.assert_called_once_with("other")
        text = output.getvalue()
        self.assertIn("Selected model updated: other", text)
        self.assertIn("Restart required", text)

    def test_model_info_rejects_invalid_model_id(self):
        with mock.patch.object(cli.model_manager, "selected_model_id", return_value="tiny"), \
                mock.patch.object(cli.model_manager, "model_info", side_effect=ValueError):
            output = io.StringIO()
            with redirect_stdout(output):
                with self.assertRaises(cli.typer.Exit):
                    cli.model_info("missing")

        self.assertIn("Model is not configured: missing", output.getvalue())

    def _lifecycle_body(self, **overrides):
        body = {
            "status": "ok",
            "lifecycle_state": "ready",
            "loaded_model": "other",
            "previous_model": "tiny",
            "elapsed_seconds": 0.4,
            "persisted": False,
        }
        body.update(overrides)
        return json.dumps(body)

    def test_model_load_reports_success_and_exits_zero(self):
        with mock.patch.object(
            cli, "_post_json", return_value=(200, self._lifecycle_body(loaded_model="tiny", previous_model=None))
        ) as post_json:
            output = io.StringIO()
            with redirect_stdout(output):
                cli.model_load("tiny", timeout=120, persist=False, as_json=False)

        post_json.assert_called_once_with(
            cli._admin_url("/admin/model/load"),
            {"model_id": "tiny", "persist": False},
            timeout=120,
        )
        printed = output.getvalue()
        self.assertIn("Requested model: tiny", printed)
        self.assertIn("Loaded model: tiny", printed)
        self.assertIn("Lifecycle: ready", printed)

    def test_model_unload_reports_success(self):
        with mock.patch.object(
            cli,
            "_post_json",
            return_value=(200, self._lifecycle_body(lifecycle_state="unloaded", loaded_model=None)),
        ) as post_json:
            output = io.StringIO()
            with redirect_stdout(output):
                cli.model_unload(timeout=30, as_json=False)

        post_json.assert_called_once_with(
            cli._admin_url("/admin/model/unload"), timeout=30
        )
        printed = output.getvalue()
        self.assertIn("Loaded model: none", printed)
        self.assertIn("Lifecycle: unloaded", printed)

    def test_model_switch_passes_persist_and_reports_it(self):
        with mock.patch.object(
            cli, "_post_json", return_value=(200, self._lifecycle_body(persisted=True))
        ) as post_json:
            output = io.StringIO()
            with redirect_stdout(output):
                cli.model_switch("other", timeout=120, persist=True, as_json=False)

        post_json.assert_called_once_with(
            cli._admin_url("/admin/model/switch"),
            {"model_id": "other", "persist": True},
            timeout=120,
        )
        printed = output.getvalue()
        self.assertIn("Previous model: tiny", printed)
        self.assertIn("Loaded model: other", printed)
        self.assertIn("Persisted to config.yaml: yes", printed)

    def test_model_switch_reports_failure_and_exits_nonzero(self):
        body = json.dumps({
            "error": "model_not_configured",
            "detail": "Model is not configured: bogus",
            "lifecycle_state": "ready",
        })

        with mock.patch.object(cli, "_post_json", return_value=(404, body)):
            output = io.StringIO()
            with redirect_stdout(output):
                with self.assertRaises(cli.typer.Exit) as ctx:
                    cli.model_switch("bogus", timeout=120, persist=False, as_json=False)

        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("Model is not configured: bogus", output.getvalue())

    def test_model_switch_json_output_prints_raw_body(self):
        with mock.patch.object(
            cli, "_post_json", return_value=(200, self._lifecycle_body())
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                with self.assertRaises(cli.typer.Exit) as ctx:
                    cli.model_switch("other", timeout=120, persist=False, as_json=True)

        self.assertEqual(ctx.exception.code, 0)
        self.assertEqual(json.loads(output.getvalue())["loaded_model"], "other")

    def test_model_load_reports_unreachable_backend_and_exits_nonzero(self):
        with mock.patch.object(
            cli, "_post_json", return_value=(None, "unavailable (connection refused)")
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                with self.assertRaises(cli.typer.Exit) as ctx:
                    cli.model_load("tiny")

        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("Backend is not reachable", output.getvalue())

    def test_gpu_list_uses_gpu_manager(self):
        detected_gpu = GPUInfo(
            index="0",
            name="RTX A4000",
            memory_total_mib=16384,
            memory_used_mib=512,
            memory_free_mib=15872,
            temperature_c=45,
            utilization_percent=12,
            driver_version="535.0",
        )

        with mock.patch.object(cli.gpu_manager, "detect_gpus", return_value=[detected_gpu]):
            output = io.StringIO()
            with redirect_stdout(output):
                cli.gpu_list()

        text = output.getvalue()
        self.assertIn("Detected GPUs", text)
        self.assertIn("GPU 0", text)
        self.assertIn("RTX A4000", text)
        self.assertIn("Utilization: 12%", text)

    def test_gpu_current_uses_gpu_manager(self):
        current_gpu = CurrentGPUInfo(
            selected_cuda_device="0",
            backend_gpu="0",
            current_model="tiny",
            available_memory_mib=1024,
            cuda_available=True,
            torch_current_device="0",
            driver_version="535.0",
        )

        with mock.patch.object(cli.gpu_manager, "current", return_value=current_gpu):
            output = io.StringIO()
            with redirect_stdout(output):
                cli.gpu_current()

        text = output.getvalue()
        self.assertIn("Selected CUDA device: 0", text)
        self.assertIn("Current model: tiny", text)
        self.assertIn("Available memory: 1024 MiB", text)

    def test_benchmark_latency_delegates_to_service(self):
        result = {
            "benchmark": "latency",
            "model": "tiny",
            "gpu": "0",
            "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
            "runs": 2,
            "concurrency": 1,
            "average_seconds": 1.5,
            "min_seconds": 1.0,
            "max_seconds": 2.0,
            "results": [],
        }

        with mock.patch.object(cli.benchmark_service, "latency", return_value=result) as latency:
            output = io.StringIO()
            with redirect_stdout(output):
                cli.benchmark_latency(
                    prompt="hello",
                    max_tokens=8,
                    runs=2,
                    concurrency=1,
                    json_output=False,
                )

        latency.assert_called_once_with("hello", 8, 2, 1)
        text = output.getvalue()
        self.assertIn("Benchmark: latency", text)
        self.assertIn("Average latency: 1.500 s", text)

    def test_benchmark_throughput_can_print_json(self):
        result = {
            "benchmark": "throughput",
            "model": "tiny",
            "gpu": "0",
            "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
            "runs": 1,
            "concurrency": 1,
            "elapsed_seconds": 2.0,
            "requests_per_second": 0.5,
            "tokens_per_second": 10.0,
            "results": [],
        }

        with mock.patch.object(cli.benchmark_service, "throughput", return_value=result):
            output = io.StringIO()
            with redirect_stdout(output):
                cli.benchmark_throughput(
                    prompt="hello",
                    max_tokens=8,
                    runs=1,
                    concurrency=1,
                    json_output=True,
                )

        decoded = json.loads(output.getvalue())
        self.assertEqual(decoded["benchmark"], "throughput")
        self.assertEqual(decoded["tokens_per_second"], 10.0)


class CheckGpuBeforeStartTests(unittest.TestCase):
    def _gpu(self, index, used_mib):
        return GPUInfo(
            index=index, name="RTX A4000", memory_total_mib=16384,
            memory_used_mib=used_mib, memory_free_mib=16384 - used_mib,
            temperature_c=50, utilization_percent=10, driver_version="470.86",
        )

    def test_proceeds_when_no_gpu_busy(self):
        with mock.patch.object(cli.gpu_manager, "busy_gpus", return_value=[]):
            self.assertTrue(cli._check_gpu_before_start())

    def test_force_skips_check_entirely(self):
        busy = [self._gpu("2", 7000)]
        with mock.patch.object(cli.gpu_manager, "busy_gpus", return_value=busy), \
                mock.patch.object(cli.gpu_manager, "idle_alternative_gpus") as alt:
            self.assertTrue(cli._check_gpu_before_start(force=True))
        alt.assert_not_called()

    def test_refuses_when_idle_alternative_exists(self):
        busy = [self._gpu("2", 7000)]
        idle = [self._gpu("0", 1)]
        with mock.patch.object(cli.gpu_manager, "busy_gpus", return_value=busy), \
                mock.patch.object(cli.gpu_manager, "idle_alternative_gpus", return_value=idle), \
                mock.patch.object(cli.typer, "confirm") as confirm:
            self.assertFalse(cli._check_gpu_before_start())
        confirm.assert_not_called()

    def test_asks_permission_when_no_idle_alternative(self):
        busy = [self._gpu("2", 7000), self._gpu("3", 6500)]
        with mock.patch.object(cli.gpu_manager, "busy_gpus", return_value=busy), \
                mock.patch.object(cli.gpu_manager, "idle_alternative_gpus", return_value=[]), \
                mock.patch.object(cli.typer, "confirm", return_value=True) as confirm:
            self.assertTrue(cli._check_gpu_before_start())
        confirm.assert_called_once()

    def test_declining_permission_refuses_to_start(self):
        busy = [self._gpu("2", 7000)]
        with mock.patch.object(cli.gpu_manager, "busy_gpus", return_value=busy), \
                mock.patch.object(cli.gpu_manager, "idle_alternative_gpus", return_value=[]), \
                mock.patch.object(cli.typer, "confirm", return_value=False):
            self.assertFalse(cli._check_gpu_before_start())

    def test_force_warns_loudly_instead_of_bypassing_silently(self):
        busy = [self._gpu("2", 7000)]
        output = io.StringIO()
        with mock.patch.object(cli.gpu_manager, "busy_gpus", return_value=busy), \
                redirect_stdout(output):
            self.assertTrue(cli._check_gpu_before_start(force=True))

        printed = output.getvalue()
        self.assertIn("GPU 2", printed)
        self.assertIn("ON TOP", printed)
        self.assertIn("Another user's job may be disrupted", printed)


def _config_with_engine(engine):
    """Config and BackendConfig are both frozen dataclasses, so rebuild
    them and patch the module-level name rather than a field."""
    return dataclasses.replace(
        cli.config, backend=dataclasses.replace(cli.config.backend, engine=engine)
    )


class ExternalRuntimeGPUCheckTests(unittest.TestCase):
    """backend.gpu constrains only the backend process. With engine:
    ollama the daemon places models using its own visible devices, so it
    needs its own check."""

    def _gpu(self, index, used=1, util=0):
        return GPUInfo(
            index=index, name="RTX A4000", memory_total_mib=16384,
            memory_used_mib=used, memory_free_mib=16384 - used,
            temperature_c=50, utilization_percent=util, driver_version="470.86",
        )

    def _run(self, unsafe, free, force=False, engine="ollama", pids=(4321,)):
        availability = cli.GPUAvailability(in_use=list(unsafe), free=list(free))
        output = io.StringIO()
        with mock.patch.object(cli, "config", _config_with_engine(engine)), \
                mock.patch(
                    "engines.ollama_engine.find_daemon_pids", return_value=list(pids)
                ), \
                mock.patch.object(
                    cli.gpu_manager, "unsafe_gpus_for_process", return_value=list(unsafe)
                ), \
                mock.patch.object(
                    cli.gpu_manager, "availability", return_value=availability
                ), \
                redirect_stdout(output):
            result = cli._check_external_runtime_gpus(force=force)
        return result, output.getvalue()

    def test_refuses_when_daemon_can_reach_a_busy_gpu(self):
        unsafe = [self._gpu("0", 6658, 80)]
        free = [self._gpu("2"), self._gpu("3")]

        allowed, printed = self._run(unsafe, free)

        self.assertFalse(allowed)
        self.assertIn("GPU 0", printed)
        self.assertIn("does NOT constrain the daemon", printed)
        # Must hand the operator the exact fix, pinned to the free GPUs.
        self.assertIn("CUDA_VISIBLE_DEVICES=2,3 ollama serve", printed)

    def test_allows_when_daemon_is_pinned_away_from_busy_gpus(self):
        allowed, printed = self._run(unsafe=[], free=[self._gpu("2")])

        self.assertTrue(allowed)
        self.assertEqual(printed, "")

    def test_force_warns_but_proceeds(self):
        unsafe = [self._gpu("1", 6658, 87)]
        allowed, printed = self._run(unsafe, free=[self._gpu("3")], force=True)

        self.assertTrue(allowed)
        self.assertIn("--force set", printed)
        self.assertIn("another user's GPU", printed)

    def test_reports_when_no_free_gpu_exists_to_suggest(self):
        unsafe = [self._gpu("0", 6658, 80)]
        allowed, printed = self._run(unsafe, free=[])

        self.assertFalse(allowed)
        self.assertIn("No free GPU is available", printed)

    def test_skipped_entirely_for_non_ollama_engines(self):
        allowed, printed = self._run(
            unsafe=[self._gpu("0", 9000)], free=[], engine="transformers"
        )

        self.assertTrue(allowed)
        self.assertEqual(printed, "")

    def test_no_daemon_found_is_not_treated_as_a_failure(self):
        allowed, _ = self._run(unsafe=[], free=[], pids=())
        self.assertTrue(allowed)

    def test_unreadable_daemon_environ_reports_unverified_and_proceeds(self):
        output = io.StringIO()
        with mock.patch.object(cli, "config", _config_with_engine("ollama")), \
                mock.patch(
                    "engines.ollama_engine.find_daemon_pids", return_value=[4321]
                ), \
                mock.patch.object(
                    cli.gpu_manager, "unsafe_gpus_for_process", return_value=None
                ), \
                redirect_stdout(output):
            allowed = cli._check_external_runtime_gpus()

        self.assertTrue(allowed)
        self.assertIn("GPU exposure unverified", output.getvalue())


class GPUAvailabilityReportTests(unittest.TestCase):
    def test_reports_counts_and_labels_each_gpu(self):
        def gpu(index, used, util):
            return GPUInfo(
                index=index, name="RTX A4000", memory_total_mib=16384,
                memory_used_mib=used, memory_free_mib=16384 - used,
                temperature_c=50, utilization_percent=util,
                driver_version="470.86",
            )

        availability = cli.GPUAvailability(
            in_use=[gpu("0", 6658, 80)], free=[gpu("2", 3, 0), gpu("3", 3, 0)]
        )
        output = io.StringIO()
        with mock.patch.object(
            cli.gpu_manager, "availability", return_value=availability
        ), redirect_stdout(output):
            cli._print_gpu_availability()

        printed = output.getvalue()
        self.assertIn("1 of 3 GPU(s) in use by other processes, 2 free", printed)
        self.assertIn("GPU 0", printed)
        self.assertIn("IN USE", printed)
        self.assertIn("GPU 2 ('RTX A4000'): free", printed)


if __name__ == "__main__":
    unittest.main()
