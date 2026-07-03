import importlib
import io
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
    typer_stub.Option = option
    typer_stub.echo = echo
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
        current_gpu = CurrentGPUInfo(
            selected_cuda_device="0",
            backend_gpu="0",
            current_model="tiny",
            available_memory_mib=1024,
            cuda_available=True,
            torch_current_device="0",
            driver_version="535.0",
        )
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


if __name__ == "__main__":
    unittest.main()
