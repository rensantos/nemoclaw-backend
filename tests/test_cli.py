import importlib
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


def _install_typer_stub():
    if "typer" in sys.modules:
        return

    typer_stub = types.ModuleType("typer")

    class FakeTyper:
        def __init__(self, *args, **kwargs):
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

    def test_gpu_info_parses_nvidia_smi_output(self):
        result = types.SimpleNamespace(stdout="512, 16384, 45\n", stderr="")

        with mock.patch.object(cli.subprocess, "run", return_value=result):
            vram, temperature = cli._gpu_info()

        self.assertEqual(vram, "512 / 16384 MiB")
        self.assertEqual(temperature, "45 C")

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

        with mock.patch.object(cli, "_backend_state", return_value=state), \
                mock.patch.object(cli, "_gpu_info", return_value=("unavailable", "unavailable")):
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


if __name__ == "__main__":
    unittest.main()
