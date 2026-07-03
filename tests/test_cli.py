import importlib
import sys
import types
import unittest
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


if __name__ == "__main__":
    unittest.main()
