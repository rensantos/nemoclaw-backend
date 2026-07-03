import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import typer

from config import config


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_DIR = PROJECT_ROOT / "run"
LOG_DIR = PROJECT_ROOT / "logs"
PID_FILE = RUN_DIR / "backend.pid"
LOG_FILE = LOG_DIR / "backend.log"
START_TIMEOUT_SECONDS = 120

app = typer.Typer(help="Operate the Nemoclaw backend.", no_args_is_help=True)


@dataclass
class BackendState:
    pid: Optional[int]
    pid_running: bool
    pid_matches_backend: bool
    health: str
    health_ok: bool
    port_open: bool
    matching_processes: List[str]

    @property
    def running(self) -> bool:
        return (
            self.managed_by_cli
            or self.health_ok
            or self.port_open
            or bool(self.matching_processes)
        )

    @property
    def managed_by_cli(self) -> bool:
        return bool(self.pid and self.pid_running and self.pid_matches_backend)


def _health_url() -> str:
    return "http://{}:{}/health".format(config.backend.host, config.backend.port)


def _server_command():
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "server:app",
        "--host",
        str(config.backend.host),
        "--port",
        str(config.backend.port),
    ]


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pid_command(pid: int) -> str:
    proc_cmdline = Path("/proc") / str(pid) / "cmdline"
    try:
        raw = proc_cmdline.read_bytes()
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        pass

    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return ""

    return result.stdout.strip()


def _pid_matches_backend(pid: Optional[int]) -> bool:
    if not pid:
        return False

    command = _pid_command(pid)
    if not command:
        return False

    if "uvicorn" in command and "server:app" in command:
        return True
    if "python" in command and "server.py" in command:
        return True

    return False


def _read_pid() -> Optional[int]:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _health_text() -> str:
    try:
        with urllib.request.urlopen(_health_url(), timeout=3) as response:
            body = response.read().decode("utf-8")
            return body
    except urllib.error.URLError as exc:
        return "unavailable ({})".format(exc)


def _health_result():
    body = _health_text()
    if body.startswith("unavailable"):
        return body, False

    try:
        data = json.loads(body)
        status = data.get("status", "ok")
    except ValueError:
        status = body

    return status, True


def _is_healthy() -> bool:
    try:
        with urllib.request.urlopen(_health_url(), timeout=3) as response:
            return 200 <= response.status < 300
    except urllib.error.URLError:
        return False


def _wait_for_health(timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _is_healthy():
            return True
        time.sleep(1)
    return False


def _health_status() -> str:
    status, _ = _health_result()
    return status


def _health_ok() -> bool:
    _, ok = _health_result()
    return ok


def _port_probe_host() -> str:
    if config.backend.host in ("0.0.0.0", "::"):
        return "127.0.0.1"
    return config.backend.host


def _port_is_open() -> bool:
    try:
        with socket.create_connection(
            (_port_probe_host(), int(config.backend.port)),
            timeout=1,
        ):
            return True
    except OSError:
        return False


def _matching_backend_processes() -> List[str]:
    patterns = [
        "uvicorn",
        "server:app",
        "--port",
        str(config.backend.port),
    ]

    try:
        result = subprocess.run(
            ["pgrep", "-af", "server:app|server.py"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return []

    matches = []
    for line in result.stdout.splitlines():
        if all(pattern in line for pattern in patterns):
            matches.append(line)
            continue
        if "python server.py" in line and str(PROJECT_ROOT) in line:
            matches.append(line)

    return matches


def _backend_state() -> BackendState:
    pid = _read_pid()
    pid_running = bool(pid and _pid_is_running(pid))
    pid_matches_backend = _pid_matches_backend(pid) if pid_running else False
    if pid and not (pid_running and pid_matches_backend):
        _remove_stale_pid(pid)
        pid = None
        pid_running = False
        pid_matches_backend = False

    health, health_ok = _health_result()
    return BackendState(
        pid=pid,
        pid_running=pid_running,
        pid_matches_backend=pid_matches_backend,
        health=health,
        health_ok=health_ok,
        port_open=_port_is_open(),
        matching_processes=_matching_backend_processes(),
    )


def _gpu_info():
    query = "memory.used,memory.total,temperature.gpu"
    cmd = [
        "nvidia-smi",
        "--id={}".format(config.backend.gpu),
        "--query-gpu={}".format(query),
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", "unavailable"

    line = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return "unavailable", "unavailable"

    return "{} / {} MiB".format(parts[0], parts[1]), "{} C".format(parts[2])


def _remove_stale_pid(pid: Optional[int]) -> None:
    if pid and not _pid_is_running(pid):
        if PID_FILE.exists():
            PID_FILE.unlink()
        typer.echo("Removed stale PID file for PID {}".format(pid))


def _terminate_pid(pid: int) -> None:
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not _pid_is_running(pid):
            return
        time.sleep(0.25)
    if _pid_is_running(pid):
        typer.echo("Backend did not stop after SIGTERM; sending SIGKILL")
        os.kill(pid, signal.SIGKILL)


def _print_config() -> None:
    typer.echo("Host: {}".format(config.backend.host))
    typer.echo("Port: {}".format(config.backend.port))
    typer.echo("GPU: {}".format(config.backend.gpu))
    typer.echo("Model: {}".format(config.model.id))
    typer.echo("Max tokens default: {}".format(config.model.max_tokens_default))
    typer.echo("Temperature default: {}".format(config.model.temperature_default))


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def start(
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait briefly for /health after starting.",
    ),
    timeout: int = typer.Option(
        START_TIMEOUT_SECONDS,
        "--timeout",
        min=1,
        help="Seconds to wait for /health.",
    ),
):
    """Start the backend in the background."""
    state = _backend_state()
    if state.managed_by_cli:
        typer.echo("Backend already running with PID {}".format(state.pid))
        return
    if state.running:
        typer.echo("Backend already appears to be running outside CLI management")
        typer.echo("Managed by CLI: no")
        typer.echo("Health: {}".format(state.health))
        typer.echo("Port open: {}".format("yes" if state.port_open else "no"))
        typer.echo("Use './backend status' for details")
        return

    RUN_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", config.backend.gpu)

    with LOG_FILE.open("ab") as log_file:
        process = subprocess.Popen(
            _server_command(),
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    typer.echo("Backend started with PID {}".format(process.pid))
    typer.echo("Model: {}".format(config.model.id))
    typer.echo("GPU: {}".format(config.backend.gpu))
    typer.echo("URL: {}".format(_health_url().replace("/health", "")))
    typer.echo("Log: {}".format(LOG_FILE))

    if wait:
        if _wait_for_health(timeout):
            typer.echo("Health: ok")
        else:
            typer.echo("Health: not ready after {} seconds".format(timeout))
            typer.echo("Check logs with: ./backend logs")


@app.command()
def stop():
    """Stop the backend."""
    state = _backend_state()
    if state.managed_by_cli:
        _terminate_pid(state.pid)
        typer.echo("Backend stopped")
        if PID_FILE.exists():
            PID_FILE.unlink()
    elif state.running:
        typer.echo("Backend appears to be running, but it is not managed by CLI")
        typer.echo("Managed by CLI: no")
        typer.echo("Health: {}".format(state.health))
        typer.echo("Port open: {}".format("yes" if state.port_open else "no"))
        if state.matching_processes:
            typer.echo("Matching process:")
            for process in state.matching_processes:
                typer.echo("  {}".format(process))
        typer.echo("Not stopping unmanaged backend to avoid killing unrelated processes")
        raise typer.Exit(code=1)
    else:
        typer.echo("Backend is not running")
        if PID_FILE.exists():
            PID_FILE.unlink()


@app.command()
def restart(
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait briefly for /health after starting.",
    ),
    timeout: int = typer.Option(
        START_TIMEOUT_SECONDS,
        "--timeout",
        min=1,
        help="Seconds to wait for /health.",
    ),
):
    """Restart the backend."""
    stop()
    start(wait=wait, timeout=timeout)


@app.command()
def status():
    """Show backend status."""
    state = _backend_state()
    vram, temperature = _gpu_info()

    typer.echo("Backend status")
    typer.echo("Running: {}".format("yes" if state.running else "no"))
    typer.echo("Managed by CLI: {}".format("yes" if state.managed_by_cli else "no"))
    typer.echo("PID: {}".format(state.pid if state.pid else "none"))
    typer.echo("Model: {}".format(config.model.id))
    typer.echo("GPU: {}".format(config.backend.gpu))
    typer.echo("Host: {}".format(config.backend.host))
    typer.echo("Port: {}".format(config.backend.port))
    typer.echo("Health: {}".format(state.health))
    typer.echo("Port open: {}".format("yes" if state.port_open else "no"))
    typer.echo("Process match: {}".format("yes" if state.matching_processes else "no"))
    typer.echo("VRAM: {}".format(vram))
    typer.echo("Temperature: {}".format(temperature))
    typer.echo("Log: {}".format(LOG_FILE))


@app.command()
def health():
    """Call the backend /health endpoint."""
    body = _health_text()
    typer.echo(body)
    if body.startswith("unavailable"):
        raise typer.Exit(code=1)


@app.command("config")
def show_config():
    """Print the active configuration."""
    _print_config()


@app.command()
def logs(
    lines: int = typer.Option(80, "--lines", "-n", min=1),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output."),
):
    """Tail the backend log."""
    if not LOG_FILE.exists():
        typer.echo("Log file does not exist yet: {}".format(LOG_FILE))
        state = _backend_state()
        if state.running and not state.managed_by_cli:
            typer.echo("Backend appears to be running outside CLI management")
            typer.echo("Check the launcher or systemd service logs for that process")
        else:
            typer.echo("Start the backend first with: ./backend start")
        return

    cmd = ["tail", "-n", str(lines)]
    if follow:
        cmd.append("-f")
    cmd.append(str(LOG_FILE))
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    app()
