import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import typer

from config import config


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_DIR = PROJECT_ROOT / "run"
LOG_DIR = PROJECT_ROOT / "logs"
PID_FILE = RUN_DIR / "backend.pid"
LOG_FILE = LOG_DIR / "backend.log"

app = typer.Typer(help="Operate the Nemoclaw backend.")


def _health_url() -> str:
    return "http://{}:{}/health".format(config.backend.host, config.backend.port)


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
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


def _health_status() -> str:
    body = _health_text()
    if body.startswith("unavailable"):
        return body
    try:
        data = json.loads(body)
        return data.get("status", "ok")
    except ValueError:
        return body


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
def start():
    """Start the backend in the background."""
    pid = _read_pid()
    if pid and _pid_is_running(pid):
        typer.echo("Backend already running with PID {}".format(pid))
        return

    RUN_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", config.backend.gpu)

    with LOG_FILE.open("ab") as log_file:
        process = subprocess.Popen(
            [sys.executable, "server.py"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    typer.echo("Backend started with PID {}".format(process.pid))
    typer.echo("Log: {}".format(LOG_FILE))


@app.command()
def stop():
    """Stop the backend."""
    pid = _read_pid()
    if pid and _pid_is_running(pid):
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not _pid_is_running(pid):
                break
            time.sleep(0.25)
        if _pid_is_running(pid):
            os.kill(pid, signal.SIGKILL)
        typer.echo("Backend stopped")
    else:
        typer.echo("Backend is not running from {}".format(PID_FILE))

    if PID_FILE.exists():
        PID_FILE.unlink()

    subprocess.run(["pkill", "-f", "uvicorn server:app --host .* --port .*"], check=False)
    subprocess.run(["pkill", "-f", "python server.py"], check=False)


@app.command()
def restart():
    """Restart the backend."""
    stop()
    start()


@app.command()
def status():
    """Show backend status."""
    pid = _read_pid()
    running = bool(pid and _pid_is_running(pid))
    vram, temperature = _gpu_info()

    typer.echo("Backend status")
    typer.echo("Running: {}".format("yes" if running else "no"))
    typer.echo("Model: {}".format(config.model.id))
    typer.echo("GPU: {}".format(config.backend.gpu))
    typer.echo("Host: {}".format(config.backend.host))
    typer.echo("Port: {}".format(config.backend.port))
    typer.echo("Health: {}".format(_health_status()))
    typer.echo("VRAM: {}".format(vram))
    typer.echo("Temperature: {}".format(temperature))


@app.command()
def health():
    """Call the backend /health endpoint."""
    typer.echo(_health_text())


@app.command("config")
def show_config():
    """Print the active configuration."""
    _print_config()


@app.command()
def logs(lines: int = typer.Option(80, "--lines", "-n"), follow: bool = True):
    """Tail the backend log."""
    if not LOG_FILE.exists():
        typer.echo("Log file does not exist yet: {}".format(LOG_FILE))
        raise typer.Exit(code=1)

    cmd = ["tail", "-n", str(lines)]
    if follow:
        cmd.append("-f")
    cmd.append(str(LOG_FILE))
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    app()
