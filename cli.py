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

from config import (
    config,
)
from services.gpu import GPUAvailability, GPUManager
from services.gpu_safety import (
    CONFIRM,
    REFUSE,
    GPUSafetyService,
    runtime_inspector_for,
)
from services.model import ModelManager
from services.benchmark import BenchmarkError, BenchmarkService


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_DIR = PROJECT_ROOT / "run"
LOG_DIR = PROJECT_ROOT / "logs"
PID_FILE = RUN_DIR / "backend.pid"
LOG_FILE = LOG_DIR / "backend.log"
START_TIMEOUT_SECONDS = 120

app = typer.Typer(help="Operate the Nemoclaw backend.", no_args_is_help=True)
model_app = typer.Typer(help="Manage configured models.")
gpu_app = typer.Typer(help="Inspect backend GPU state.")
benchmark_app = typer.Typer(help="Run local backend benchmarks.")
app.add_typer(model_app, name="model")
app.add_typer(gpu_app, name="gpu")
app.add_typer(benchmark_app, name="benchmark")
gpu_manager = GPUManager(config)
model_manager = ModelManager()
benchmark_service = BenchmarkService(config, model_manager, gpu_manager)
gpu_safety_service = GPUSafetyService(
    config, gpu_manager, runtime_inspector_for(config)
)


@dataclass
class BackendState:
    pid: Optional[int]
    pid_running: bool
    pid_matches_backend: bool
    health: str
    health_ok: bool
    port_open: bool
    matching_processes: List[str]
    lifecycle_state: str = "unknown"
    # Runtime model reported by /health, which can differ from config.yaml's
    # model.id after a non-persisted './backend model switch'.
    loaded_model: Optional[str] = None
    target_model: Optional[str] = None

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


def _admin_url(path: str) -> str:
    return "http://{}:{}{}".format(config.backend.host, config.backend.port, path)


def _post_json(url: str, payload: Optional[dict] = None, timeout: int = 5):
    """POST JSON to url. Returns (status_code, body_text); status_code is
    None when the backend is unreachable."""
    data = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")
    except urllib.error.URLError as exc:
        return None, "unavailable ({})".format(exc)


def _echo_lifecycle_result(
    status_code: Optional[int], body: str, requested_model: Optional[str], as_json: bool
) -> None:
    """Prints a lifecycle result and exits non-zero on failure."""
    if status_code is None:
        if as_json:
            typer.echo(json.dumps({"error": "unreachable", "detail": body}))
        else:
            typer.echo("Backend is not reachable: {}".format(body))
        raise typer.Exit(code=1)

    try:
        data = json.loads(body)
    except ValueError:
        typer.echo("Unexpected response from backend: {}".format(body))
        raise typer.Exit(code=1)

    if as_json:
        typer.echo(json.dumps(data))
        raise typer.Exit(code=0 if status_code == 200 else 1)

    if status_code != 200:
        typer.echo("Failed: {}".format(data.get("detail", body)))
        typer.echo("Lifecycle: {}".format(data.get("lifecycle_state", "unknown")))
        raise typer.Exit(code=1)

    if requested_model:
        typer.echo("Requested model: {}".format(requested_model))
    typer.echo("Previous model: {}".format(data.get("previous_model") or "none"))
    typer.echo("Loaded model: {}".format(data.get("loaded_model") or "none"))
    typer.echo("Lifecycle: {}".format(data.get("lifecycle_state", "unknown")))
    typer.echo("Elapsed: {}s".format(data.get("elapsed_seconds", 0)))
    if data.get("persisted"):
        typer.echo("Persisted to config.yaml: yes")


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


def _lifecycle_result():
    body = _health_text()
    if body.startswith("unavailable"):
        return "unknown", False

    try:
        data = json.loads(body)
        lifecycle_state = data.get("lifecycle_state", "unknown")
    except ValueError:
        lifecycle_state = "unknown"

    return lifecycle_state, True


def _lifecycle_state() -> str:
    state, _ = _lifecycle_result()
    return state


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
    loaded_model, target_model = _runtime_models()
    return BackendState(
        pid=pid,
        pid_running=pid_running,
        pid_matches_backend=pid_matches_backend,
        health=health,
        health_ok=health_ok,
        port_open=_port_is_open(),
        matching_processes=_matching_backend_processes(),
        lifecycle_state=_lifecycle_state(),
        loaded_model=loaded_model,
        target_model=target_model,
    )


def _runtime_models():
    body = _health_text()
    if body.startswith("unavailable"):
        return None, None
    try:
        data = json.loads(body)
    except ValueError:
        return None, None
    return data.get("loaded_model"), data.get("target_model")


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
    typer.echo("Quantization: {}".format(config.model.quantization))
    typer.echo("Max tokens default: {}".format(config.model.max_tokens_default))
    typer.echo("Temperature default: {}".format(config.model.temperature_default))


def _scan_local_cache():
    """Wraps engines.transformers_engine.scan_local_cache().

    Returns None (not {}) when torch/transformers aren't installed in
    this environment (e.g. an Ollama-only Local Node) so callers can skip
    the cache display line entirely instead of reporting a misleading
    "no" for something that was never actually checked.
    """
    try:
        from engines.transformers_engine import scan_local_cache
    except ImportError:
        return None
    return scan_local_cache()


def _display_model(model, current_id: str, detailed: bool = False, cache_lookup=None) -> None:
    marker = " (current)" if str(model["id"]) == current_id else ""
    typer.echo("Model: {}{}".format(model["id"], marker))
    typer.echo("  Name: {}".format(model.get("name", model["id"])))
    typer.echo("  Path: {}".format(model.get("path", model["id"])))
    typer.echo("  Engine: {}".format(model.get("engine", "transformers")))
    typer.echo("  Device: {}".format(model.get("device", "cuda")))
    if cache_lookup is not None and model.get("engine", "transformers") == "transformers":
        cached = cache_lookup.get(str(model["id"]))
        if cached is not None:
            typer.echo("  Cached locally: yes ({})".format(cached.size_on_disk_str))
        else:
            typer.echo("  Cached locally: no (first load will download from Hugging Face)")

    if detailed:
        if "gpu" in model:
            typer.echo("  GPU: {}".format(model["gpu"]))
        if "max_tokens_default" in model:
            typer.echo("  Max tokens default: {}".format(model["max_tokens_default"]))
        else:
            typer.echo("  Max tokens default: {}".format(config.model.max_tokens_default))
        if "temperature_default" in model:
            typer.echo("  Temperature default: {}".format(model["temperature_default"]))
        else:
            typer.echo("  Temperature default: {}".format(config.model.temperature_default))


def _mib(value) -> str:
    if value is None:
        return "unavailable"
    return "{} MiB".format(value)


def _percent(value) -> str:
    if value is None:
        return "unavailable"
    return "{}%".format(value)


def _temperature(value) -> str:
    if value is None:
        return "unavailable"
    return "{} C".format(value)


def _vram_usage(gpu) -> str:
    return "{} / {}".format(_mib(gpu.memory_used_mib), _mib(gpu.memory_total_mib))


def _print_gpu(gpu) -> None:
    typer.echo("GPU {}".format(gpu.index))
    typer.echo("  Name: {}".format(gpu.name))
    typer.echo("  Total VRAM: {}".format(_mib(gpu.memory_total_mib)))
    typer.echo("  Used VRAM: {}".format(_mib(gpu.memory_used_mib)))
    typer.echo("  Free VRAM: {}".format(_mib(gpu.memory_free_mib)))
    typer.echo("  Temperature: {}".format(_temperature(gpu.temperature_c)))
    typer.echo("  Utilization: {}".format(_percent(gpu.utilization_percent)))


def _print_busy_gpu_warning() -> None:
    """Warn if any configured backend.gpu index already shows significant
    memory usage - can't be this backend's own model if it isn't running,
    so it's evidence of another process on this shared box."""
    busy = gpu_manager.busy_gpus(own_pids=gpu_safety_service._own_pids())
    if not busy:
        typer.echo("Other GPU usage: none detected on configured GPU(s)")
        return
    typer.echo("Other GPU usage: WARNING - configured GPU(s) already busy")
    for gpu in busy:
        typer.echo("  GPU {} ('{}'): {}".format(gpu.index, gpu.name, _vram_usage(gpu)))
    typer.echo("  Driver: {}".format(gpu.driver_version))


def _seconds(value) -> str:
    if value is None:
        return "unavailable"
    return "{:.3f} s".format(value)


def _rate(value, suffix: str) -> str:
    if value is None:
        return "unavailable"
    return "{:.3f} {}".format(value, suffix)


def _print_vram_snapshot(label: str, snapshot) -> None:
    typer.echo("{}:".format(label))
    typer.echo("  GPU: {}".format(snapshot.get("gpu", "unavailable")))
    if snapshot.get("name"):
        typer.echo("  Name: {}".format(snapshot["name"]))
    typer.echo("  Total VRAM: {}".format(_mib(snapshot.get("memory_total_mib"))))
    typer.echo("  Used VRAM: {}".format(_mib(snapshot.get("memory_used_mib"))))
    typer.echo("  Free VRAM: {}".format(_mib(snapshot.get("memory_free_mib"))))
    typer.echo("  Temperature: {}".format(_temperature(snapshot.get("temperature_c"))))
    typer.echo("  Utilization: {}".format(_percent(snapshot.get("utilization_percent"))))


def _print_benchmark_result(result, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
        return

    typer.echo("Benchmark: {}".format(result["benchmark"]))
    typer.echo("Model: {}".format(result["model"]))
    typer.echo("GPU: {}".format(result["gpu"]))
    typer.echo("Endpoint: {}".format(result["endpoint"]))
    typer.echo("Runs: {}".format(result["runs"]))
    typer.echo("Concurrency: {}".format(result["concurrency"]))
    if result.get("concurrency_note"):
        typer.echo("Concurrency note: {}".format(result["concurrency_note"]))

    if result["benchmark"] == "latency":
        typer.echo("Average latency: {}".format(_seconds(result.get("average_seconds"))))
        typer.echo("Minimum latency: {}".format(_seconds(result.get("min_seconds"))))
        typer.echo("Maximum latency: {}".format(_seconds(result.get("max_seconds"))))
    elif result["benchmark"] == "throughput":
        typer.echo("Elapsed: {}".format(_seconds(result.get("elapsed_seconds"))))
        typer.echo("Requests/sec: {}".format(
            _rate(result.get("requests_per_second"), "requests/sec")
        ))
        typer.echo("Tokens/sec: {}".format(
            _rate(result.get("tokens_per_second"), "tokens/sec")
        ))
    elif result["benchmark"] == "vram":
        _print_vram_snapshot("Before", result.get("vram_before", {}))
        _print_vram_snapshot("Peak", result.get("vram_peak", {}))
        _print_vram_snapshot("After", result.get("vram_after", {}))
    elif result["benchmark"] == "first-token-latency":
        if not result.get("available"):
            typer.echo("Available: no")
            typer.echo("Message: {}".format(result.get("message")))
        else:
            typer.echo("Average first token: {}".format(
                _seconds(result.get("average_seconds"))
            ))
            typer.echo("Min: {}".format(_seconds(result.get("min_seconds"))))
            typer.echo("Max: {}".format(_seconds(result.get("max_seconds"))))
            reasoning = [
                run.get("first_reasoning_seconds")
                for run in result.get("results", [])
                if run.get("first_reasoning_seconds") is not None
            ]
            if reasoning:
                # Reasoning arrives first for a thinking model; showing it
                # separately makes clear why the answer took as long as it did.
                typer.echo(
                    "First reasoning token: {} average (excluded from the "
                    "metric)".format(_seconds(sum(reasoning) / len(reasoning)))
                )


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def _print_gpu_availability(availability=None) -> None:
    """Box-wide census of who is using what. Checked dynamically across
    every detected GPU - nothing assumes which indexes are usually busy on
    this shared machine."""
    if availability is None:
        availability = gpu_manager.availability(
            own_pids=gpu_safety_service._own_pids()
        )
    typer.echo("GPU availability: {}".format(availability.summary_line()))
    for gpu in availability.in_use:
        typer.echo(
            "  GPU {} ('{}'): IN USE by another process - {}, {}".format(
                gpu.index, gpu.name, _vram_usage(gpu), _percent(gpu.utilization_percent)
            )
        )
    for gpu in availability.ours:
        typer.echo(
            "  GPU {} ('{}'): our own model (reclaimable) - {}".format(
                gpu.index, gpu.name, _vram_usage(gpu)
            )
        )
    for gpu in availability.free:
        typer.echo(
            "  GPU {} ('{}'): free - {}".format(gpu.index, gpu.name, _vram_usage(gpu))
        )


def _print_runtime_exposure(exposure) -> None:
    """Renders one external-runtime finding from a StartDecision."""
    if not exposure.verified:
        typer.echo(
            "Ollama daemon (PID {}): cannot read its CUDA_VISIBLE_DEVICES; "
            "GPU exposure unverified.".format(exposure.pid)
        )
        return

    typer.echo(
        "WARNING: the Ollama daemon (PID {}) can reach GPU(s) that another "
        "process is already using:".format(exposure.pid)
    )
    for gpu in exposure.unsafe:
        typer.echo(
            "  GPU {} ('{}'): {}, {}".format(
                gpu.index, gpu.name, _vram_usage(gpu), _percent(gpu.utilization_percent)
            )
        )
    typer.echo(
        "  backend.gpu ({}) does NOT constrain the daemon - it places "
        "models itself.".format(config.backend.gpu)
    )

    if exposure.has_safe_placement:
        indexes = ",".join(str(gpu.index) for gpu in exposure.usable)
        typer.echo(
            "  Free GPU(s) remain ({}), and Ollama schedules by free VRAM, "
            "so it should pick those - proceeding.".format(
                ", ".join(str(gpu.index) for gpu in exposure.usable)
            )
        )
        typer.echo(
            "  To rule it out entirely, restart the daemon pinned to them:"
            "\n    CUDA_VISIBLE_DEVICES={} ollama serve".format(indexes)
        )
    else:
        typer.echo("  No free GPU is available on this box right now.")


def _check_gpu_before_start(force: bool = False) -> bool:
    """Renders the GPU safety verdict and returns whether to proceed.

    The decision itself belongs to GPUSafetyService; this only formats it
    and asks the operator when the service says a human should choose.
    """
    decision = gpu_safety_service.evaluate_start(force=force)

    _print_gpu_availability(decision.availability)
    for exposure in decision.exposures:
        _print_runtime_exposure(exposure)

    blocking = decision.blocking_exposure
    if blocking is not None:
        if decision.forced:
            typer.echo(
                "--force set: starting anyway. The daemon has no free GPU "
                "and may place a model on another user's job."
            )
            return True
        typer.echo(
            "Refusing to start: every GPU the model runtime can reach is "
            "already in use, so it has no safe placement. Wait for a GPU to "
            "free up, or rerun with --force."
        )
        return False

    if decision.configured_busy:
        typer.echo("WARNING: configured GPU(s) already in use by another process:")
        for gpu in decision.configured_busy:
            typer.echo(
                "  GPU {} ('{}'): {}".format(gpu.index, gpu.name, _vram_usage(gpu))
            )

    if decision.forced and decision.configured_busy:
        typer.echo(
            "--force set: starting anyway, ON TOP of the GPU(s) listed above. "
            "Another user's job may be disrupted."
        )
        return True

    if decision.outcome == REFUSE:
        typer.echo("Idle GPU(s) available instead:")
        for gpu in decision.alternatives:
            typer.echo(
                "  GPU {} ('{}'): {}".format(gpu.index, gpu.name, _vram_usage(gpu))
            )
        typer.echo(
            "Refusing to start on a busy GPU while an idle alternative "
            "exists. Update backend.gpu in config/config.yaml and try "
            "again, or rerun with --force to override."
        )
        return False

    if decision.outcome == CONFIRM:
        typer.echo("No idle GPU alternative detected on this box.")
        return typer.confirm("Continue starting on the busy GPU(s) anyway?")

    return True


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
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Start even if the configured GPU already shows usage from "
        "another process, skipping the check/prompt entirely.",
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

    if not _check_gpu_before_start(force=force):
        typer.echo("Not starting.")
        raise typer.Exit(code=1)

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
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Start even if the configured GPU already shows usage from "
        "another process, skipping the check/prompt entirely.",
    ),
):
    """Restart the backend."""
    stop()
    start(wait=wait, timeout=timeout, force=force)


@app.command()
def status():
    """Show backend status."""
    state = _backend_state()
    current_gpu = gpu_manager.current()

    typer.echo("Backend status")
    typer.echo("Running: {}".format("yes" if state.running else "no"))
    typer.echo("Managed by CLI: {}".format("yes" if state.managed_by_cli else "no"))
    typer.echo("PID: {}".format(state.pid if state.pid else "none"))
    typer.echo("Model: {}".format(config.model.id))
    if state.loaded_model and state.loaded_model != config.model.id:
        typer.echo("Loaded model (runtime): {}".format(state.loaded_model))
    if state.target_model:
        typer.echo("Target model (transition): {}".format(state.target_model))
    typer.echo("GPU: {}".format(config.backend.gpu))
    typer.echo("Host: {}".format(config.backend.host))
    typer.echo("Port: {}".format(config.backend.port))
    typer.echo("Health: {}".format(state.health))
    typer.echo("Lifecycle: {}".format(state.lifecycle_state))
    typer.echo("Port open: {}".format("yes" if state.port_open else "no"))
    typer.echo("Process match: {}".format("yes" if state.matching_processes else "no"))
    typer.echo("VRAM: {}".format(_mib(current_gpu.available_memory_mib)))
    typer.echo("Temperature: {}".format(_temperature(
        next(
            (
                gpu.temperature_c
                for gpu in gpu_manager.detect_gpus()
                if str(gpu.index) == str(config.backend.gpu)
            ),
            None,
        )
    )))
    _print_busy_gpu_warning()
    _print_gpu_availability()
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


@model_app.command("list")
def model_list():
    """Show all configured models."""
    current_id = model_manager.selected_model_id()
    cache_lookup = _scan_local_cache()

    typer.echo("Configured models")
    for model in model_manager.list_models():
        _display_model(model, current_id, cache_lookup=cache_lookup)


@model_app.command("local")
def model_local():
    """Show model repos actually present in the local Hugging Face cache.

    Read-only: reports real disk state, independent of config.yaml's
    model.available catalog. Never downloads or deletes anything.
    """
    cache_lookup = _scan_local_cache()
    if cache_lookup is None:
        typer.echo("torch/transformers are not installed in this environment;")
        typer.echo("cannot scan the local Hugging Face cache here.")
        raise typer.Exit(code=1)
    if not cache_lookup:
        typer.echo("No cached Hugging Face models found on this machine.")
        return

    configured_ids = {str(m["id"]) for m in model_manager.list_models()}
    typer.echo("Locally cached Hugging Face models")
    for repo_id, cached in sorted(cache_lookup.items()):
        in_catalog = " (in config.yaml)" if repo_id in configured_ids else " (not in config.yaml)"
        typer.echo("Model: {}{}".format(repo_id, in_catalog))
        typer.echo("  Size on disk: {}".format(cached.size_on_disk_str))
        typer.echo("  Last modified: {}".format(cached.last_modified_str))


@model_app.command("current")
def model_current():
    """Show the selected default model."""
    current_id = model_manager.selected_model_id()

    typer.echo("Selected/default model")
    try:
        model = model_manager.current_model()
    except ValueError:
        typer.echo("Model is selected but not configured: {}".format(current_id))
        raise typer.Exit(code=1)
    _display_model(model, current_id, detailed=True, cache_lookup=_scan_local_cache())
    typer.echo("Loaded model: determined by the running backend process")


@model_app.command("use")
def model_use(model_id: str):
    """Select a configured model for the next backend start."""
    try:
        model_manager.validate_model(model_id)
    except ValueError:
        typer.echo("Model is not configured: {}".format(model_id))
        typer.echo("Run './backend model list' to see configured models")
        raise typer.Exit(code=1)

    current_id = model_manager.selected_model_id()
    if model_id == current_id:
        typer.echo("Model already selected: {}".format(model_id))
        return

    model_manager.select_model(model_id)
    typer.echo("Selected model updated: {}".format(model_id))
    typer.echo("Config: {}".format(model_manager.config_path))

    state = _backend_state()
    if state.running:
        typer.echo("Backend is currently running")
        typer.echo("Restart required for this change to affect the loaded model")


@model_app.command("info")
def model_info(model_id: str):
    """Show details for a configured model."""
    current_id = model_manager.selected_model_id()

    try:
        model = model_manager.model_info(model_id)
    except ValueError:
        typer.echo("Model is not configured: {}".format(model_id))
        typer.echo("Run './backend model list' to see configured models")
        raise typer.Exit(code=1)

    typer.echo("Configured model info")
    _display_model(model, current_id, detailed=True, cache_lookup=_scan_local_cache())
    typer.echo("Configured model: yes")
    typer.echo("Selected/default model: {}".format("yes" if model_id == current_id else "no"))
    typer.echo("Loaded model: determined by the running backend process")


@model_app.command("load")
def model_load(
    model_id: str,
    timeout: int = typer.Option(120, "--timeout", help="Seconds to wait for the transition."),
    persist: bool = typer.Option(False, "--persist", help="Also write model.id to config.yaml."),
    as_json: bool = typer.Option(False, "--json", help="Print the raw JSON result."),
):
    """Load a model into the running backend."""
    status_code, body = _post_json(
        _admin_url("/admin/model/load"),
        {"model_id": model_id, "persist": persist},
        timeout=timeout,
    )
    _echo_lifecycle_result(status_code, body, model_id, as_json)


@model_app.command("unload")
def model_unload(
    timeout: int = typer.Option(120, "--timeout", help="Seconds to wait for the transition."),
    as_json: bool = typer.Option(False, "--json", help="Print the raw JSON result."),
):
    """Unload the currently loaded model."""
    status_code, body = _post_json(
        _admin_url("/admin/model/unload"), timeout=timeout
    )
    _echo_lifecycle_result(status_code, body, None, as_json)


@model_app.command("switch")
def model_switch(
    model_id: str,
    timeout: int = typer.Option(120, "--timeout", help="Seconds to wait for the transition."),
    persist: bool = typer.Option(False, "--persist", help="Also write model.id to config.yaml."),
    as_json: bool = typer.Option(False, "--json", help="Print the raw JSON result."),
):
    """Switch the running backend to a different model."""
    status_code, body = _post_json(
        _admin_url("/admin/model/switch"),
        {"model_id": model_id, "persist": persist},
        timeout=timeout,
    )
    _echo_lifecycle_result(status_code, body, model_id, as_json)


@gpu_app.command("list")
def gpu_list():
    """Show detected GPUs."""
    gpus = gpu_manager.detect_gpus()
    if not gpus:
        typer.echo("No GPUs detected")
        return

    typer.echo("Detected GPUs")
    for gpu in gpus:
        _print_gpu(gpu)

    typer.echo("")
    _print_gpu_availability()


@gpu_app.command("pin-free")
def gpu_pin_free(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show the decision without restarting anything."
    ),
):
    """Restart the Ollama daemon pinned to the GPUs that are free now.

    Ollama splits a model across every GPU it can see once one card isn't
    enough, so on a shared box it takes a slice of cards other people
    could otherwise have had to themselves. CUDA_VISIBLE_DEVICES is the
    only lever, and it can only be set at daemon start - hence a restart.

    Chosen fresh each run rather than pinned in config: which GPUs are
    free changes, and a static pin would fail exactly when the configured
    cards are taken and others are idle.
    """
    if config.backend.engine != "ollama":
        typer.echo(
            "backend.engine is '{}'; GPU pinning applies to the Ollama "
            "daemon only.".format(config.backend.engine)
        )
        raise typer.Exit(code=1)

    from engines.ollama_engine import find_daemon_pids, restart_daemon_pinned

    pids = find_daemon_pids()
    if not pids:
        typer.echo("No local Ollama daemon found; nothing to pin.")
        raise typer.Exit(code=1)

    _print_gpu_availability()

    required_mib = _estimated_model_vram_mib()
    if required_mib is None:
        typer.echo(
            "Cannot determine the VRAM needed for '{}' (is the tag pulled "
            "and the daemon reachable?).".format(config.model.id)
        )
        raise typer.Exit(code=1)

    selection = gpu_manager.select_gpus_for(
        required_mib, own_pids=gpu_safety_service._own_pids()
    )
    typer.echo(
        "Model {} needs ~{} (estimated from its on-disk size).".format(
            config.model.id, _mib(required_mib)
        )
    )

    if selection is None:
        typer.echo(
            "Refusing: the GPUs that are free right now cannot hold it. "
            "Wait for a GPU to free up, or configure a smaller model."
        )
        raise typer.Exit(code=1)

    indexes = [str(gpu.index) for gpu in selection]
    typer.echo(
        "Selected GPU {} ({} of {} card(s), lowest index first).".format(
            ",".join(indexes), len(indexes), len(gpu_manager.detect_gpus())
        )
    )
    for gpu in selection:
        typer.echo("  GPU {} ('{}'): {}".format(gpu.index, gpu.name, _vram_usage(gpu)))

    _print_model_fit_table(selection)

    if dry_run:
        typer.echo("Dry run: daemon left untouched.")
        return

    typer.echo(
        "Restarting the Ollama daemon drops any resident model (it "
        "reloads on the next request) and briefly interrupts inference."
    )
    if not yes and not typer.confirm(
        "Restart Ollama daemon PID {} pinned to GPU {}?".format(
            pids[0], ",".join(indexes)
        )
    ):
        typer.echo("Not restarting.")
        raise typer.Exit(code=1)

    try:
        new_pid = restart_daemon_pinned(pids[0], indexes)
    except (OSError, RuntimeError) as exc:
        typer.echo("Failed to restart the daemon: {}".format(exc))
        raise typer.Exit(code=1)

    typer.echo("Ollama daemon restarted as PID {} on GPU {}".format(new_pid, ",".join(indexes)))
    typer.echo(
        "Note: this is not persistent. A daemon restart or reboot returns "
        "to whatever its startup command sets."
    )


def _estimated_model_vram_mib():
    from engines.ollama_engine import OllamaEngine

    try:
        return OllamaEngine(config).estimated_vram_mib()
    except Exception:
        return None


def _pulled_model_sizes():
    from engines.ollama_engine import OllamaEngine

    try:
        return OllamaEngine(config).pulled_model_sizes()
    except Exception:
        return {}


def _print_model_fit_table(selection) -> None:
    """Show every pulled model against the chosen GPUs.

    The frontend can switch model at any time, but the daemon's GPU set
    only changes when it restarts - so a later switch can outgrow this
    pin. Listing the other pulled tags now makes that visible before it
    bites, instead of at load time.
    """
    sizes = _pulled_model_sizes()
    if len(sizes) < 2:
        return

    capacity = sum(gpu.memory_total_mib or 0 for gpu in selection)
    typer.echo("Pulled models against this selection ({} total):".format(_mib(capacity)))
    for name, required in sorted(sizes.items(), key=lambda item: -item[1]):
        marker = "fits" if required <= capacity else "DOES NOT FIT"
        current = " (current)" if name == config.model.id else ""
        typer.echo("  {:<28} ~{:>8} - {}{}".format(name, _mib(required), marker, current))
    typer.echo(
        "  Switching to a model marked DOES NOT FIT needs another "
        "'gpu pin-free' run, or more free GPUs."
    )


@gpu_app.command("current")
def gpu_current():
    """Show the configured backend GPU and current CUDA state."""
    current_gpu = gpu_manager.current()

    typer.echo("Current GPU")
    typer.echo("Selected CUDA device: {}".format(current_gpu.selected_cuda_device))
    typer.echo("Current backend GPU: {}".format(current_gpu.backend_gpu))
    typer.echo("Current model: {}".format(current_gpu.current_model))
    typer.echo("Available memory: {}".format(_mib(current_gpu.available_memory_mib)))
    typer.echo("CUDA available: {}".format("yes" if current_gpu.cuda_available else "no"))
    typer.echo("Torch current device: {}".format(current_gpu.torch_current_device))
    typer.echo("Driver: {}".format(current_gpu.driver_version))
    _print_busy_gpu_warning()


@gpu_app.command("monitor")
def gpu_monitor(interval: float = typer.Option(2.0, "--interval", "-i", min=0.5)):
    """Refresh GPU utilization, VRAM, and temperature until Ctrl+C."""
    typer.echo("GPU monitor. Press Ctrl+C to stop.")
    try:
        while True:
            gpus = gpu_manager.detect_gpus()
            if not gpus:
                typer.echo("No GPUs detected")
            else:
                typer.echo("GPU status")
                for gpu in gpus:
                    typer.echo(
                        "GPU {} | util {} | vram {} | temp {}".format(
                            gpu.index,
                            _percent(gpu.utilization_percent),
                            _vram_usage(gpu),
                            _temperature(gpu.temperature_c),
                        )
                    )
            time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo("GPU monitor stopped")


@benchmark_app.command("latency")
def benchmark_latency(
    prompt: str = typer.Option(
        "Say hello from Nemoclaw in one sentence.",
        "--prompt",
        "-p",
        help="Prompt to send to the local backend.",
    ),
    max_tokens: int = typer.Option(
        config.model.max_tokens_default,
        "--max-tokens",
        min=1,
        help="Maximum completion tokens.",
    ),
    runs: int = typer.Option(3, "--runs", "-r", min=1, help="Number of requests."),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "-c",
        min=1,
        help="Accepted for future concurrent benchmarking; currently sequential.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print structured JSON."),
):
    """Measure request latency against the OpenAI-compatible endpoint."""
    try:
        result = benchmark_service.latency(prompt, max_tokens, runs, concurrency)
    except (BenchmarkError, ValueError) as exc:
        typer.echo("Benchmark failed: {}".format(exc))
        raise typer.Exit(code=1)
    _print_benchmark_result(result, json_output)


@benchmark_app.command("throughput")
def benchmark_throughput(
    prompt: str = typer.Option(
        "Say hello from Nemoclaw in one sentence.",
        "--prompt",
        "-p",
        help="Prompt to send to the local backend.",
    ),
    max_tokens: int = typer.Option(
        config.model.max_tokens_default,
        "--max-tokens",
        min=1,
        help="Maximum completion tokens.",
    ),
    runs: int = typer.Option(3, "--runs", "-r", min=1, help="Number of requests."),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "-c",
        min=1,
        help="Accepted for future concurrent benchmarking; currently sequential.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print structured JSON."),
):
    """Measure request and token throughput."""
    try:
        result = benchmark_service.throughput(prompt, max_tokens, runs, concurrency)
    except (BenchmarkError, ValueError) as exc:
        typer.echo("Benchmark failed: {}".format(exc))
        raise typer.Exit(code=1)
    _print_benchmark_result(result, json_output)


@benchmark_app.command("vram")
def benchmark_vram(
    prompt: str = typer.Option(
        "Say hello from Nemoclaw in one sentence.",
        "--prompt",
        "-p",
        help="Prompt to send to the local backend.",
    ),
    max_tokens: int = typer.Option(
        config.model.max_tokens_default,
        "--max-tokens",
        min=1,
        help="Maximum completion tokens.",
    ),
    runs: int = typer.Option(3, "--runs", "-r", min=1, help="Number of requests."),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "-c",
        min=1,
        help="Accepted for future concurrent benchmarking; currently sequential.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print structured JSON."),
):
    """Measure VRAM before, during, and after local endpoint requests."""
    try:
        result = benchmark_service.vram(prompt, max_tokens, runs, concurrency)
    except (BenchmarkError, ValueError) as exc:
        typer.echo("Benchmark failed: {}".format(exc))
        raise typer.Exit(code=1)
    _print_benchmark_result(result, json_output)


@benchmark_app.command("first-token-latency")
def benchmark_first_token_latency(
    prompt: str = typer.Option(
        "Say hello from Nemoclaw in one sentence.",
        "--prompt",
        "-p",
        help="Prompt to send to the local backend.",
    ),
    max_tokens: int = typer.Option(
        config.model.max_tokens_default,
        "--max-tokens",
        min=1,
        help="Maximum completion tokens.",
    ),
    runs: int = typer.Option(3, "--runs", "-r", min=1, help="Number of requests."),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "-c",
        min=1,
        help="Accepted for future concurrent benchmarking; currently sequential.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print structured JSON."),
):
    """Measure time to the first answer token over a real stream."""
    try:
        result = benchmark_service.first_token_latency(prompt, max_tokens, runs, concurrency)
    except (BenchmarkError, ValueError) as exc:
        typer.echo("Benchmark failed: {}".format(exc))
        raise typer.Exit(code=1)
    _print_benchmark_result(result, json_output)


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
