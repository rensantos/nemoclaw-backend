"""Start-time GPU safety decisions.

`GPUManager` reports what the GPUs *are*; this decides what to *do* about
it — proceed, ask, or refuse. That decision used to live in `cli.py`,
which AGENTS.md is explicit should be a delivery surface and never an
owner. The CLI now renders the verdict this produces instead of computing
it, which also makes the policy testable without a terminal.

The rules encode two hard-won constraints, both from live use on a shared
box:

- Our own model's VRAM must never count against us. It is reclaimable, and
  treating it as another user's job made the backend refuse to start
  because of the model it was itself serving.
- An external model runtime (the Ollama daemon) places models using the
  devices *it* was launched with, which `backend.gpu` does not constrain.
  But refusing whenever it can merely reach a busy GPU would block almost
  every start on a shared machine and make `--force` routine, destroying
  the value of the warning. So: warn while anything is still free, refuse
  only when the runtime has no safe placement left.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from services.gpu import GPUAvailability, GPUInfo, GPUManager


PROCEED = "proceed"
CONFIRM = "confirm"
REFUSE = "refuse"


@dataclass
class RuntimeExposure:
    """What an external model runtime can currently reach."""

    pid: int
    verified: bool
    unsafe: List[GPUInfo] = field(default_factory=list)
    usable: List[GPUInfo] = field(default_factory=list)

    @property
    def has_safe_placement(self) -> bool:
        return bool(self.usable)


@dataclass
class StartDecision:
    """Everything the CLI needs to explain a start attempt."""

    outcome: str
    availability: GPUAvailability
    exposures: List[RuntimeExposure] = field(default_factory=list)
    configured_busy: List[GPUInfo] = field(default_factory=list)
    alternatives: List[GPUInfo] = field(default_factory=list)
    forced: bool = False

    @property
    def allowed(self) -> bool:
        """True when the start may go ahead without asking."""
        return self.outcome == PROCEED

    @property
    def blocking_exposure(self) -> Optional[RuntimeExposure]:
        """The runtime exposure that caused a refusal, if any."""
        for exposure in self.exposures:
            if exposure.unsafe and not exposure.has_safe_placement:
                return exposure
        return None


class GPUSafetyService:
    """Decides whether it is safe to start serving on this machine."""

    def __init__(self, config, gpu_manager: GPUManager, runtime=None):
        self.config = config
        self.gpu_manager = gpu_manager
        # Optional engine-specific runtime inspector; None when the active
        # engine has no separate model runtime to account for.
        self.runtime = runtime

    def evaluate_start(self, force: bool = False) -> StartDecision:
        own_pids = self._own_pids()
        availability = self.gpu_manager.availability(own_pids=own_pids)
        exposures = self._runtime_exposures(own_pids)

        blocked = [
            exposure for exposure in exposures
            if exposure.unsafe and not exposure.has_safe_placement
        ]
        if blocked:
            return StartDecision(
                outcome=PROCEED if force else REFUSE,
                availability=availability,
                exposures=exposures,
                forced=force,
            )

        configured_busy = self.gpu_manager.busy_gpus(own_pids=own_pids)
        if not configured_busy:
            return StartDecision(
                outcome=PROCEED, availability=availability, exposures=exposures
            )

        if force:
            return StartDecision(
                outcome=PROCEED,
                availability=availability,
                exposures=exposures,
                configured_busy=configured_busy,
                forced=True,
            )

        alternatives = self.gpu_manager.idle_alternative_gpus(own_pids=own_pids)
        return StartDecision(
            # An idle card elsewhere means reconfiguring is the right fix,
            # so refuse. With nothing better to suggest, ask instead of
            # deciding for the operator.
            outcome=REFUSE if alternatives else CONFIRM,
            availability=availability,
            exposures=exposures,
            configured_busy=configured_busy,
            alternatives=alternatives,
        )

    def _own_pids(self) -> List[int]:
        if self.runtime is None:
            return []
        return self.runtime.runtime_pids()

    def _runtime_exposures(self, own_pids) -> List[RuntimeExposure]:
        if self.runtime is None:
            return []

        exposures = []
        for pid in self.runtime.daemon_pids():
            unsafe = self.gpu_manager.unsafe_gpus_for_process(pid, own_pids=own_pids)
            if unsafe is None:
                # Cannot read its visible devices; say so rather than
                # assuming either safety or danger.
                exposures.append(RuntimeExposure(pid=pid, verified=False))
                continue
            if not unsafe:
                continue
            exposures.append(
                RuntimeExposure(
                    pid=pid,
                    verified=True,
                    unsafe=unsafe,
                    # "usable" not "free": a card holding only our own
                    # model can be reclaimed, so it is a safe placement.
                    usable=self.gpu_manager.availability(own_pids=own_pids).usable,
                )
            )
        return exposures


class OllamaRuntimeInspector:
    """Locates the Ollama daemon and the PIDs whose VRAM is ours.

    Keeps the engine-specific lookups behind an interface so
    GPUSafetyService holds policy only, with no knowledge of Ollama.
    """

    def daemon_pids(self) -> List[int]:
        from engines.ollama_engine import find_daemon_pids

        return find_daemon_pids()

    def runtime_pids(self) -> List[int]:
        from engines.ollama_engine import find_runtime_pids

        return find_runtime_pids()


def runtime_inspector_for(config):
    """The runtime inspector for the active engine, or None when the
    engine loads models in-process and has nothing separate to inspect."""
    if config.backend.engine == "ollama":
        return OllamaRuntimeInspector()
    return None
