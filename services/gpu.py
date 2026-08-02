import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GPUInfo:
    index: str
    name: str
    memory_total_mib: Optional[int]
    memory_used_mib: Optional[int]
    memory_free_mib: Optional[int]
    temperature_c: Optional[int]
    utilization_percent: Optional[int]
    driver_version: str = "unavailable"


@dataclass
class GPUAvailability:
    """Box-wide GPU census: which cards are in use by anyone, which are
    free. Deliberately says nothing about backend.gpu - this is "what is
    the state of this shared machine", not "is my configured GPU ok".
    """

    in_use: List["GPUInfo"]
    free: List["GPUInfo"]

    @property
    def total(self) -> int:
        return len(self.in_use) + len(self.free)

    def summary_line(self) -> str:
        if self.total == 0:
            return "No GPUs detected"
        return "{} of {} GPU(s) in use by other processes, {} free".format(
            len(self.in_use), self.total, len(self.free)
        )


@dataclass
class CurrentGPUInfo:
    selected_cuda_device: str
    backend_gpu: str
    current_model: str
    available_memory_mib: Optional[int]
    cuda_available: bool
    torch_current_device: str
    driver_version: str


class GPUManager:
    """Single service responsible for GPU discovery and status."""

    def __init__(self, config):
        self.config = config

    def detect_gpus(self) -> List[GPUInfo]:
        return self._detect_with_nvidia_smi()

    def current(self) -> CurrentGPUInfo:
        gpus = self.detect_gpus()
        backend_gpu = str(self.config.backend.gpu)
        selected = self._gpu_by_index(gpus, backend_gpu)
        cuda_available, torch_current_device = self._torch_cuda_state()

        return CurrentGPUInfo(
            selected_cuda_device=backend_gpu,
            backend_gpu=backend_gpu,
            current_model=self.config.model.id,
            available_memory_mib=selected.memory_free_mib if selected else None,
            cuda_available=cuda_available,
            torch_current_device=torch_current_device,
            driver_version=selected.driver_version if selected else self.driver_version(),
        )

    def gpu_name(self) -> Optional[str]:
        gpus = self.detect_gpus()
        selected = self._gpu_by_index(gpus, str(self.config.backend.gpu))
        return selected.name if selected else None

    def gpu_is_in_use(
        self,
        gpu: GPUInfo,
        threshold_mib: int = 500,
        utilization_threshold_percent: int = 10,
    ) -> bool:
        """Whether a GPU shows signs of someone else's work.

        Memory alone is not enough: a compute-heavy job with a small
        resident footprint reads as idle by VRAM but is very much in use.
        Either signal crossing its threshold counts as busy, because the
        cost of a false positive (we pick a different card) is far lower
        than a false negative (we land on another user's job).
        """
        if gpu.memory_used_mib is not None and gpu.memory_used_mib > threshold_mib:
            return True
        if (
            gpu.utilization_percent is not None
            and gpu.utilization_percent > utilization_threshold_percent
        ):
            return True
        return False

    def availability(
        self,
        threshold_mib: int = 500,
        utilization_threshold_percent: int = 10,
    ) -> GPUAvailability:
        """Census of every GPU on the box, split into in-use and free.

        Checks all detected GPUs dynamically - it never assumes which
        indexes are "the busy ones" on this shared machine.
        """
        in_use, free = [], []
        for gpu in self.detect_gpus():
            target = in_use if self.gpu_is_in_use(
                gpu, threshold_mib, utilization_threshold_percent
            ) else free
            target.append(gpu)
        return GPUAvailability(in_use=in_use, free=free)

    def busy_gpus(
        self,
        threshold_mib: int = 500,
        utilization_threshold_percent: int = 10,
    ) -> List[GPUInfo]:
        """Configured backend.gpu index(es) already showing usage by
        another process. Intended to be checked before this process has
        loaded a model itself, so any usage found belongs to someone else
        - there is no per-process attribution here (nvidia-smi's basic
        query doesn't provide it), just "is this GPU already not idle".
        backend.gpu may be a comma-separated multi-GPU value (e.g. "2,3");
        each index is checked independently.
        """
        gpus = self.detect_gpus()
        indexes = [part.strip() for part in str(self.config.backend.gpu).split(",")]
        busy = []
        for index in indexes:
            gpu = self._gpu_by_index(gpus, index)
            if gpu and self.gpu_is_in_use(
                gpu, threshold_mib, utilization_threshold_percent
            ):
                busy.append(gpu)
        return busy

    def idle_alternative_gpus(
        self,
        threshold_mib: int = 500,
        utilization_threshold_percent: int = 10,
    ) -> List[GPUInfo]:
        """GPUs on this box NOT among backend.gpu's configured index(es)
        that are themselves idle - genuine alternatives a human could
        reconfigure backend.gpu to use instead of a busy configured GPU.
        Pairs with busy_gpus() to decide whether "no other way" applies.
        """
        configured = {part.strip() for part in str(self.config.backend.gpu).split(",")}
        return [
            gpu
            for gpu in self.detect_gpus()
            if str(gpu.index) not in configured
            and not self.gpu_is_in_use(
                gpu, threshold_mib, utilization_threshold_percent
            )
        ]

    def visible_gpu_indexes_for_process(self, pid) -> Optional[List[str]]:
        """Which GPU indexes a running process can actually reach, read
        from its own CUDA_VISIBLE_DEVICES in /proc/<pid>/environ.

        Returns every detected index when the variable is unset - that is
        CUDA's real default (a process sees all GPUs), and treating "unset"
        as "restricted" would be exactly the wrong way to be wrong here.
        Returns None when the environment cannot be read at all (no /proc,
        process gone, not permitted), so callers can distinguish "sees
        everything" from "unknown" instead of guessing.
        """
        try:
            with open("/proc/{}/environ".format(pid), "rb") as environ_file:
                raw = environ_file.read().decode("utf-8", "replace")
        except (OSError, ValueError):
            return None

        all_indexes = [str(gpu.index) for gpu in self.detect_gpus()]
        for entry in raw.split("\0"):
            if entry.startswith("CUDA_VISIBLE_DEVICES="):
                value = entry.split("=", 1)[1].strip()
                if not value:
                    return []
                return [part.strip() for part in value.split(",") if part.strip()]
        return all_indexes

    def unsafe_gpus_for_process(
        self,
        pid,
        threshold_mib: int = 500,
        utilization_threshold_percent: int = 10,
    ) -> Optional[List[GPUInfo]]:
        """GPUs that `pid` can reach AND that someone else is already
        using - i.e. cards this process could disrupt if its scheduler
        chose them.

        This is the check that matters for an external model runtime like
        the Ollama daemon: backend.gpu constrains only this backend's own
        process, while the daemon places weights using whatever devices
        *it* was launched with. Returns None when visibility is unknown.
        """
        visible = self.visible_gpu_indexes_for_process(pid)
        if visible is None:
            return None
        visible_set = set(visible)
        return [
            gpu
            for gpu in self.detect_gpus()
            if str(gpu.index) in visible_set
            and self.gpu_is_in_use(gpu, threshold_mib, utilization_threshold_percent)
        ]

    def driver_version(self) -> str:
        gpus = self.detect_gpus()
        if not gpus:
            return "unavailable"
        return gpus[0].driver_version

    def _detect_with_nvidia_smi(self) -> List[GPUInfo]:
        query = (
            "index,name,memory.total,memory.used,memory.free,"
            "temperature.gpu,utilization.gpu,driver_version"
        )
        command = [
            "nvidia-smi",
            "--query-gpu={}".format(query),
            "--format=csv,noheader,nounits",
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return []

        gpus = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 8:
                continue
            gpus.append(
                GPUInfo(
                    index=parts[0],
                    name=parts[1],
                    memory_total_mib=self._int_or_none(parts[2]),
                    memory_used_mib=self._int_or_none(parts[3]),
                    memory_free_mib=self._int_or_none(parts[4]),
                    temperature_c=self._int_or_none(parts[5]),
                    utilization_percent=self._int_or_none(parts[6]),
                    driver_version=parts[7] or "unavailable",
                )
            )

        return gpus

    def _gpu_by_index(self, gpus: List[GPUInfo], index: str):
        for gpu in gpus:
            if str(gpu.index) == str(index):
                return gpu
        return None

    def _torch_cuda_state(self):
        try:
            import torch
        except ImportError:
            return False, "unavailable"

        try:
            if not torch.cuda.is_available():
                return False, "unavailable"
            return True, str(torch.cuda.current_device())
        except Exception:
            return False, "unavailable"

    def _int_or_none(self, value: str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
