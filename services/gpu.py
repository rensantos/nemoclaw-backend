import subprocess
from dataclasses import dataclass, field
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
class GPUProcess:
    """A compute process holding memory on a GPU, as attributed by
    nvidia-smi --query-compute-apps."""

    gpu_index: str
    pid: int
    process_name: str
    memory_mib: Optional[int]


@dataclass
class GPUAvailability:
    """Box-wide GPU census. Deliberately says nothing about backend.gpu -
    this is "what is the state of this shared machine", not "is my
    configured GPU ok".

    ``ours`` is the important distinction: a GPU held by our own model
    runtime is *reclaimable* - we can evict and reload it. Lumping it in
    with another user's job would make the backend refuse to restart
    because of its own model.
    """

    in_use: List["GPUInfo"]
    free: List["GPUInfo"]
    ours: List["GPUInfo"] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.in_use) + len(self.free) + len(self.ours)

    def summary_line(self) -> str:
        if self.total == 0:
            return "No GPUs detected"
        parts = ["{} of {} GPU(s) in use by other processes".format(
            len(self.in_use), self.total
        )]
        if self.ours:
            parts.append("{} held by our own model (reclaimable)".format(len(self.ours)))
        parts.append("{} free".format(len(self.free)))
        return ", ".join(parts)

    @property
    def usable(self) -> List["GPUInfo"]:
        """GPUs we may place a model on: genuinely free, plus those only
        our own runtime is holding."""
        return self.free + self.ours


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

    def gpu_processes(self) -> List[GPUProcess]:
        """Per-process GPU memory attribution from nvidia-smi.

        --query-compute-apps reports gpu_uuid rather than index, so this
        maps uuid -> index itself. Returns an empty list when nvidia-smi
        is missing or reports nothing attributable; callers must treat
        that as "unknown", not "nobody is using the GPU".
        """
        rows = self._nvidia_smi(
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            expected_fields=4,
        )
        if not rows:
            return []

        index_by_uuid = {
            parts[1]: parts[0]
            for parts in self._nvidia_smi("--query-gpu=index,uuid", expected_fields=2)
        }

        processes = []
        for parts in rows:
            pid = self._int_or_none(parts[1])
            if pid is None:
                continue
            processes.append(
                GPUProcess(
                    gpu_index=index_by_uuid.get(parts[0], parts[0]),
                    pid=pid,
                    process_name=parts[2],
                    memory_mib=self._int_or_none(parts[3]),
                )
            )
        return processes

    def gpu_owner(
        self,
        gpu: GPUInfo,
        own_pids=None,
        processes: Optional[List[GPUProcess]] = None,
        threshold_mib: int = 500,
        utilization_threshold_percent: int = 10,
    ) -> str:
        """Classify a GPU as "free", "ours", or "others".

        "ours" means every attributable compute process on the card
        belongs to our own model runtime, so we may reclaim it - the model
        can be evicted and reloaded. Without this the backend would refuse
        to restart because of the model it is itself serving.

        Falls back to the memory/utilization heuristic when nvidia-smi
        attributes no process to a busy GPU: usage we cannot attribute is
        treated as somebody else's, never as ours.
        """
        own_pids = set(own_pids or ())
        if processes is None:
            processes = self.gpu_processes()

        on_this_gpu = [p for p in processes if str(p.gpu_index) == str(gpu.index)]
        if on_this_gpu:
            if all(p.pid in own_pids for p in on_this_gpu):
                return "ours"
            return "others"

        if self.gpu_is_in_use(gpu, threshold_mib, utilization_threshold_percent):
            return "others"
        return "free"

    def availability(
        self,
        threshold_mib: int = 500,
        utilization_threshold_percent: int = 10,
        own_pids=None,
    ) -> GPUAvailability:
        """Census of every GPU on the box: in use by others, held by our
        own runtime (reclaimable), or free.

        Checks all detected GPUs dynamically - it never assumes which
        indexes are "the busy ones" on this shared machine.
        """
        processes = self.gpu_processes()
        in_use, free, ours = [], [], []
        for gpu in self.detect_gpus():
            owner = self.gpu_owner(
                gpu, own_pids, processes, threshold_mib, utilization_threshold_percent
            )
            {"others": in_use, "ours": ours, "free": free}[owner].append(gpu)
        return GPUAvailability(in_use=in_use, free=free, ours=ours)

    def busy_gpus(
        self,
        threshold_mib: int = 500,
        utilization_threshold_percent: int = 10,
        own_pids=None,
    ) -> List[GPUInfo]:
        """Configured backend.gpu index(es) in use by *another* process.

        GPUs held only by our own runtime (own_pids) are excluded: that
        memory is reclaimable, and reporting it would mean warning about
        the model this backend is itself serving. backend.gpu may be a
        comma-separated multi-GPU value (e.g. "2,3"); each index is
        checked independently.
        """
        gpus = self.detect_gpus()
        processes = self.gpu_processes()
        indexes = [part.strip() for part in str(self.config.backend.gpu).split(",")]
        busy = []
        for index in indexes:
            gpu = self._gpu_by_index(gpus, index)
            if gpu and self.gpu_owner(
                gpu, own_pids, processes, threshold_mib, utilization_threshold_percent
            ) == "others":
                busy.append(gpu)
        return busy

    def idle_alternative_gpus(
        self,
        threshold_mib: int = 500,
        utilization_threshold_percent: int = 10,
        own_pids=None,
    ) -> List[GPUInfo]:
        """GPUs on this box NOT among backend.gpu's configured index(es)
        that we could use instead - free, or held only by our own runtime
        and therefore reclaimable. Pairs with busy_gpus() to decide
        whether "no other way" applies.
        """
        configured = {part.strip() for part in str(self.config.backend.gpu).split(",")}
        processes = self.gpu_processes()
        return [
            gpu
            for gpu in self.detect_gpus()
            if str(gpu.index) not in configured
            and self.gpu_owner(
                gpu, own_pids, processes, threshold_mib, utilization_threshold_percent
            ) != "others"
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
        own_pids=None,
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
        processes = self.gpu_processes()
        return [
            gpu
            for gpu in self.detect_gpus()
            if str(gpu.index) in visible_set
            and self.gpu_owner(
                gpu, own_pids, processes, threshold_mib, utilization_threshold_percent
            ) == "others"
        ]

    def driver_version(self) -> str:
        gpus = self.detect_gpus()
        if not gpus:
            return "unavailable"
        return gpus[0].driver_version

    def _nvidia_smi(self, query: str, expected_fields: int) -> List[List[str]]:
        """Runs an nvidia-smi CSV query, returning parsed rows. An
        unavailable nvidia-smi yields no rows rather than raising."""
        try:
            result = subprocess.run(
                ["nvidia-smi", query, "--format=csv,noheader,nounits"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return []

        rows = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= expected_fields:
                rows.append(parts)
        return rows

    def _detect_with_nvidia_smi(self) -> List[GPUInfo]:
        query = (
            "index,name,memory.total,memory.used,memory.free,"
            "temperature.gpu,utilization.gpu,driver_version"
        )

        gpus = []
        for parts in self._nvidia_smi("--query-gpu={}".format(query), expected_fields=8):
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
