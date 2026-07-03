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
