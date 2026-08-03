"""Host resource inspection: disk and system RAM.

GPUManager owns GPU discovery and VRAM. Nothing owned disk or system RAM,
and both decide whether a model can actually be served here:

- **Disk** decides whether a model can be *downloaded at all*. On a shared
  machine this is the constraint that can hurt other people, not just this
  project - UBI's single volume is ~99% full with other researchers' data.
- **System RAM** decides more than it looks. Ollama sizes its context
  against system RAM, not combined VRAM, so a box with plenty of VRAM can
  still refuse a large context (observed live on UBI, recorded in the
  frontend's NODE_MODEL_NUM_CTX_CEILINGS).

Every reading is optional. A value that cannot be determined is reported
as None, never as zero: "unknown free space" and "no free space" must not
look the same to a caller deciding whether a download is safe.
"""

import os
import shutil
from dataclasses import dataclass
from typing import List, Optional

MIB = 1024 * 1024


@dataclass(frozen=True)
class DiskInfo:
    path: str
    total_mib: Optional[int]
    free_mib: Optional[int]

    @property
    def used_percent(self) -> Optional[float]:
        if not self.total_mib or self.free_mib is None:
            return None
        return round((self.total_mib - self.free_mib) / self.total_mib * 100, 1)


@dataclass(frozen=True)
class MemoryInfo:
    total_mib: Optional[int]
    available_mib: Optional[int]


@dataclass(frozen=True)
class HostResources:
    disk: Optional[DiskInfo]
    memory: Optional[MemoryInfo]
    gpus: List[dict]

    @property
    def total_vram_mib(self) -> Optional[int]:
        values = [gpu.get("memory_total_mib") for gpu in self.gpus]
        known = [value for value in values if value]
        return sum(known) if known else None

    @property
    def free_vram_mib(self) -> Optional[int]:
        values = [gpu.get("memory_free_mib") for gpu in self.gpus]
        known = [value for value in values if value is not None]
        return sum(known) if known else None


class HostResourceService:
    """Reports what this machine physically has available."""

    def __init__(self, gpu_manager=None, engine=None):
        self.gpu_manager = gpu_manager
        self.engine = engine

    def disk(self, path: Optional[str] = None) -> Optional[DiskInfo]:
        """Free space on the filesystem holding `path`.

        Defaults to where the engine stores models, so the number answers
        the question actually being asked ("can I download this?") rather
        than reporting some unrelated mount.
        """
        target = path or self.model_storage_path() or os.path.expanduser("~")
        try:
            usage = shutil.disk_usage(target)
        except OSError:
            return None
        return DiskInfo(
            path=target,
            total_mib=usage.total // MIB,
            free_mib=usage.free // MIB,
        )

    def model_storage_path(self) -> Optional[str]:
        if self.engine is None:
            return None
        getter = getattr(self.engine, "model_storage_path", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    def memory(self) -> Optional[MemoryInfo]:
        """System RAM from /proc/meminfo.

        MemAvailable is the kernel's own estimate of what a new workload
        could actually get; it is much closer to the truth than MemFree,
        which ignores reclaimable cache.
        """
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                fields = {}
                for line in handle:
                    key, _, rest = line.partition(":")
                    parts = rest.split()
                    if parts and parts[0].isdigit():
                        fields[key.strip()] = int(parts[0])  # kB
        except OSError:
            return None
        total = fields.get("MemTotal")
        available = fields.get("MemAvailable")
        return MemoryInfo(
            total_mib=total // 1024 if total else None,
            available_mib=available // 1024 if available else None,
        )

    def gpus(self) -> List[dict]:
        if self.gpu_manager is None:
            return []
        try:
            detected = self.gpu_manager.detect_gpus()
        except Exception:
            return []
        return [
            {
                "index": gpu.index,
                "name": gpu.name,
                "memory_total_mib": gpu.memory_total_mib,
                "memory_used_mib": gpu.memory_used_mib,
                "memory_free_mib": gpu.memory_free_mib,
            }
            for gpu in detected
        ]

    def snapshot(self) -> HostResources:
        return HostResources(disk=self.disk(), memory=self.memory(), gpus=self.gpus())
