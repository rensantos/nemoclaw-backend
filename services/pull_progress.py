"""Live progress for an in-flight model download.

POST /admin/model/pull answers only when the transfer is over, which for a
9GB model is the better part of an hour. That left every caller unable to
tell a working download from a hung one - observed live: the Telegram bot
driving this node sat silent for most of an hour on deepseek-r1:14b,
holding one request that would not return, while the daemon knew the byte
count the whole time.

Ollama streams that byte count. This records the latest reading so
GET /admin/model/pull/status can answer "how far along" without touching
the transfer itself.

Single-slot and in-memory on purpose: a node pulls one model at a time
(the disk gates in services/inference.py are written assuming that), and
progress that survived a restart would describe a download that died with
the process.
"""

import threading
import time


class PullProgress:
    """Thread-safe record of the current or most recent download.

    Written by the thread running the pull and read by whichever thread
    serves the status request - FastAPI runs sync endpoints in a worker
    threadpool, so those are genuinely concurrent, and a blocked pull does
    not stop the status call.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._reset(None)

    def _reset(self, model_id):
        self.model_id = model_id
        self.active = model_id is not None
        self.status = "starting" if model_id else "idle"
        self.started_at = time.time() if model_id else None
        self.updated_at = self.started_at
        self.finished_at = None
        self.error = None
        # {digest: {"completed": int, "total": int}} - kept per layer
        # rather than as running sums because the daemon reports each
        # layer's cumulative byte count, so adding what arrives would
        # count the same bytes again on every event.
        self._layers = {}

    def start(self, model_id):
        with self._lock:
            self._reset(model_id)

    def record(self, event):
        """Fold one Ollama pull event into the running totals.

        Unknown or byte-less events (`pulling manifest`, `verifying sha256
        digest`) still move `status`: during those phases there is nothing
        to count, and a caller showing only bytes would look stalled.
        """
        if not isinstance(event, dict):
            return
        with self._lock:
            status = event.get("status")
            if status:
                self.status = status
            digest = event.get("digest")
            total = event.get("total")
            if digest and total:
                layer = self._layers.setdefault(digest, {"completed": 0, "total": 0})
                layer["total"] = int(total)
                completed = event.get("completed")
                if completed is not None:
                    layer["completed"] = int(completed)
            self.updated_at = time.time()

    def finish(self, error=None):
        with self._lock:
            self.active = False
            self.error = None if error is None else str(error)
            self.finished_at = time.time()
            self.updated_at = self.finished_at
            if error is None:
                # A completed download is complete by definition: the last
                # event before "success" is routinely a few KiB short, and
                # a final reading of 99% reads as a stall.
                for layer in self._layers.values():
                    layer["completed"] = layer["total"]
                self.status = "success"

    def snapshot(self):
        """A plain dict for the status endpoint.

        `total_bytes` covers the layers the daemon has announced so far,
        not the model's eventual size - it is not knowable up front (see
        docs/model-pull-design.md Section 3), so on a multi-blob model the
        total can grow and the percentage step back mid-download. Reported
        as measured rather than smoothed into a guess.
        """
        with self._lock:
            completed = sum(layer["completed"] for layer in self._layers.values())
            total = sum(layer["total"] for layer in self._layers.values())
            return {
                "active": self.active,
                "model_id": self.model_id,
                "status": self.status,
                "completed_bytes": completed,
                "total_bytes": total,
                "percent": round(completed * 100.0 / total, 1) if total else None,
                "layers": len(self._layers),
                "started_at": self.started_at,
                "updated_at": self.updated_at,
                "finished_at": self.finished_at,
                "error": self.error,
            }
