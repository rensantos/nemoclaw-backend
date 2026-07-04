from enum import Enum


class LifecycleState(str, Enum):
    """Runtime lifecycle states owned by InferenceService.

    See docs/model-lifecycle-design.md for the full state machine and
    transition rules.
    """

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    SWITCHING = "switching"
    DEGRADED = "degraded"
