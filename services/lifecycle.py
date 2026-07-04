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


LIFECYCLE_NOT_IMPLEMENTED_DETAIL = "Model lifecycle operations are not implemented yet."


def lifecycle_not_implemented_response(lifecycle_state: LifecycleState) -> dict:
    """Fixed stub body for /admin/model/* endpoints (Phase 5 Increment 2).

    Returns the given lifecycle_state without changing it; callers must not
    use this to perform any lifecycle transition.
    """
    return {
        "error": "not_implemented",
        "detail": LIFECYCLE_NOT_IMPLEMENTED_DETAIL,
        "lifecycle_state": lifecycle_state.value,
    }
