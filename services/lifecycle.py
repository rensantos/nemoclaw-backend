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


class LifecycleUnavailableError(Exception):
    """Raised when an inference request arrives while the service is not
    ``ready`` (docs/model-lifecycle-design.md, Concurrency Model: requests
    during a transition are rejected, never queued).
    """

    def __init__(self, state: "LifecycleState"):
        self.state = state
        super().__init__(
            "Backend is not ready to serve inference (lifecycle state: "
            "{}); the request was rejected rather than queued.".format(state.value)
        )


class LifecycleConflictError(Exception):
    """Raised when a lifecycle operation is not legal from the current
    state - e.g. loading a different model while one is already ready.
    """


# Mirrors docs/model-lifecycle-design.md's State Transition Table exactly.
# Kept as data so an illegal transition fails loudly instead of silently
# corrupting state.
LEGAL_TRANSITIONS = {
    LifecycleState.UNLOADED: (LifecycleState.LOADING,),
    LifecycleState.LOADING: (LifecycleState.READY, LifecycleState.DEGRADED),
    LifecycleState.READY: (
        LifecycleState.UNLOADING,
        LifecycleState.SWITCHING,
        LifecycleState.DEGRADED,
    ),
    LifecycleState.UNLOADING: (LifecycleState.UNLOADED, LifecycleState.DEGRADED),
    LifecycleState.SWITCHING: (LifecycleState.READY, LifecycleState.DEGRADED),
    LifecycleState.DEGRADED: (LifecycleState.LOADING, LifecycleState.UNLOADED),
}


def validate_transition(from_state: LifecycleState, to_state: LifecycleState) -> None:
    if to_state not in LEGAL_TRANSITIONS[from_state]:
        raise LifecycleConflictError(
            "Illegal lifecycle transition {} -> {}".format(
                from_state.value, to_state.value
            )
        )


_HEALTH_STATUS_BY_LIFECYCLE_STATE = {
    LifecycleState.READY: "ok",
    LifecycleState.DEGRADED: "degraded",
    LifecycleState.UNLOADED: "unavailable",
    LifecycleState.LOADING: "unavailable",
    LifecycleState.UNLOADING: "unavailable",
    LifecycleState.SWITCHING: "unavailable",
}


def health_status_for_lifecycle_state(state: LifecycleState) -> str:
    """Projects HealthResponse.status from lifecycle_state.

    Matches the mapping pinned in openapi/backend-node.openapi.yaml's
    HealthResponse.status description exactly; status is not an
    independent truth source.
    """
    return _HEALTH_STATUS_BY_LIFECYCLE_STATE[state]
