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


class PullNotSupportedError(Exception):
    """Raised when a download is requested from an engine that cannot."""

    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        super().__init__(
            "{} cannot download models. Only Ollama-backed nodes support "
            "this; for a Transformers node, fetch the model into the "
            "Hugging Face cache and add it to config.yaml's "
            "model.available.".format(engine_name)
        )


class EmbeddingsNotSupportedError(Exception):
    """Raised when embeddings are requested from an engine that cannot."""

    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        super().__init__(
            "{} cannot produce embeddings. Only Ollama-backed nodes support "
            "this, because an embedding model has to be served alongside "
            "the loaded chat model rather than replacing it.".format(engine_name)
        )


class InsufficientDiskError(Exception):
    """Raised when a download would not fit, or would leave too little.

    Deliberately an error rather than a warning: everywhere else a
    heuristic in this backend only warns, because being wrong costs us a
    failed request. Filling a shared machine's disk costs other people
    their work (see docs/model-pull-design.md Section 3).
    """

    def __init__(self, message: str, required_mib=None, free_mib=None):
        self.required_mib = required_mib
        self.free_mib = free_mib
        super().__init__(message)


class StreamingNotSupportedError(Exception):
    """Raised when a streaming request reaches an engine that cannot
    stream. Distinct from a lifecycle problem: the backend is healthy, the
    active engine simply has no incremental path (TransformersEngine
    generates a whole response in one call)."""

    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        super().__init__(
            "Streaming is not supported by {}. Send the request without "
            '"stream": true, or run an engine that supports it.'.format(engine_name)
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
