from abc import ABC, abstractmethod
from typing import List, Optional


class EngineUnavailableError(Exception):
    """Raised when an engine's backing runtime is unreachable (e.g. an
    external daemon is down), as opposed to a request-level error.

    ``partial_health`` may carry any engine fields the caller was able to
    determine before the unreachable condition was hit (e.g. host-level GPU
    presence, which does not depend on the daemon), so callers that need to
    keep responding (like a health check) don't have to fully re-derive
    them.
    """

    def __init__(self, message: str, partial_health: Optional[dict] = None):
        super().__init__(message)
        self.partial_health = partial_health or {}


class ModelNotFoundError(Exception):
    """Raised when a request names a model this engine instance cannot
    serve (docs/ollama-engine-design.md Section 1's reject-on-mismatch
    decision). Not raised by every engine: TransformersEngine keeps its
    existing echo-and-serve quirk, out of scope for this decision.
    """

    def __init__(self, requested_model: str, servable_model: str):
        self.requested_model = requested_model
        self.servable_model = servable_model
        super().__init__(
            "Model '{}' does not exist or is not currently loaded by this "
            "backend instance (servable model: '{}')".format(
                requested_model, servable_model
            )
        )


class ModelUnavailableError(RuntimeError):
    """Raised by an engine's pre-flight check when a lifecycle target
    cannot be used and **nothing has been changed yet**.

    The distinction matters: a pre-flight failure means the previously
    loaded model is still intact and serving, so the service restores the
    prior lifecycle state instead of reporting ``degraded``. Failures
    raised after the engine has begun releasing or swapping state are
    ordinary exceptions and do lead to ``degraded``.
    """


class LifecycleNotSupportedError(Exception):
    """Raised when a runtime lifecycle operation is requested against an
    engine that cannot perform it safely.

    TransformersEngine owns its own Python/CUDA state in-process, where
    ``del model`` + ``empty_cache()`` is best-effort only and can leave
    allocator fragmentation behind (docs/model-lifecycle-design.md,
    "CUDA Cleanup"). Refusing is the honest answer until worker
    supervision exists; the alternative would be a swap that appears to
    succeed and silently poisons later loads.
    """

    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        super().__init__(
            "Runtime model lifecycle is not supported by {}. Its CUDA state "
            "is owned in-process, where cleanup is best-effort only; a "
            "supervised worker process is required for a reliable "
            "load/unload/switch boundary (see "
            "docs/model-lifecycle-design.md). Change model.id in "
            "config/config.yaml and restart the backend "
            "instead.".format(engine_name)
        )


class InferenceEngine(ABC):
    """Minimal interface required by the Nemoclaw Backend API."""

    # Whether load/unload/switch can be performed safely against a running
    # process. The engine declares the capability; InferenceService decides
    # what to do about it, keeping lifecycle policy in the service layer.
    supports_runtime_lifecycle = False

    # Whether this engine can download a model on request. False by
    # default so a new engine cannot silently appear to support it; the
    # service raises PullNotSupportedError (501) instead.
    supports_pull = False

    supports_streaming = False

    # False by default so an engine that cannot embed reports a capability
    # rather than crashing, the same posture as supports_pull/streaming
    # above. The service turns this into a 501 rather than a 500.
    supports_embeddings = False

    def chat_stream(
        self,
        messages: List,
        max_tokens: Optional[int],
        temperature: Optional[float],
        requested_model: Optional[str] = None,
        think: Optional[bool] = None,
    ):
        """Yields incremental chat deltas.

        Each item is a dict with any of "content", "reasoning" (text
        fragments) and "usage" (final token counts, on the last item).
        Only engines advertising ``supports_streaming`` implement this.
        """
        raise NotImplementedError

    def installed_models(self) -> List[str]:
        """Model ids physically present on this runtime right now.

        Distinct from the configured catalog, which is an allowlist and
        can disagree with reality in both directions: it may name models
        that were never downloaded here, and miss models someone
        downloaded directly (`ollama pull`) without going through
        /admin/model/pull. Both were observed live on the same node - six
        catalogued-but-absent tags listed as choices, while an actually
        usable one was invisible.

        Empty means "cannot enumerate", not "nothing installed", so
        callers must not treat it as an authoritative absence.
        """
        return []

    def model_runtime_info(self, model_ids: List[str]) -> dict:
        """Runtime facts about configured models, keyed by model id.

        The catalog of *selectable* models is configuration, owned by
        ModelManager. Whether each one is actually usable right now -
        downloaded, and small enough for the GPUs this runtime can reach -
        is a runtime fact only the engine knows. InferenceService joins
        the two for /v1/models.

        Each value may carry "pulled" (bool), "size_mib" (int) and "fits"
        (bool). Omit any key that cannot be determined rather than
        guessing; the default is to know nothing.
        """
        return {}

    def vram_warning_for(self, model_id: str) -> Optional[str]:
        """A human-readable caution about serving model_id, or None.

        Lets an engine surface a fit problem the service layer cannot see
        (e.g. the model is larger than the GPUs the runtime can reach)
        without failing the operation, since such checks are estimates.
        """
        return None

    def model_storage_path(self) -> Optional[str]:
        """Where this engine stores model weights on disk.

        Lets resource reporting answer "is there room to download this?"
        against the filesystem that would actually receive the download.
        None means unknown, which callers must not treat as "anywhere".
        """
        return None

    def runtime_pids(self) -> List[int]:
        """PIDs that may hold GPU memory on this engine's behalf.

        Lets GPU checks tell our own model's VRAM apart from another
        user's job, so the backend never refuses to start because of the
        model it is itself serving. Empty means "no separate runtime
        process to attribute" - in-process engines hold memory under the
        backend's own PID.
        """
        return []

    @abstractmethod
    def load_model(self, model_id: Optional[str] = None) -> None:
        """Loads model_id, or the configured default when None."""

    @abstractmethod
    def unload_model(self) -> None:
        pass

    def switch_model(self, model_id: str) -> None:
        """Transitions from the currently loaded model to model_id.

        Default is unload-then-load; engines that can verify the target
        before giving up the current model should override this so a bad
        target leaves the old model serving.
        """
        self.unload_model()
        self.load_model(model_id)

    @abstractmethod
    def health(self):
        pass

    @abstractmethod
    def list_models(self):
        pass

    @abstractmethod
    def chat(
        self,
        messages: List,
        max_tokens: Optional[int],
        temperature: Optional[float],
        requested_model: Optional[str] = None,
        think: Optional[bool] = None,
    ):
        pass

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        think: Optional[bool] = None,
    ):
        pass

    def embed(self, texts: List[str], model: str) -> List[List[float]]:
        """Embedding vectors for texts, one per input, in order.

        `model` is required and is deliberately NOT the loaded chat model:
        embedding models are separate models (nomic-embed-text and
        friends), so this must not go through the requested-model check
        that chat() uses, and must not trigger a model switch. Engines
        that can serve several models at once (Ollama) satisfy this
        naturally; engines that hold exactly one model in memory should
        keep returning False from supports_embeddings rather than
        unloading the chat model to answer an embedding request.

        Raises NotImplementedError by default - concrete, not abstract, so
        adding this capability does not break existing engines.
        """
        raise NotImplementedError(
            "{} cannot produce embeddings".format(type(self).__name__)
        )
