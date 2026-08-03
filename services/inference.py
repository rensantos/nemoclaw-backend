import contextlib
import logging
import threading
import time
from typing import Optional

from config import settings
from engines.base import (
    EngineUnavailableError,
    InferenceEngine,
    LifecycleNotSupportedError,
    ModelUnavailableError,
)
from services.gpu import GPUManager
from services.lifecycle import (
    LifecycleConflictError,
    LifecycleState,
    LifecycleUnavailableError,
    health_status_for_lifecycle_state,
    validate_transition,
)
from services.model import ModelManager

_logger = logging.getLogger(__name__)

DEFAULT_DRAIN_TIMEOUT_SECONDS = 120


class InferenceService:
    """Application service that owns runtime lifecycle state and delegates
    inference work to an engine."""

    def __init__(
        self,
        engine: InferenceEngine,
        gpu_manager: Optional[GPUManager] = None,
        model_manager: Optional[ModelManager] = None,
    ):
        self.engine = engine
        self.gpu_manager = gpu_manager
        self.model_manager = model_manager
        self.target_model_id = None
        self.transition_started_at = None
        self._active_requests = 0
        self._requests = threading.Condition()
        self._transition_lock = threading.Lock()
        self._warn_if_gpu_busy()
        self.engine.load_model()
        self.lifecycle_state = LifecycleState.READY
        self.loaded_model_id = getattr(engine, "model_id", None)

    def _warn_if_gpu_busy(self):
        """Log a warning for any configured GPU that already shows
        significant memory usage before this engine has loaded anything -
        on UBI's shared box, that memory can only belong to another
        process (docs/problems.md's GPU 0/1 note). No gpu_manager means no
        config was available to check against (e.g. some direct-engine
        test construction) - silently skipped, not an error.
        """
        if self.gpu_manager is None:
            return
        # Exclude our own model runtime: on a restart the previous model
        # may still be resident, and warning about it would be warning
        # about ourselves.
        own_pids = self.engine.runtime_pids()
        for gpu in self.gpu_manager.busy_gpus(own_pids=own_pids):
            _logger.warning(
                "GPU %s ('%s') already has %sMiB/%sMiB used before this "
                "backend has loaded a model - likely another process's "
                "job on this shared box, not this backend's own usage",
                gpu.index,
                gpu.name,
                gpu.memory_used_mib,
                gpu.memory_total_mib,
            )

    def health(self):
        try:
            health = dict(self.engine.health())
        except EngineUnavailableError as exc:
            self.lifecycle_state = LifecycleState.DEGRADED
            health = dict(exc.partial_health)
            health.setdefault("model", getattr(self.engine, "model_id", None))
            health.setdefault("cuda", False)
            health.setdefault("gpu", None)

        health["status"] = health_status_for_lifecycle_state(self.lifecycle_state)
        health["lifecycle_state"] = self.lifecycle_state.value
        health["loaded_model"] = self.loaded_model_id
        health["target_model"] = self.target_model_id
        return health

    def list_models(self):
        """Every model a caller may select, with what is known about each.

        Joins ModelManager's configured catalog (which models are allowed)
        with the engine's runtime facts (which are actually downloaded and
        fit the available GPUs), so a frontend can build a model picker
        and grey out what is not usable rather than offering choices that
        fail on selection.

        Falls back to the engine's own listing when no ModelManager was
        supplied, which is only the case in direct-engine test
        construction.
        """
        if self.model_manager is None:
            return self.engine.list_models()

        entries = [
            model
            for model in self.model_manager.list_models()
            if self._servable_by_active_engine(model)
        ]
        runtime_info = self._model_runtime_info([str(m["id"]) for m in entries])

        created = int(time.time())
        return {
            "object": "list",
            "data": [
                self._model_object(str(model["id"]), created, runtime_info)
                for model in entries
            ],
        }

    def _servable_by_active_engine(self, model) -> bool:
        """Catalog entries carry the engine they belong to; listing
        Transformers repos while running Ollama would offer choices this
        instance cannot serve at all."""
        entry_engine = model.get("engine")
        return not entry_engine or entry_engine == settings.backend.engine

    def _model_runtime_info(self, model_ids):
        try:
            return self.engine.model_runtime_info(model_ids)
        except EngineUnavailableError:
            raise
        except Exception:
            # Unknown beats wrong: the catalog is still worth returning.
            return {}

    def _model_object(self, model_id, created, runtime_info):
        model_object = {
            "id": model_id,
            "object": "model",
            "created": created,
            "owned_by": settings.backend.engine,
            "loaded": model_id == self.loaded_model_id,
        }
        model_object.update(runtime_info.get(model_id, {}))
        return model_object

    def chat(self, messages, max_tokens, temperature, requested_model=None, think=None):
        with self._serving():
            try:
                return self.engine.chat(
                    messages, max_tokens, temperature, requested_model, think
                )
            except EngineUnavailableError:
                self.lifecycle_state = LifecycleState.DEGRADED
                raise

    def generate_text(self, prompt, max_new_tokens, temperature, think=None):
        with self._serving():
            try:
                return self.engine.generate_text(
                    prompt, max_new_tokens, temperature, think
                )
            except EngineUnavailableError:
                self.lifecycle_state = LifecycleState.DEGRADED
                raise

    # ---- lifecycle operations (docs/model-lifecycle-design.md) ----

    def load_model(self, model_id, persist=False, timeout=DEFAULT_DRAIN_TIMEOUT_SECONDS):
        """Loads model_id when nothing is loaded.

        Idempotent when the same model is already ready; loading a
        *different* model while one is ready is a conflict that directs the
        operator to switch instead.
        """
        with self._transition_lock:
            self._require_lifecycle_support()
            self._validate_model_id(model_id)

            if self.lifecycle_state == LifecycleState.READY:
                if model_id == self.loaded_model_id:
                    return self._result(model_id, model_id, 0.0, persisted=False)
                raise LifecycleConflictError(
                    "Model '{}' is already loaded. Use 'model switch {}' to "
                    "change the running model.".format(self.loaded_model_id, model_id)
                )

            return self._transition(
                LifecycleState.LOADING,
                LifecycleState.READY,
                model_id,
                lambda: self.engine.load_model(model_id),
                persist,
                timeout,
                warning=self._vram_warning(model_id),
            )

    def unload_model(self, timeout=DEFAULT_DRAIN_TIMEOUT_SECONDS):
        """Releases the loaded model. Idempotent when already unloaded."""
        with self._transition_lock:
            self._require_lifecycle_support()

            if self.lifecycle_state == LifecycleState.UNLOADED:
                return self._result(None, None, 0.0, persisted=False)

            if self.lifecycle_state == LifecycleState.DEGRADED:
                # The table's recovery edge is degraded -> unloaded directly;
                # there is no degraded -> unloading. Nothing is known to be
                # serving, so there is nothing to drain and the engine call
                # is best-effort.
                return self._give_up_and_unload()

            return self._transition(
                LifecycleState.UNLOADING,
                LifecycleState.UNLOADED,
                None,
                self.engine.unload_model,
                persist=False,
                timeout=timeout,
            )

    def switch_model(
        self, model_id, persist=False, timeout=DEFAULT_DRAIN_TIMEOUT_SECONDS
    ):
        """Moves from one ready model to another in a single transition."""
        with self._transition_lock:
            self._require_lifecycle_support()
            self._validate_model_id(model_id)

            if self.lifecycle_state != LifecycleState.READY:
                raise LifecycleConflictError(
                    "Cannot switch from lifecycle state '{}'; no model is "
                    "currently serving. Use 'model load {}' "
                    "instead.".format(self.lifecycle_state.value, model_id)
                )

            return self._transition(
                LifecycleState.SWITCHING,
                LifecycleState.READY,
                model_id,
                lambda: self.engine.switch_model(model_id),
                persist,
                timeout,
                warning=self._vram_warning(model_id),
            )

    def _give_up_and_unload(self):
        previous_model = self.loaded_model_id
        try:
            self.engine.unload_model()
        except Exception as exc:
            _logger.warning(
                "Best-effort unload from degraded state failed, reporting "
                "unloaded anyway: %s", exc
            )

        validate_transition(self.lifecycle_state, LifecycleState.UNLOADED)
        self.lifecycle_state = LifecycleState.UNLOADED
        self.loaded_model_id = None
        self.target_model_id = None
        self.transition_started_at = None
        return self._result(None, previous_model, 0.0, persisted=False)

    def _vram_warning(self, model_id):
        """Engine-supplied caution (e.g. the model is bigger than the GPUs
        the runtime can reach). Advisory only - never fails the call.

        Computed here rather than inside the engine's switch/load so the
        estimate is made once and a failing check can never break a
        transition that would otherwise succeed.
        """
        try:
            warning = self.engine.vram_warning_for(model_id)
        except Exception:
            return None
        if warning:
            _logger.warning("%s", warning)
        return warning

    def _transition(
        self,
        transitional_state,
        final_state,
        model_id,
        operation,
        persist,
        timeout,
        warning=None,
    ):
        previous_model = self.loaded_model_id
        previous_state = self.lifecycle_state
        started_at = time.monotonic()

        validate_transition(self.lifecycle_state, transitional_state)
        self.lifecycle_state = transitional_state
        self.target_model_id = model_id
        self.transition_started_at = started_at

        # New requests are already rejected by _serving() now that state is
        # not READY; this waits out the ones that were already in flight.
        self._drain(timeout)

        try:
            operation()
        except ModelUnavailableError:
            # Pre-flight failure: the engine rejected the target before
            # touching the loaded model, so the previous one is still
            # intact and serving. Reporting degraded here would take a
            # healthy backend offline over a bad request.
            self.lifecycle_state = previous_state
            self.target_model_id = None
            self.transition_started_at = None
            raise
        except Exception:
            validate_transition(self.lifecycle_state, LifecycleState.DEGRADED)
            self.lifecycle_state = LifecycleState.DEGRADED
            self.target_model_id = None
            self.transition_started_at = None
            raise

        validate_transition(self.lifecycle_state, final_state)
        self.lifecycle_state = final_state
        self.loaded_model_id = model_id
        self.target_model_id = None
        self.transition_started_at = None

        persisted = False
        if persist and model_id is not None:
            self.model_manager.select_model(model_id)
            persisted = True

        return self._result(
            model_id,
            previous_model,
            time.monotonic() - started_at,
            persisted,
            warning,
        )

    def _result(self, loaded_model, previous_model, elapsed, persisted, warning=None):
        result = {
            "status": "ok",
            "lifecycle_state": self.lifecycle_state.value,
            "loaded_model": loaded_model,
            "previous_model": previous_model,
            "elapsed_seconds": round(elapsed, 3),
            "persisted": persisted,
        }
        if warning:
            result["warning"] = warning
        return result

    def _require_lifecycle_support(self):
        if not self.engine.supports_runtime_lifecycle:
            raise LifecycleNotSupportedError(type(self.engine).__name__)
        if self.model_manager is None:
            raise LifecycleConflictError(
                "This InferenceService was constructed without a ModelManager, "
                "so a lifecycle target cannot be validated against the "
                "configured model catalog."
            )

    def _validate_model_id(self, model_id):
        """Rejects anything outside config.yaml's model.available before any
        runtime change happens (design doc's Failure Modes: "model id is not
        configured: reject before any runtime change"). Raises ValueError.
        """
        self.model_manager.validate_model(model_id)

    @contextlib.contextmanager
    def _serving(self):
        with self._requests:
            if self.lifecycle_state != LifecycleState.READY:
                raise LifecycleUnavailableError(self.lifecycle_state)
            self._active_requests += 1
        try:
            yield
        finally:
            with self._requests:
                self._active_requests -= 1
                self._requests.notify_all()

    def _drain(self, timeout):
        """Waits for in-flight requests to finish. Past the timeout the
        transition proceeds anyway (design doc: "stop accepting more work
        and restart the worker anyway"), leaving a warning behind rather
        than blocking an operator's unload indefinitely.
        """
        deadline = time.monotonic() + timeout
        with self._requests:
            while self._active_requests > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _logger.warning(
                        "Drain timed out after %ss with %s request(s) still "
                        "in flight; proceeding with the transition anyway",
                        timeout,
                        self._active_requests,
                    )
                    return
                self._requests.wait(remaining)


def _build_engine(config):
    """Construct the InferenceEngine selected by config.backend.engine.

    config.py's load_config() already validates backend.engine against
    VALID_ENGINES at startup, so only "transformers" or "ollama" are ever
    seen here in practice; the final branch is a fail-fast guard for
    direct callers, not a silent fallback.
    """
    engine_name = config.backend.engine

    if engine_name == "transformers":
        from engines.transformers_engine import TransformersEngine

        return TransformersEngine(config)

    if engine_name == "ollama":
        from engines.ollama_engine import OllamaEngine

        return OllamaEngine(config)

    raise ValueError(
        "Unknown backend.engine '{}'; valid values: transformers, ollama".format(
            engine_name
        )
    )


def create_inference_service():
    return InferenceService(
        _build_engine(settings), GPUManager(settings), ModelManager()
    )
