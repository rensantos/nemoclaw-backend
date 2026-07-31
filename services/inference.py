import logging
from typing import Optional

from config import settings
from engines.base import EngineUnavailableError, InferenceEngine
from services.gpu import GPUManager
from services.lifecycle import (
    LifecycleState,
    health_status_for_lifecycle_state,
    lifecycle_not_implemented_response,
)

_logger = logging.getLogger(__name__)


class InferenceService:
    """Application service that owns runtime lifecycle state and delegates
    inference work to an engine."""

    def __init__(self, engine: InferenceEngine, gpu_manager: Optional[GPUManager] = None):
        self.engine = engine
        self.gpu_manager = gpu_manager
        self._warn_if_gpu_busy()
        self.engine.load_model()
        self.lifecycle_state = LifecycleState.READY

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
        for gpu in self.gpu_manager.busy_gpus():
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
        return health

    def lifecycle_stub_response(self):
        """Fixed not-implemented body for /admin/model/* stub endpoints.

        Does not change lifecycle_state; load/unload/switch are not
        implemented yet (Phase 5 Increment 2).
        """
        return lifecycle_not_implemented_response(self.lifecycle_state)

    def list_models(self):
        return self.engine.list_models()

    def chat(self, messages, max_tokens, temperature, requested_model=None, think=None):
        try:
            return self.engine.chat(
                messages, max_tokens, temperature, requested_model, think
            )
        except EngineUnavailableError:
            self.lifecycle_state = LifecycleState.DEGRADED
            raise

    def generate_text(self, prompt, max_new_tokens, temperature, think=None):
        try:
            return self.engine.generate_text(prompt, max_new_tokens, temperature, think)
        except EngineUnavailableError:
            self.lifecycle_state = LifecycleState.DEGRADED
            raise


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
    return InferenceService(_build_engine(settings), GPUManager(settings))
