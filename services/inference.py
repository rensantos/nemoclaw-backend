from config import settings
from engines.base import InferenceEngine
from services.lifecycle import LifecycleState, lifecycle_not_implemented_response


class InferenceService:
    """Application service that owns runtime lifecycle state and delegates
    inference work to an engine."""

    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self.engine.load_model()
        self.lifecycle_state = LifecycleState.READY

    def health(self):
        health = dict(self.engine.health())
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

    def chat(self, messages, max_tokens, temperature):
        return self.engine.chat(messages, max_tokens, temperature)

    def generate_text(self, prompt, max_new_tokens, temperature):
        return self.engine.generate_text(prompt, max_new_tokens, temperature)


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
    return InferenceService(_build_engine(settings))
