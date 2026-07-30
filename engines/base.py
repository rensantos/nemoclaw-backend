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


class InferenceEngine(ABC):
    """Minimal interface required by the Nemoclaw Backend API."""

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def unload_model(self) -> None:
        pass

    @abstractmethod
    def health(self):
        pass

    @abstractmethod
    def list_models(self):
        pass

    @abstractmethod
    def chat(self, messages: List, max_tokens: Optional[int], temperature: Optional[float]):
        pass

    @abstractmethod
    def generate_text(self, prompt: str, max_new_tokens: int, temperature: float):
        pass
