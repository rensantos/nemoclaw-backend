from abc import ABC, abstractmethod
from typing import List, Optional


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
