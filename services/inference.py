from config import settings
from engines.base import InferenceEngine


class InferenceService:
    """Application service that delegates inference work to an engine."""

    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self.engine.load_model()

    def health(self):
        return self.engine.health()

    def list_models(self):
        return self.engine.list_models()

    def chat(self, messages, max_tokens, temperature):
        return self.engine.chat(messages, max_tokens, temperature)

    def generate_text(self, prompt, max_new_tokens, temperature):
        return self.engine.generate_text(prompt, max_new_tokens, temperature)


def create_inference_service():
    from engines.transformers_engine import TransformersEngine

    return InferenceService(TransformersEngine(settings))
