from typing import List, Optional

from pydantic import BaseModel

from config import settings


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    # OllamaEngine-only (engines/ollama_engine.py); TransformersEngine
    # ignores it, there's no equivalent concept. None means "use
    # model.think_default from config", which itself defaults to None
    # ("don't send think at all, let Ollama use its own default").
    think: Optional[bool] = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = settings.max_tokens_default
    temperature: float = settings.temperature_default
    think: Optional[bool] = None


class ModelLifecycleRequest(BaseModel):
    model_id: str
    # Runtime-only by default. True also rewrites config.yaml's model.id via
    # ModelManager, so a restart keeps the new model.
    persist: bool = False
