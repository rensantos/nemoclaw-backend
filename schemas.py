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


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = settings.max_tokens_default
    temperature: float = settings.temperature_default


class ModelLifecycleRequest(BaseModel):
    model_id: str
