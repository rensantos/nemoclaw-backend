import time
import uuid

from fastapi import APIRouter, HTTPException

from config import settings
from model_runtime import generate_chat, generate_text, health, list_models
from schemas import ChatCompletionRequest, GenerateRequest


router = APIRouter()


@router.get("/health")
def health_check():
    return health()


@router.get("/v1/models")
def models():
    return list_models()


@router.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    model_id = req.model or settings.model_id
    result = generate_chat(req.messages, req.max_tokens, req.temperature)

    return {
        "id": "chatcmpl-{}".format(uuid.uuid4().hex),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["content"],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        },
    }


@router.post("/generate")
def generate(req: GenerateRequest):
    return generate_text(req.prompt, req.max_new_tokens, req.temperature)
