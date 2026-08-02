import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from config import settings
from engines.base import (
    EngineUnavailableError,
    LifecycleNotSupportedError,
    ModelNotFoundError,
    ModelUnavailableError,
)
from services.inference import create_inference_service
from services.lifecycle import LifecycleConflictError, LifecycleUnavailableError
from schemas import ChatCompletionRequest, GenerateRequest, ModelLifecycleRequest


router = APIRouter()
inference_service = create_inference_service()


@router.get("/health")
def health_check():
    return inference_service.health()


@router.get("/v1/models")
def models():
    return inference_service.list_models()


@router.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    model_id = req.model or settings.model_id
    try:
        result = inference_service.chat(
            req.messages, req.max_tokens, req.temperature, req.model, req.think
        )
    except ModelNotFoundError as exc:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": (
                        "The model '{}' does not exist or is not currently "
                        "loaded by this backend instance.".format(
                            exc.requested_model
                        )
                    ),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )
    except (EngineUnavailableError, LifecycleUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))

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
    try:
        return inference_service.generate_text(
            req.prompt,
            req.max_new_tokens,
            req.temperature,
            req.think,
        )
    except (EngineUnavailableError, LifecycleUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _lifecycle_call(operation):
    """Runs a lifecycle operation and maps its failures to HTTP status
    codes. /admin/* is management surface and carries no OpenAI
    compatibility guarantee, so these use a plain error/detail body.
    """
    try:
        return operation()
    except LifecycleNotSupportedError as exc:
        return _lifecycle_error(501, "lifecycle_not_supported", str(exc))
    except ValueError as exc:
        return _lifecycle_error(404, "model_not_configured", str(exc))
    except LifecycleConflictError as exc:
        return _lifecycle_error(409, "lifecycle_conflict", str(exc))
    except EngineUnavailableError as exc:
        return _lifecycle_error(503, "engine_unavailable", str(exc))
    except ModelUnavailableError as exc:
        # Pre-flight rejection: nothing changed, previous model still serving.
        return _lifecycle_error(409, "model_unavailable", str(exc))
    except RuntimeError as exc:
        return _lifecycle_error(503, "lifecycle_failed", str(exc))


def _lifecycle_error(status_code: int, error: str, detail: str):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "detail": detail,
            "lifecycle_state": inference_service.lifecycle_state.value,
        },
    )


@router.post("/admin/model/load")
def admin_model_load(req: ModelLifecycleRequest):
    return _lifecycle_call(
        lambda: inference_service.load_model(req.model_id, req.persist)
    )


@router.post("/admin/model/unload")
def admin_model_unload():
    return _lifecycle_call(inference_service.unload_model)


@router.post("/admin/model/switch")
def admin_model_switch(req: ModelLifecycleRequest):
    return _lifecycle_call(
        lambda: inference_service.switch_model(req.model_id, req.persist)
    )
