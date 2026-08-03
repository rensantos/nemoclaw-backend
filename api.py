import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from config import settings
from engines.base import (
    EngineUnavailableError,
    LifecycleNotSupportedError,
    ModelNotFoundError,
    ModelUnavailableError,
)
from services.inference import create_inference_service
from services.lifecycle import (
    LifecycleConflictError,
    LifecycleUnavailableError,
    StreamingNotSupportedError,
)
from schemas import ChatCompletionRequest, GenerateRequest, ModelLifecycleRequest


router = APIRouter()

_inference_service = None


def get_inference_service():
    """FastAPI dependency yielding the process-wide InferenceService.

    Built on first use rather than at import, so importing this module is
    cheap and side-effect free - it previously constructed the service at
    import time, which loads a model and made the API layer untestable
    without a GPU. app.py still triggers construction at startup, so
    server behaviour is unchanged.
    """
    global _inference_service
    if _inference_service is None:
        _inference_service = create_inference_service()
    return _inference_service


def set_inference_service(service):
    """Test seam: install a service before any request is served."""
    global _inference_service
    _inference_service = service


@router.get("/health")
def health_check(service=Depends(get_inference_service)):
    return service.health()


@router.get("/v1/models")
def models(service=Depends(get_inference_service)):
    try:
        return service.list_models()
    except EngineUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _sse_chunks(deltas, completion_id: str, model_id: str):
    """Renders engine deltas as OpenAI-convention SSE.

    Each event is a chat.completion.chunk; the stream ends with the
    conventional [DONE] sentinel. "usage" rides on the final chunk, which
    OpenAI only sends when asked for it - harmless extra information for
    clients that ignore it, and the only way a streaming caller can see
    token counts at all.
    """
    created = int(time.time())

    def chunk(delta, finish_reason=None, usage=None):
        body = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        if usage:
            body["usage"] = usage
        return "data: {}\n\n".format(json.dumps(body))

    yield chunk({"role": "assistant"})
    try:
        for delta in deltas:
            if delta.get("reasoning"):
                yield chunk({"reasoning": delta["reasoning"]})
            if delta.get("content"):
                yield chunk({"content": delta["content"]})
            if delta.get("usage"):
                yield chunk({}, finish_reason="stop", usage=delta["usage"])
    except (EngineUnavailableError, LifecycleUnavailableError) as exc:
        # The response has already begun, so the status code is spent.
        # Surfacing the error in-band beats a silently truncated answer.
        yield "data: {}\n\n".format(
            json.dumps({"error": {"message": str(exc), "type": "server_error"}})
        )
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionRequest, service=Depends(get_inference_service)
):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    model_id = req.model or settings.model_id

    if req.stream:
        try:
            deltas = service.chat_stream(
                req.messages, req.max_tokens, req.temperature, req.model, req.think
            )
        except StreamingNotSupportedError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except LifecycleUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return StreamingResponse(
            _sse_chunks(
                deltas, "chatcmpl-{}".format(uuid.uuid4().hex), model_id
            ),
            media_type="text/event-stream",
        )

    try:
        result = service.chat(
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

    message = {"role": "assistant", "content": result["content"]}
    # Nemoclaw extension: a reasoning model's hidden thinking, kept out of
    # "content" so OpenAI clients render only the answer. Omitted entirely
    # when the model did no reasoning.
    if result.get("reasoning"):
        message["reasoning"] = result["reasoning"]

    return {
        "id": "chatcmpl-{}".format(uuid.uuid4().hex),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
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
def generate(req: GenerateRequest, service=Depends(get_inference_service)):
    try:
        return service.generate_text(
            req.prompt,
            req.max_new_tokens,
            req.temperature,
            req.think,
        )
    except (EngineUnavailableError, LifecycleUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _lifecycle_call(service, operation):
    """Runs a lifecycle operation and maps its failures to HTTP status
    codes. /admin/* is management surface and carries no OpenAI
    compatibility guarantee, so these use a plain error/detail body.
    """
    try:
        return operation()
    except LifecycleNotSupportedError as exc:
        return _lifecycle_error(service, 501, "lifecycle_not_supported", str(exc))
    except ValueError as exc:
        return _lifecycle_error(service, 404, "model_not_configured", str(exc))
    except LifecycleConflictError as exc:
        return _lifecycle_error(service, 409, "lifecycle_conflict", str(exc))
    except EngineUnavailableError as exc:
        return _lifecycle_error(service, 503, "engine_unavailable", str(exc))
    except ModelUnavailableError as exc:
        # Pre-flight rejection: nothing changed, previous model still serving.
        return _lifecycle_error(service, 409, "model_unavailable", str(exc))
    except RuntimeError as exc:
        return _lifecycle_error(service, 503, "lifecycle_failed", str(exc))


def _lifecycle_error(service, status_code: int, error: str, detail: str):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "detail": detail,
            "lifecycle_state": service.lifecycle_state.value,
        },
    )


@router.post("/admin/model/load")
def admin_model_load(
    req: ModelLifecycleRequest, service=Depends(get_inference_service)
):
    return _lifecycle_call(
        service, lambda: service.load_model(req.model_id, req.persist)
    )


@router.post("/admin/model/unload")
def admin_model_unload(service=Depends(get_inference_service)):
    return _lifecycle_call(service, service.unload_model)


@router.post("/admin/model/switch")
def admin_model_switch(
    req: ModelLifecycleRequest, service=Depends(get_inference_service)
):
    return _lifecycle_call(
        service, lambda: service.switch_model(req.model_id, req.persist)
    )
