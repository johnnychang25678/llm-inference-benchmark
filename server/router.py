"""API routes for the inference gateway."""

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engines import BaseEngine, CompletionRequest
from server.metrics import MetricsCollector
from server.middleware.tenant import get_tenant_id

router = APIRouter()


class CompletionRequestBody(BaseModel):
    """OpenAI-compatible completion request body."""

    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    stop: list[str] | None = None


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""

    id: str
    object: str = "text_completion"
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


class ChatMessage(BaseModel):
    """Chat message."""

    role: str
    content: str


class ChatCompletionRequestBody(BaseModel):
    """OpenAI-compatible chat completion request body."""

    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    stop: list[str] | None = None


@dataclass
class RouterDependencies:
    """Dependencies for router handlers."""

    engine: BaseEngine
    metrics: MetricsCollector
    model_name: str


_deps: RouterDependencies | None = None


def init_router(deps: RouterDependencies) -> None:
    """Initialize router with dependencies.

    Args:
        deps: Router dependencies.
    """
    global _deps
    _deps = deps


def get_deps() -> RouterDependencies:
    """Get router dependencies.

    Returns:
        Router dependencies.

    Raises:
        RuntimeError: If router not initialized.
    """
    if _deps is None:
        raise RuntimeError("Router not initialized. Call init_router first.")
    return _deps


@router.post("/v1/completions", response_model=None)
async def create_completion(
    request: Request,
    body: CompletionRequestBody,
) -> CompletionResponse | StreamingResponse:
    """Create a text completion.

    Args:
        request: FastAPI request.
        body: Completion request body.

    Returns:
        Completion response or streaming response.
    """
    deps = get_deps()
    tenant_id = get_tenant_id(request)

    # Start metrics collection
    req_metrics = deps.metrics.start_request(tenant_id)

    try:
        completion_request = CompletionRequest(
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            stream=body.stream,
            stop=body.stop,
        )

        if body.stream:
            return StreamingResponse(
                _stream_completion(completion_request, req_metrics, deps),
                media_type="text/event-stream",
            )

        # Non-streaming response
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        async for chunk in deps.engine.complete(completion_request):
            if req_metrics.first_token_time is None:
                req_metrics.record_first_token()
            full_text += chunk.text
            if chunk.prompt_tokens:
                prompt_tokens = chunk.prompt_tokens
            if chunk.completion_tokens:
                completion_tokens = chunk.completion_tokens

        req_metrics.record_completion(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            status="success",
        )
        deps.metrics.end_request(req_metrics)

        return CompletionResponse(
            id=f"cmpl-{id(request)}",
            model=deps.model_name,
            choices=[
                {
                    "text": full_text,
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    except Exception as e:
        req_metrics.record_completion(status="error")
        deps.metrics.end_request(req_metrics)
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _stream_completion(
    request: CompletionRequest,
    req_metrics: Any,
    deps: RouterDependencies,
) -> Any:
    """Stream completion chunks as SSE events.

    Args:
        request: Completion request.
        req_metrics: Request metrics tracker.
        deps: Router dependencies.

    Yields:
        SSE-formatted completion chunks.
    """
    import json

    prompt_tokens = 0
    completion_tokens = 0

    try:
        async for chunk in deps.engine.complete(request):
            if req_metrics.first_token_time is None:
                req_metrics.record_first_token()

            if chunk.prompt_tokens:
                prompt_tokens = chunk.prompt_tokens
            if chunk.completion_tokens:
                completion_tokens = chunk.completion_tokens

            data = {
                "id": f"cmpl-stream",
                "object": "text_completion",
                "model": deps.model_name,
                "choices": [
                    {
                        "text": chunk.text,
                        "index": 0,
                        "finish_reason": chunk.finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

        req_metrics.record_completion(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            status="success",
        )
    except Exception:
        req_metrics.record_completion(status="error")
        raise
    finally:
        deps.metrics.end_request(req_metrics)


@router.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    request: Request,
    body: ChatCompletionRequestBody,
) -> CompletionResponse | StreamingResponse:
    """Create a chat completion.

    Converts chat messages to a prompt and forwards to the completion endpoint.

    Args:
        request: FastAPI request.
        body: Chat completion request body.

    Returns:
        Completion response or streaming response.
    """
    # Convert chat messages to prompt
    prompt_parts = []
    for msg in body.messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")

    prompt_parts.append("Assistant:")
    prompt = "\n\n".join(prompt_parts)

    # Forward to completion endpoint
    completion_body = CompletionRequestBody(
        prompt=prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        stream=body.stream,
        stop=body.stop,
    )

    return await create_completion(request, completion_body)
