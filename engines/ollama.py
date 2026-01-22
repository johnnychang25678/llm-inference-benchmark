"""Ollama inference engine adapter."""

import json
from collections.abc import AsyncIterator

import httpx

from engines.base import BaseEngine, CompletionChunk, CompletionRequest


class OllamaEngine(BaseEngine):
    """Adapter for Ollama inference server.

    Connects to Ollama's HTTP API for text generation with streaming support.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        timeout: float = 120.0,
    ) -> None:
        """Initialize Ollama engine adapter.

        Args:
            base_url: Ollama server URL.
            model: Model name to use for generation.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionChunk]:
        """Generate completion using Ollama API.

        Args:
            request: The completion request.

        Yields:
            CompletionChunk objects with generated text.
        """
        client = await self._get_client()

        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        if request.stop:
            payload["options"]["stop"] = request.stop

        if request.stream:
            async with client.stream(
                "POST",
                "/api/generate",
                json=payload,
            ) as response:
                response.raise_for_status()
                prompt_tokens: int | None = None
                completion_tokens = 0

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    data = json.loads(line)

                    if "error" in data:
                        raise RuntimeError(f"Ollama error: {data['error']}")

                    text = data.get("response", "")
                    done = data.get("done", False)

                    if text:
                        completion_tokens += 1

                    if done:
                        prompt_tokens = data.get("prompt_eval_count")
                        final_completion_tokens = data.get("eval_count", completion_tokens)
                        yield CompletionChunk(
                            text=text,
                            finish_reason="stop",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=final_completion_tokens,
                        )
                    elif text:
                        yield CompletionChunk(text=text)
        else:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise RuntimeError(f"Ollama error: {data['error']}")

            yield CompletionChunk(
                text=data.get("response", ""),
                finish_reason="stop",
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
            )

    async def health(self) -> bool:
        """Check Ollama server health.

        Returns:
            True if server is responsive.
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
