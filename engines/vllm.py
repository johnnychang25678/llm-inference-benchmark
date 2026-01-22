"""vLLM inference engine adapter (stub implementation)."""

from collections.abc import AsyncIterator

from engines.base import BaseEngine, CompletionChunk, CompletionRequest


class VLLMEngine(BaseEngine):
    """Stub adapter for vLLM inference server.

    This is a placeholder implementation demonstrating the adapter pattern.
    A real implementation would connect to a vLLM server via its OpenAI-compatible
    API or native gRPC interface.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
    ) -> None:
        """Initialize vLLM engine adapter.

        Args:
            base_url: vLLM server URL.
            model: Model name for request routing.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionChunk]:
        """Generate completion using vLLM API.

        Args:
            request: The completion request.

        Yields:
            CompletionChunk objects with generated text.

        Raises:
            NotImplementedError: This is a stub implementation.
        """
        raise NotImplementedError(
            "vLLM adapter is a stub. Implement connection to vLLM's "
            "OpenAI-compatible API at /v1/completions for production use."
        )
        yield  # Make this a generator

    async def health(self) -> bool:
        """Check vLLM server health.

        Returns:
            False (stub implementation).
        """
        return False

    async def close(self) -> None:
        """Close any resources."""
        pass
