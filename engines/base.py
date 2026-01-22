"""Abstract base class for inference engine adapters."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionRequest:
    """Request for text completion."""

    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = True
    stop: list[str] | None = None


@dataclass(frozen=True)
class CompletionChunk:
    """A chunk of completion response (for streaming)."""

    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class BaseEngine(ABC):
    """Abstract base class for inference engine adapters.

    All engine implementations must inherit from this class and implement
    the complete() and health() methods.
    """

    @abstractmethod
    async def complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionChunk]:
        """Generate completion for the given request.

        Args:
            request: The completion request with prompt and parameters.

        Yields:
            CompletionChunk objects containing generated text.
        """
        ...

    @abstractmethod
    async def health(self) -> bool:
        """Check if the engine is healthy and ready to serve requests.

        Returns:
            True if the engine is healthy, False otherwise.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources used by the engine."""
        ...
