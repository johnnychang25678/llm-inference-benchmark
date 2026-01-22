"""Tests for engine adapters."""

import pytest

from engines.base import BaseEngine, CompletionChunk, CompletionRequest
from engines.sglang import SGLangEngine
from engines.vllm import VLLMEngine


class TestCompletionRequest:
    """Tests for CompletionRequest dataclass."""

    def test_default_values(self) -> None:
        """Request should have sensible defaults."""
        request = CompletionRequest(prompt="Hello")
        assert request.prompt == "Hello"
        assert request.max_tokens == 512
        assert request.temperature == 0.7
        assert request.stream is True
        assert request.stop is None

    def test_custom_values(self) -> None:
        """Request should accept custom values."""
        request = CompletionRequest(
            prompt="Test",
            max_tokens=100,
            temperature=0.5,
            stream=False,
            stop=["END"],
        )
        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.stream is False
        assert request.stop == ["END"]

    def test_frozen(self) -> None:
        """Request should be immutable."""
        request = CompletionRequest(prompt="Hello")
        with pytest.raises(AttributeError):
            request.prompt = "Changed"  # type: ignore


class TestCompletionChunk:
    """Tests for CompletionChunk dataclass."""

    def test_minimal_chunk(self) -> None:
        """Chunk should work with just text."""
        chunk = CompletionChunk(text="Hello")
        assert chunk.text == "Hello"
        assert chunk.finish_reason is None
        assert chunk.prompt_tokens is None
        assert chunk.completion_tokens is None

    def test_final_chunk(self) -> None:
        """Final chunk should have all fields."""
        chunk = CompletionChunk(
            text="",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=50,
        )
        assert chunk.finish_reason == "stop"
        assert chunk.prompt_tokens == 10
        assert chunk.completion_tokens == 50


class TestVLLMStub:
    """Tests for vLLM stub adapter."""

    @pytest.fixture
    def engine(self) -> VLLMEngine:
        """Create vLLM engine."""
        return VLLMEngine()

    @pytest.mark.asyncio
    async def test_health_returns_false(self, engine: VLLMEngine) -> None:
        """Stub should report unhealthy."""
        result = await engine.health()
        assert result is False

    @pytest.mark.asyncio
    async def test_complete_raises_not_implemented(
        self, engine: VLLMEngine
    ) -> None:
        """Stub should raise NotImplementedError."""
        request = CompletionRequest(prompt="Hello")
        with pytest.raises(NotImplementedError):
            async for _ in engine.complete(request):
                pass

    @pytest.mark.asyncio
    async def test_close_succeeds(self, engine: VLLMEngine) -> None:
        """Close should succeed without error."""
        await engine.close()


class TestSGLangStub:
    """Tests for SGLang stub adapter."""

    @pytest.fixture
    def engine(self) -> SGLangEngine:
        """Create SGLang engine."""
        return SGLangEngine()

    @pytest.mark.asyncio
    async def test_health_returns_false(self, engine: SGLangEngine) -> None:
        """Stub should report unhealthy."""
        result = await engine.health()
        assert result is False

    @pytest.mark.asyncio
    async def test_complete_raises_not_implemented(
        self, engine: SGLangEngine
    ) -> None:
        """Stub should raise NotImplementedError."""
        request = CompletionRequest(prompt="Hello")
        with pytest.raises(NotImplementedError):
            async for _ in engine.complete(request):
                pass


class TestBaseEngineInterface:
    """Tests to verify BaseEngine interface."""

    def test_cannot_instantiate_base(self) -> None:
        """BaseEngine should not be instantiable."""
        with pytest.raises(TypeError):
            BaseEngine()  # type: ignore

    def test_vllm_is_base_engine(self) -> None:
        """VLLMEngine should be a BaseEngine."""
        engine = VLLMEngine()
        assert isinstance(engine, BaseEngine)

    def test_sglang_is_base_engine(self) -> None:
        """SGLangEngine should be a BaseEngine."""
        engine = SGLangEngine()
        assert isinstance(engine, BaseEngine)
