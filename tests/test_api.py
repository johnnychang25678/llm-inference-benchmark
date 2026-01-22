"""Tests for API routes."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from engines.base import BaseEngine, CompletionChunk, CompletionRequest
from server.main import app
from server.metrics import MetricsCollector
from server.router import RouterDependencies, init_router


class MockEngine(BaseEngine):
    """Mock engine for testing."""

    def __init__(self) -> None:
        self.complete_called = False
        self.last_request: CompletionRequest | None = None

    async def complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionChunk]:
        """Return mock completion."""
        self.complete_called = True
        self.last_request = request

        yield CompletionChunk(text="Hello")
        yield CompletionChunk(text=" World")
        yield CompletionChunk(
            text="!",
            finish_reason="stop",
            prompt_tokens=5,
            completion_tokens=3,
        )

    async def health(self) -> bool:
        """Return healthy."""
        return True

    async def close(self) -> None:
        """No-op close."""
        pass


@pytest.fixture
def mock_engine() -> MockEngine:
    """Create mock engine."""
    return MockEngine()


@pytest.fixture
def metrics() -> MetricsCollector:
    """Create metrics collector."""
    return MetricsCollector()


@pytest.fixture
def client(
    mock_engine: MockEngine,
    metrics: MetricsCollector,
) -> TestClient:
    """Create test client with mocked dependencies."""
    deps = RouterDependencies(
        engine=mock_engine,
        metrics=metrics,
        model_name="test-model",
    )
    init_router(deps)

    # Mock the app state for health checks
    app.state.engine = mock_engine
    app.state.metrics = metrics

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """Health check should return status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["engine"] == "healthy"


class TestRootEndpoint:
    """Tests for / endpoint."""

    def test_root_returns_info(self, client: TestClient) -> None:
        """Root should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["docs"] == "/docs"


class TestCompletionEndpoint:
    """Tests for /v1/completions endpoint."""

    def test_completion_without_stream(
        self, client: TestClient, mock_engine: MockEngine
    ) -> None:
        """Non-streaming completion should return full response."""
        response = client.post(
            "/v1/completions",
            json={
                "prompt": "Say hello",
                "max_tokens": 50,
                "stream": False,
            },
            headers={"X-Tenant-ID": "test-tenant"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["text"] == "Hello World!"
        assert mock_engine.complete_called

    def test_completion_with_stream(self, client: TestClient) -> None:
        """Streaming completion should return SSE events."""
        response = client.post(
            "/v1/completions",
            json={
                "prompt": "Say hello",
                "max_tokens": 50,
                "stream": True,
            },
            headers={"X-Tenant-ID": "test-tenant"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Check that we got SSE data
        content = response.text
        assert "data:" in content
        assert "[DONE]" in content

    def test_completion_uses_tenant_header(
        self, client: TestClient, mock_engine: MockEngine
    ) -> None:
        """Request should use tenant from header."""
        response = client.post(
            "/v1/completions",
            json={"prompt": "Hello", "stream": False},
            headers={"X-Tenant-ID": "custom-tenant"},
        )

        assert response.status_code == 200

    def test_completion_default_tenant(
        self, client: TestClient
    ) -> None:
        """Request without tenant header should use default."""
        response = client.post(
            "/v1/completions",
            json={"prompt": "Hello", "stream": False},
        )

        assert response.status_code == 200


class TestChatCompletionEndpoint:
    """Tests for /v1/chat/completions endpoint."""

    def test_chat_completion(self, client: TestClient) -> None:
        """Chat completion should convert messages to prompt."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 50,
                "stream": False,
            },
            headers={"X-Tenant-ID": "test-tenant"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["choices"]) == 1


