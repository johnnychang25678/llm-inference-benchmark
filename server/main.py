"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import psutil
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from engines import OllamaEngine
from server.config import settings
from server.metrics import MetricsCollector, OLLAMA_MEMORY
from server.middleware import TenantMiddleware
from server.router import RouterDependencies, init_router, router


async def update_ollama_memory() -> None:
    """Background task to sample Ollama memory usage.

    Periodically reads Ollama process memory and updates the Prometheus gauge.
    """
    while True:
        memory_mb = 0.0
        for proc in psutil.process_iter(["name", "memory_info"]):
            try:
                if proc.info["name"] and "ollama" in proc.info["name"].lower():
                    memory_mb += proc.info["memory_info"].rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        OLLAMA_MEMORY.set(memory_mb)
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Application lifespan manager.

    Initializes and cleans up resources on startup and shutdown.
    """
    # Initialize engine
    engine = OllamaEngine(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout,
    )

    # Initialize metrics
    metrics = MetricsCollector()

    # Initialize router dependencies
    deps = RouterDependencies(
        engine=engine,
        metrics=metrics,
        model_name=settings.ollama_model,
    )
    init_router(deps)

    # Store in app state for health checks
    app.state.engine = engine
    app.state.metrics = metrics

    # Start background task for memory monitoring
    memory_task = asyncio.create_task(update_ollama_memory())

    yield

    # Cleanup
    memory_task.cancel()
    await engine.close()


app = FastAPI(
    title="LLM Inference Gateway",
    description="Multi-tenant inference gateway with observability",
    version="0.1.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(TenantMiddleware)

# Include API routes
app.include_router(router)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint.

    Returns:
        Health status of the gateway and backend engine.
    """
    engine_healthy = await app.state.engine.health()
    return {
        "status": "healthy" if engine_healthy else "degraded",
        "engine": "healthy" if engine_healthy else "unhealthy",
        "model": settings.ollama_model,
    }


@app.get("/metrics")
async def metrics() -> Any:
    """Prometheus metrics endpoint.

    Returns:
        Prometheus-formatted metrics.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint.

    Returns:
        Welcome message.
    """
    return {
        "message": "LLM Inference Gateway",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
