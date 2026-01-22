"""Prometheus metrics collector for inference gateway."""

import time
from dataclasses import dataclass, field

from prometheus_client import Counter, Gauge, Histogram

# Request latency histogram with tenant and status labels
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Total request latency in seconds",
    labelnames=["tenant", "status"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

# Time to first token histogram
TTFT = Histogram(
    "inference_time_to_first_token_seconds",
    "Time to first token in seconds",
    labelnames=["tenant"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Tokens per second histogram
TOKENS_PER_SECOND = Histogram(
    "inference_tokens_per_second",
    "Token generation rate (tokens/second)",
    labelnames=["tenant"],
    buckets=(1, 5, 10, 20, 50, 100, 200, 500),
)

# Total tokens generated counter
TOKENS_GENERATED = Counter(
    "inference_tokens_generated_total",
    "Total tokens generated",
    labelnames=["tenant", "type"],  # type: prompt or completion
)

# Queue depth gauge (requests in flight per tenant)
QUEUE_DEPTH = Gauge(
    "inference_queue_depth",
    "Current requests in flight",
    labelnames=["tenant"],
)

# Ollama memory usage gauge
OLLAMA_MEMORY = Gauge(
    "ollama_memory_mb",
    "Ollama process memory usage in MB",
)


@dataclass
class RequestMetrics:
    """Metrics collected during a single request."""

    tenant_id: str
    start_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None
    end_time: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    status: str = "success"

    def record_first_token(self) -> None:
        """Record when the first token was generated."""
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def record_completion(
        self,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        status: str = "success",
    ) -> None:
        """Record request completion.

        Args:
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            status: Request status (success, error).
        """
        self.end_time = time.perf_counter()
        if prompt_tokens is not None:
            self.prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            self.completion_tokens = completion_tokens
        self.status = status

    def emit(self) -> None:
        """Emit all collected metrics to Prometheus."""
        if self.end_time is None:
            self.end_time = time.perf_counter()

        total_latency = self.end_time - self.start_time
        REQUEST_LATENCY.labels(tenant=self.tenant_id, status=self.status).observe(
            total_latency
        )

        if self.first_token_time is not None:
            ttft = self.first_token_time - self.start_time
            TTFT.labels(tenant=self.tenant_id).observe(ttft)

            # Calculate tokens/second (from first token to end)
            generation_time = self.end_time - self.first_token_time
            if generation_time > 0 and self.completion_tokens > 0:
                tps = self.completion_tokens / generation_time
                TOKENS_PER_SECOND.labels(tenant=self.tenant_id).observe(tps)

        if self.prompt_tokens > 0:
            TOKENS_GENERATED.labels(tenant=self.tenant_id, type="prompt").inc(
                self.prompt_tokens
            )
        if self.completion_tokens > 0:
            TOKENS_GENERATED.labels(tenant=self.tenant_id, type="completion").inc(
                self.completion_tokens
            )


class MetricsCollector:
    """Coordinates metrics collection across the gateway."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        pass

    def start_request(self, tenant_id: str) -> RequestMetrics:
        """Start tracking a new request.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            RequestMetrics instance to track the request.
        """
        QUEUE_DEPTH.labels(tenant=tenant_id).inc()
        return RequestMetrics(tenant_id=tenant_id)

    def end_request(self, metrics: RequestMetrics) -> None:
        """End request tracking and emit metrics.

        Args:
            metrics: RequestMetrics instance from start_request.
        """
        QUEUE_DEPTH.labels(tenant=metrics.tenant_id).dec()
        metrics.emit()
