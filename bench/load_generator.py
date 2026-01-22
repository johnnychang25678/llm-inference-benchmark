"""Async load generator for benchmarking."""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass

import httpx

from bench.metrics import RequestResult, ScenarioResults
from bench.scenarios import Prompt, Scenario


@dataclass
class LoadGeneratorConfig:
    """Load generator configuration."""

    gateway_url: str = "http://localhost:8000"
    timeout: float = 120.0
    max_retries: int = 0


class LoadGenerator:
    """Async load generator for benchmark scenarios."""

    def __init__(self, config: LoadGeneratorConfig | None = None) -> None:
        """Initialize load generator.

        Args:
            config: Generator configuration.
        """
        self.config = config or LoadGeneratorConfig()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.gateway_url,
                timeout=httpx.Timeout(self.config.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _send_request(
        self,
        prompt: Prompt,
        tenant_id: str,
    ) -> RequestResult:
        """Send a single completion request.

        Args:
            prompt: The prompt to send.
            tenant_id: Tenant identifier.

        Returns:
            Request result with metrics.
        """
        request_id = str(uuid.uuid4())[:8]
        client = await self._get_client()

        start_time = time.perf_counter()
        ttft: float | None = None
        completion_tokens = 0
        prompt_tokens = 0
        status = "success"
        error_message: str | None = None

        try:
            async with client.stream(
                "POST",
                "/v1/completions",
                json={
                    "prompt": prompt.to_completion_prompt(),
                    "temperature": 0.7,
                    "stream": True,
                },
                headers={"X-Tenant-ID": tenant_id},
            ) as response:
                if response.status_code == 429:
                    status = "rate_limited"
                    error_message = "Rate limit exceeded"
                elif response.status_code != 200:
                    status = "error"
                    error_message = f"HTTP {response.status_code}"
                else:
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        if ttft is None:
                            ttft = (time.perf_counter() - start_time) * 1000

                        try:
                            data = json.loads(data_str)
                            text = data.get("choices", [{}])[0].get("text", "")
                            if text:
                                completion_tokens += 1
                        except json.JSONDecodeError:
                            pass

        except httpx.HTTPStatusError as e:
            status = "error"
            error_message = str(e)
        except httpx.TimeoutException:
            status = "error"
            error_message = "Request timeout"
        except Exception as e:
            status = "error"
            error_message = str(e)

        end_time = time.perf_counter()
        total_latency_ms = (end_time - start_time) * 1000

        # Calculate tokens per second
        if ttft is not None and completion_tokens > 0:
            generation_time = total_latency_ms - ttft
            tokens_per_second = (
                (completion_tokens / generation_time) * 1000
                if generation_time > 0
                else 0
            )
        else:
            tokens_per_second = 0

        return RequestResult(
            request_id=request_id,
            tenant_id=tenant_id,
            prompt_name=prompt.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft or 0,
            total_latency_ms=total_latency_ms,
            tokens_per_second=tokens_per_second,
            status=status,
            error_message=error_message,
        )

    async def _worker(
        self,
        prompts: list[Prompt],
        tenant_id: str,
        results: list[RequestResult],
        stop_event: asyncio.Event,
    ) -> None:
        """Worker coroutine that sends requests until stopped.

        Args:
            prompts: List of prompts to cycle through.
            tenant_id: Tenant identifier.
            results: Shared list to append results.
            stop_event: Event to signal stop.
        """
        prompt_idx = 0

        while not stop_event.is_set():
            prompt = prompts[prompt_idx % len(prompts)]
            result = await self._send_request(prompt, tenant_id)
            results.append(result)
            prompt_idx += 1

    async def run_scenario(
        self,
        scenario: Scenario,
        on_progress=None,
    ) -> ScenarioResults:
        """Run a benchmark scenario.

        Args:
            scenario: Scenario to run.
            on_progress: Optional callback for progress updates.

        Returns:
            Scenario results with all request metrics.
        """
        from datetime import datetime

        results = ScenarioResults(
            scenario_name=scenario.name,
            concurrency=scenario.concurrency,
            prompt_size=scenario.prompts[0].name if scenario.prompts else "unknown",
            start_time=datetime.now(),
        )

        # Warmup
        if scenario.warmup_requests > 0 and on_progress:
            on_progress(f"Warming up with {scenario.warmup_requests} requests...")

        for _ in range(scenario.warmup_requests):
            await self._send_request(scenario.prompts[0], scenario.tenant_id)

        # Run benchmark
        if on_progress:
            on_progress(
                f"Running {scenario.name} with concurrency {scenario.concurrency}..."
            )

        stop_event = asyncio.Event()
        request_results: list[RequestResult] = []

        # Start workers
        workers = [
            asyncio.create_task(
                self._worker(
                    scenario.prompts,
                    scenario.tenant_id,
                    request_results,
                    stop_event,
                )
            )
            for _ in range(scenario.concurrency)
        ]

        # Wait for duration
        await asyncio.sleep(scenario.duration_seconds)
        stop_event.set()

        # Wait for workers to finish current requests
        await asyncio.gather(*workers, return_exceptions=True)

        results.requests = request_results
        results.end_time = datetime.now()

        return results

    async def run_cache_test(
        self,
        iterations: int = 10,
        prompt_size: str = "large",
        tenant_id: str = "cache_test",
        on_progress=None,
    ) -> dict:
        """Run cache effect isolation test.

        Compares TTFT between:
        - Same prompt repeated (warm cache)
        - Different prompts each time (cold cache)

        Args:
            iterations: Number of requests per test.
            prompt_size: Prompt size to use (small, medium, large).
            tenant_id: Tenant identifier.
            on_progress: Optional callback for progress updates.

        Returns:
            Dictionary with warm/cold cache TTFT statistics.
        """
        from bench.scenarios import PROMPTS, VARYING_PROMPTS

        # Get prompts
        static_prompt = PROMPTS.get(prompt_size)
        varying_prompts = VARYING_PROMPTS.get(prompt_size, [])

        if not static_prompt:
            raise ValueError(f"Unknown prompt size: {prompt_size}")
        if len(varying_prompts) < 2:
            raise ValueError(f"Need at least 2 varying prompts for {prompt_size}")

        # Test 1: Same prompt (warm cache after first request)
        if on_progress:
            on_progress(f"Testing warm cache (same prompt, {iterations} requests)...")

        warm_ttfts: list[float] = []
        for i in range(iterations):
            result = await self._send_request(static_prompt, tenant_id)
            if result.status == "success":
                warm_ttfts.append(result.ttft_ms)
            if on_progress and (i + 1) % 5 == 0:
                on_progress(f"  Warm cache: {i + 1}/{iterations} requests")

        # Test 2: Different prompts (cold cache each time)
        if on_progress:
            on_progress(f"Testing cold cache (varying prompts, {iterations} requests)...")

        cold_ttfts: list[float] = []
        for i in range(iterations):
            prompt = varying_prompts[i % len(varying_prompts)]
            result = await self._send_request(prompt, tenant_id)
            if result.status == "success":
                cold_ttfts.append(result.ttft_ms)
            if on_progress and (i + 1) % 5 == 0:
                on_progress(f"  Cold cache: {i + 1}/{iterations} requests")

        def stats(values: list[float]) -> dict:
            if not values:
                return {"mean": 0, "min": 0, "max": 0, "p50": 0, "count": 0}
            sorted_vals = sorted(values)
            p50_idx = len(sorted_vals) // 2
            return {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "p50": sorted_vals[p50_idx],
                "count": len(values),
            }

        warm_stats = stats(warm_ttfts[1:])  # Skip first (cold start)
        cold_stats = stats(cold_ttfts)
        first_request_ttft = warm_ttfts[0] if warm_ttfts else 0

        cache_benefit_ms = cold_stats["mean"] - warm_stats["mean"]
        cache_benefit_pct = (
            (cache_benefit_ms / cold_stats["mean"]) * 100
            if cold_stats["mean"] > 0
            else 0
        )

        return {
            "prompt_size": prompt_size,
            "iterations": iterations,
            "first_request_ttft_ms": first_request_ttft,
            "warm_cache": warm_stats,
            "cold_cache": cold_stats,
            "cache_benefit_ms": cache_benefit_ms,
            "cache_benefit_percent": cache_benefit_pct,
        }

