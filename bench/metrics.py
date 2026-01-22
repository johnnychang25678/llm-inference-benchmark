"""Benchmark metrics collection and export."""

import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RequestResult:
    """Result from a single benchmark request."""

    request_id: str
    tenant_id: str
    prompt_name: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_latency_ms: float
    tokens_per_second: float
    status: str  # success, error, timeout
    ollama_num_parallel: int = 1
    error_message: str | None = None


@dataclass
class ScenarioResults:
    """Aggregated results for a benchmark scenario."""

    scenario_name: str
    concurrency: int
    prompt_size: str
    start_time: datetime
    ollama_num_parallel: int = 1
    end_time: datetime | None = None
    requests: list[RequestResult] = field(default_factory=list)
    peak_memory_mb: float | None = None
    oom_detected: bool = False

    @property
    def total_requests(self) -> int:
        """Total number of requests."""
        return len(self.requests)

    @property
    def successful_requests(self) -> int:
        """Number of successful requests."""
        return sum(1 for r in self.requests if r.status == "success")

    @property
    def error_rate(self) -> float:
        """Error rate as percentage."""
        if not self.requests:
            return 0.0
        errors = sum(1 for r in self.requests if r.status != "success")
        return (errors / len(self.requests)) * 100

    def _successful_values(self, attr: str) -> list[float]:
        """Get values for successful requests."""
        return [
            getattr(r, attr)
            for r in self.requests
            if r.status == "success" and getattr(r, attr, None) is not None
        ]

    @property
    def ttft_stats(self) -> dict[str, float]:
        """TTFT statistics in milliseconds."""
        values = self._successful_values("ttft_ms")
        return self._compute_stats(values)

    @property
    def latency_stats(self) -> dict[str, float]:
        """Total latency statistics in milliseconds."""
        values = self._successful_values("total_latency_ms")
        return self._compute_stats(values)

    @property
    def tps_stats(self) -> dict[str, float]:
        """Tokens per second statistics."""
        values = self._successful_values("tokens_per_second")
        return self._compute_stats(values)

    @staticmethod
    def _compute_stats(values: list[float]) -> dict[str, float]:
        """Compute statistics for a list of values."""
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "p50": sorted_values[int(n * 0.50)] if n > 0 else 0,
            "p95": sorted_values[int(n * 0.95)] if n > 0 else 0,
            "p99": sorted_values[int(n * 0.99)] if n > 0 else 0,
        }

    def to_summary(self) -> dict:
        """Convert to summary dictionary."""
        duration = 0.0
        if self.end_time and self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
            "scenario_name": self.scenario_name,
            "concurrency": self.concurrency,
            "ollama_num_parallel": self.ollama_num_parallel,
            "prompt_size": self.prompt_size,
            "duration_seconds": duration,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "error_rate_percent": round(self.error_rate, 2),
            "requests_per_second": (
                round(self.successful_requests / duration, 2) if duration > 0 else 0
            ),
            "ttft": self.ttft_stats,
            "latency": self.latency_stats,
            "tokens_per_second": self.tps_stats,
            "peak_memory_mb": self.peak_memory_mb,
            "oom_detected": self.oom_detected,
        }


class ResultsExporter:
    """Export benchmark results to various formats."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize exporter.

        Args:
            output_dir: Directory for output files.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_csv(self, results: list[ScenarioResults], filename: str) -> Path:
        """Export individual request results to CSV.

        Args:
            results: List of scenario results.
            filename: Output filename (without extension).

        Returns:
            Path to created file.
        """
        output_path = self.output_dir / f"{filename}.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "scenario",
                    "concurrency",
                    "ollama_num_parallel",
                    "peak_memory_mb",
                    "oom_detected",
                    "request_id",
                    "tenant_id",
                    "prompt_name",
                    "prompt_tokens",
                    "completion_tokens",
                    "ttft_ms",
                    "total_latency_ms",
                    "tokens_per_second",
                    "status",
                    "error_message",
                ],
            )
            writer.writeheader()

            for scenario in results:
                for req in scenario.requests:
                    row = asdict(req)
                    row["scenario"] = scenario.scenario_name
                    row["concurrency"] = scenario.concurrency
                    row["ollama_num_parallel"] = scenario.ollama_num_parallel
                    row["peak_memory_mb"] = scenario.peak_memory_mb
                    row["oom_detected"] = scenario.oom_detected
                    writer.writerow(row)

        return output_path

    def export_json(self, results: list[ScenarioResults], filename: str) -> Path:
        """Export summary results to JSON.

        Args:
            results: List of scenario results.
            filename: Output filename (without extension).

        Returns:
            Path to created file.
        """
        output_path = self.output_dir / f"{filename}.json"

        summaries = [r.to_summary() for r in results]

        with open(output_path, "w") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "scenarios": summaries,
                },
                f,
                indent=2,
            )

        return output_path

    def print_summary(self, results: list[ScenarioResults]) -> None:
        """Print results summary to console.

        Args:
            results: List of scenario results.
        """
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)

        for r in results:
            summary = r.to_summary()
            print(f"\n{summary['scenario_name']}")
            print("-" * 40)
            print(f"  Concurrency:     {summary['concurrency']}")
            print(f"  Ollama Parallel: {summary['ollama_num_parallel']}")
            print(f"  Prompt Size:     {summary['prompt_size']}")
            print(f"  Duration:        {summary['duration_seconds']:.1f}s")
            print(f"  Total Requests:  {summary['total_requests']}")
            print(f"  Success Rate:    {100 - summary['error_rate_percent']:.1f}%")
            print(f"  Requests/sec:    {summary['requests_per_second']:.2f}")
            print(f"  TTFT p50/p99:    {summary['ttft']['p50']:.0f}ms / {summary['ttft']['p99']:.0f}ms")
            print(f"  Latency p50/p99: {summary['latency']['p50']:.0f}ms / {summary['latency']['p99']:.0f}ms")
            print(f"  Tokens/sec:      {summary['tokens_per_second']['mean']:.1f} (mean)")
            if summary['peak_memory_mb']:
                print(f"  Peak Memory:     {summary['peak_memory_mb']:.0f} MB")
            if summary['oom_detected']:
                print(f"  OOM Detected:    Yes")

        print("\n" + "=" * 80)
