"""CLI for benchmark tools."""

import asyncio
from datetime import datetime
from itertools import groupby
from pathlib import Path

import click
import httpx

from bench.load_generator import LoadGenerator, LoadGeneratorConfig
from bench.metrics import ResultsExporter, ScenarioResults
from bench.ollama_manager import OllamaManager, OllamaUnresponsiveError
from bench.scenarios import get_matrix_scenarios


@click.group()
@click.option(
    "--gateway-url",
    default="http://localhost:8000",
    help="Gateway URL",
)
@click.pass_context
def cli(ctx: click.Context, gateway_url: str) -> None:
    """LLM Inference Benchmark CLI."""
    ctx.ensure_object(dict)
    ctx.obj["gateway_url"] = gateway_url


@cli.command()
@click.option(
    "--concurrency",
    default="1,4,16",
    help="Comma-separated concurrency levels",
)
@click.option(
    "--prompt-sizes",
    default="small,medium,large",
    help="Comma-separated prompt sizes",
)
@click.option(
    "--ollama-num-parallel",
    default="1",
    help="Comma-separated OLLAMA_NUM_PARALLEL values to test",
)
@click.option(
    "--duration",
    default=30,
    type=int,
    help="Duration per scenario in seconds",
)
@click.option(
    "--output",
    default="results",
    help="Output directory for results",
)
@click.option(
    "--vary-prompts",
    is_flag=True,
    default=False,
    help="Use varying prompts to avoid cache benefits (tests true parallelism)",
)
@click.option(
    "--ollama-num-ctx",
    default=32768,
    type=int,
    help="OLLAMA_NUM_CTX context window size (default: 32768 for qwen2.5:7b)",
)
@click.pass_context
def run(
    ctx: click.Context,
    concurrency: str,
    prompt_sizes: str,
    ollama_num_parallel: str,
    duration: int,
    output: str,
    vary_prompts: bool,
    ollama_num_ctx: int,
) -> None:
    """Run benchmark matrix."""
    gateway_url = ctx.obj["gateway_url"]
    concurrency_levels = [int(c.strip()) for c in concurrency.split(",")]
    sizes = [s.strip() for s in prompt_sizes.split(",")]
    parallel_values = [int(p.strip()) for p in ollama_num_parallel.split(",")]

    total_scenarios = len(concurrency_levels) * len(sizes) * len(parallel_values)

    click.echo(f"Gateway: {gateway_url}")
    click.echo(f"Concurrency levels: {concurrency_levels}")
    click.echo(f"Prompt sizes: {sizes}")
    click.echo(f"Ollama parallel values: {parallel_values}")
    click.echo(f"Ollama context window: {ollama_num_ctx}")
    click.echo(f"Duration per scenario: {duration}s")
    click.echo(f"Vary prompts: {vary_prompts}")
    click.echo(f"Total scenarios: {total_scenarios}")
    click.echo()

    asyncio.run(
        _run_benchmark(
            gateway_url=gateway_url,
            concurrency_levels=concurrency_levels,
            prompt_sizes=sizes,
            parallel_values=parallel_values,
            duration=duration,
            output_dir=Path(output),
            vary_prompts=vary_prompts,
            num_ctx=ollama_num_ctx,
        )
    )


async def _run_benchmark(
    gateway_url: str,
    concurrency_levels: list[int],
    prompt_sizes: list[str],
    parallel_values: list[int],
    duration: int,
    output_dir: Path,
    vary_prompts: bool = False,
    num_ctx: int = 32768,
) -> None:
    """Run benchmark scenarios with Ollama lifecycle management."""
    config = LoadGeneratorConfig(gateway_url=gateway_url)
    generator = LoadGenerator(config)
    exporter = ResultsExporter(output_dir)
    ollama_manager = OllamaManager()

    # Generate all scenarios
    scenarios = get_matrix_scenarios(
        prompt_sizes=prompt_sizes,
        concurrency_levels=concurrency_levels,
        ollama_num_parallel_values=parallel_values,
        duration_seconds=float(duration),
        vary_prompts=vary_prompts,
    )

    # Group scenarios by ollama_num_parallel for efficient restarts
    scenarios_by_parallel = {
        k: list(v)
        for k, v in groupby(
            sorted(scenarios, key=lambda s: s.ollama_num_parallel),
            key=lambda s: s.ollama_num_parallel,
        )
    }

    all_results: list[ScenarioResults] = []
    scenario_idx = 0
    total_scenarios = len(scenarios)

    try:
        for parallel_value in sorted(scenarios_by_parallel.keys()):
            group_scenarios = scenarios_by_parallel[parallel_value]

            # Restart Ollama with new parallel value
            click.echo(f"\n{'='*60}")
            click.echo(f"Configuring Ollama: OLLAMA_NUM_PARALLEL={parallel_value}, OLLAMA_NUM_CTX={num_ctx}")
            click.echo(f"{'='*60}")

            if not await ollama_manager.restart_with_parallel(parallel_value, num_ctx):
                click.echo(
                    f"  ERROR: Failed to start Ollama with parallel={parallel_value}"
                )
                # Create OOM results for all scenarios in this group
                for scenario in group_scenarios:
                    scenario_idx += 1
                    results = ScenarioResults(
                        scenario_name=scenario.name,
                        concurrency=scenario.concurrency,
                        prompt_size=scenario.prompts[0].name if scenario.prompts else "unknown",
                        start_time=datetime.now(),
                        ollama_num_parallel=parallel_value,
                        oom_detected=True,
                    )
                    results.end_time = datetime.now()
                    all_results.append(results)
                continue

            # Get initial memory
            initial_memory = ollama_manager.get_ollama_memory_mb()
            click.echo(f"  Ollama started. Initial memory: {initial_memory:.0f} MB")
            click.echo()

            # Run scenarios for this parallel value
            oom_detected = False
            for scenario in group_scenarios:
                scenario_idx += 1

                if oom_detected:
                    # Skip remaining scenarios if OOM detected
                    results = ScenarioResults(
                        scenario_name=scenario.name,
                        concurrency=scenario.concurrency,
                        prompt_size=scenario.prompts[0].name if scenario.prompts else "unknown",
                        start_time=datetime.now(),
                        ollama_num_parallel=parallel_value,
                        oom_detected=True,
                    )
                    results.end_time = datetime.now()
                    all_results.append(results)
                    click.echo(
                        f"[{scenario_idx}/{total_scenarios}] Skipping {scenario.name} (OOM)"
                    )
                    continue

                click.echo(
                    f"[{scenario_idx}/{total_scenarios}] Running {scenario.name}..."
                )

                try:
                    # Check Ollama health before running
                    if not await ollama_manager.check_health():
                        raise OllamaUnresponsiveError("Ollama not responding")

                    results = await generator.run_scenario(
                        scenario,
                        on_progress=lambda msg: click.echo(f"  {msg}"),
                    )

                    # Update with parallel info and memory
                    results.ollama_num_parallel = parallel_value
                    results.peak_memory_mb = ollama_manager.get_ollama_memory_mb()

                    # Check health after scenario
                    if not await ollama_manager.check_health():
                        results.oom_detected = True
                        oom_detected = True
                        click.echo("  WARNING: Ollama became unresponsive (possible OOM)")

                    all_results.append(results)

                    # Print quick summary
                    summary = results.to_summary()
                    memory_str = (
                        f", Memory={summary['peak_memory_mb']:.0f}MB"
                        if summary['peak_memory_mb']
                        else ""
                    )
                    click.echo(
                        f"  Completed: {summary['successful_requests']}/{summary['total_requests']} "
                        f"requests, TTFT p50={summary['ttft']['p50']:.0f}ms{memory_str}"
                    )

                except OllamaUnresponsiveError:
                    click.echo(f"  ERROR: Ollama unresponsive (OOM detected)")
                    results = ScenarioResults(
                        scenario_name=scenario.name,
                        concurrency=scenario.concurrency,
                        prompt_size=scenario.prompts[0].name if scenario.prompts else "unknown",
                        start_time=datetime.now(),
                        ollama_num_parallel=parallel_value,
                        oom_detected=True,
                    )
                    results.end_time = datetime.now()
                    all_results.append(results)
                    oom_detected = True

                click.echo()

    finally:
        await generator.close()

    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = exporter.export_csv(all_results, f"benchmark_{timestamp}")
    json_path = exporter.export_json(all_results, f"benchmark_{timestamp}")

    click.echo(f"Results exported to:")
    click.echo(f"  CSV:  {csv_path}")
    click.echo(f"  JSON: {json_path}")

    exporter.print_summary(all_results)


@cli.command()
@click.option(
    "--format",
    type=click.Choice(["csv", "json", "both"]),
    default="both",
    help="Export format",
)
@click.option(
    "--input",
    "input_dir",
    default="results",
    help="Input directory with results",
)
@click.option(
    "--output",
    "output_dir",
    default="results",
    help="Output directory",
)
def export(format: str, input_dir: str, output_dir: str) -> None:
    """Export benchmark results."""
    click.echo(f"Export functionality - results already saved during run")
    click.echo(f"Check {output_dir}/ for CSV and JSON files")


@cli.command()
@click.option(
    "--num-ctx",
    default=32768,
    type=int,
    help="Context window size (fixed)",
)
@click.pass_context
def memory_test(ctx: click.Context, num_ctx: int) -> None:
    """Test if prompt length affects KV cache memory.

    Sends prompts of varying lengths and measures memory after each.
    """
    gateway_url = ctx.obj["gateway_url"]

    click.echo("=" * 60)
    click.echo("Prompt Length vs Memory Test")
    click.echo("=" * 60)
    click.echo(f"Gateway: {gateway_url}")
    click.echo(f"Context window: {num_ctx}")
    click.echo()

    asyncio.run(_run_memory_test(gateway_url, num_ctx))


async def _run_memory_test(gateway_url: str, num_ctx: int) -> None:
    """Run memory test with varying prompt lengths."""
    from bench.ollama_manager import OllamaManager

    ollama_manager = OllamaManager()
    ollama_url = "http://localhost:11434"

    # Define prompts of increasing length
    base_text = "The quick brown fox jumps over the lazy dog. "
    prompt_configs = [
        ("tiny", "Say hi.", 2),
        ("small", base_text * 10, 100),
        ("medium", base_text * 100, 1000),
        ("large", base_text * 500, 5000),
        ("xlarge", base_text * 1500, 15000),
        ("xxlarge", base_text * 2500, 25000),
    ]

    # Restart Ollama with fixed settings
    click.echo(f"Restarting Ollama (p=1, ctx={num_ctx})...")
    if not await ollama_manager.restart_with_parallel(num_parallel=1, num_ctx=num_ctx):
        click.echo("ERROR: Failed to start Ollama")
        return

    await asyncio.sleep(2)
    baseline_memory = ollama_manager.get_ollama_memory_mb()
    click.echo(f"Baseline memory (model not loaded): {baseline_memory:.0f} MB")
    click.echo()

    results = []

    # Call Ollama directly to pass num_ctx in options
    async with httpx.AsyncClient(timeout=300.0) as client:
        for name, prompt_text, approx_tokens in prompt_configs:
            click.echo(f"Testing '{name}' prompt (~{approx_tokens} tokens)...")

            try:
                # Call Ollama directly with num_ctx option
                response = await client.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": "qwen2.5:7b-instruct",
                        "prompt": f"Summarize: {prompt_text}\n\nSummary:",
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 10,
                            "num_ctx": num_ctx,
                        },
                    },
                )

                if response.status_code != 200:
                    click.echo(f"  ERROR: HTTP {response.status_code}")
                    continue

            except Exception as e:
                click.echo(f"  ERROR: {e}")
                continue

            await asyncio.sleep(0.5)
            current_memory = ollama_manager.get_ollama_memory_mb()
            click.echo(f"  Memory: {current_memory:.0f} MB")

            results.append({
                "name": name,
                "approx_tokens": approx_tokens,
                "memory_mb": current_memory,
            })

    # Print summary
    click.echo()
    click.echo("=" * 60)
    click.echo("RESULTS SUMMARY")
    click.echo("=" * 60)
    click.echo()
    click.echo(f"{'Prompt':>10} | {'~Tokens':>10} | {'Memory':>10}")
    click.echo("-" * 40)

    for r in results:
        click.echo(f"{r['name']:>10} | {r['approx_tokens']:>10} | {r['memory_mb']:>9.0f}MB")

    if len(results) >= 2:
        first, last = results[0], results[-1]
        token_ratio = last["approx_tokens"] / first["approx_tokens"]
        mem_diff = last["memory_mb"] - first["memory_mb"]

        click.echo()
        click.echo("-" * 40)
        click.echo(f"Tokens increased {token_ratio:.0f}x ({first['approx_tokens']} → {last['approx_tokens']})")
        click.echo(f"Memory difference: {mem_diff:+.0f} MB")

        if abs(mem_diff) < 100:
            click.echo()
            click.echo("CONCLUSION: Memory is roughly constant regardless of prompt length.")
            click.echo("Ollama pre-allocates KV cache based on num_ctx, not actual prompt size.")


@cli.command()
@click.option(
    "--iterations",
    default=10,
    type=int,
    help="Number of requests per test",
)
@click.option(
    "--prompt-size",
    default="large",
    type=click.Choice(["small", "medium", "large"]),
    help="Prompt size to test",
)
@click.pass_context
def cache_test(ctx: click.Context, iterations: int, prompt_size: str) -> None:
    """Test KV cache effect on TTFT.

    Compares TTFT between same prompt (warm cache) vs different prompts (cold cache).
    Runs with concurrency=1 to isolate cache effect from queueing.
    """
    gateway_url = ctx.obj["gateway_url"]

    click.echo("=" * 60)
    click.echo("KV Cache Effect Test")
    click.echo("=" * 60)
    click.echo(f"Gateway: {gateway_url}")
    click.echo(f"Prompt size: {prompt_size}")
    click.echo(f"Iterations: {iterations}")
    click.echo(f"Concurrency: 1 (isolated test)")
    click.echo()

    asyncio.run(_run_cache_test(gateway_url, iterations, prompt_size))


async def _run_cache_test(
    gateway_url: str,
    iterations: int,
    prompt_size: str,
) -> None:
    """Run cache test."""
    config = LoadGeneratorConfig(gateway_url=gateway_url)
    generator = LoadGenerator(config)

    try:
        results = await generator.run_cache_test(
            iterations=iterations,
            prompt_size=prompt_size,
            on_progress=lambda msg: click.echo(msg),
        )
    finally:
        await generator.close()

    click.echo()
    click.echo("=" * 60)
    click.echo("RESULTS")
    click.echo("=" * 60)
    click.echo()
    click.echo(f"First request (cold start): {results['first_request_ttft_ms']:.0f}ms")
    click.echo()
    click.echo("Warm Cache (same prompt repeated):")
    click.echo(f"  Requests:  {results['warm_cache']['count']}")
    click.echo(f"  TTFT mean: {results['warm_cache']['mean']:.0f}ms")
    click.echo(f"  TTFT p50:  {results['warm_cache']['p50']:.0f}ms")
    click.echo(f"  TTFT min:  {results['warm_cache']['min']:.0f}ms")
    click.echo(f"  TTFT max:  {results['warm_cache']['max']:.0f}ms")
    click.echo()
    click.echo("Cold Cache (different prompts):")
    click.echo(f"  Requests:  {results['cold_cache']['count']}")
    click.echo(f"  TTFT mean: {results['cold_cache']['mean']:.0f}ms")
    click.echo(f"  TTFT p50:  {results['cold_cache']['p50']:.0f}ms")
    click.echo(f"  TTFT min:  {results['cold_cache']['min']:.0f}ms")
    click.echo(f"  TTFT max:  {results['cold_cache']['max']:.0f}ms")
    click.echo()
    click.echo("-" * 60)
    click.echo(f"Cache Benefit: {results['cache_benefit_ms']:.0f}ms ({results['cache_benefit_percent']:.1f}% faster)")
    click.echo("-" * 60)


if __name__ == "__main__":
    cli()
