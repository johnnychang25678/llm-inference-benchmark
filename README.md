# LLM Inference Benchmark

A tiny LLM inference benchmark project. I built this to learn about LLM inferencing performance characteristics, especially around parallelism and context window size. 

This project uses [Ollama](https://ollama.com/) as the LLM backend, specifically the `qwen2.5:7b-instruct` model, `FastAPI` for the gateway, and `Prometheus` + `Grafana` for observability. It can be run on Apple Silicon (M1/M2/M4) Macs.

See the benchmark results and my learnings in the [Benchmark Results and Findings](#benchmark-results-and-findings) section below.

## Quick Start

```bash
# 1. Install Ollama (Mac)
brew install ollama

# 2. Serve ollama
ollama serve

# 3. Pull the model
ollama pull qwen2.5:7b-instruct

# 4. Install Python dependencies
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# 5. Start the gateway
python -m server.main

# 6. Start observability stack (Prometheus + Grafana)
docker-compose up -d

# 7. Verify health
curl http://localhost:8000/health

# 8. View Grafana
open http://localhost:3000  # admin/admin

# 9. (Optional) Chat with the LLM model via web UI
open http://localhost:3001
```


## Benchmarking

### 1. Start Services
```bash
# Terminal 1: Gateway
python -m server.main

# Terminal 2: Observability
docker-compose up -d
```

### 2. Run Benchmark Matrix with Parallel Testing
```bash
# Test impact of OLLAMA_NUM_PARALLEL on throughput and memory
bench run --concurrency 1,4 --prompt-sizes small,medium,large,xlarge --ollama-num-parallel 1,2,4 --duration 60

# Use --vary-prompts to test without cache benefits (isolates true parallelism)
bench run --concurrency 4 --prompt-sizes large --ollama-num-parallel 1,4 --duration 30 --vary-prompts
```
The benchmark will:
- Restart Ollama with each `--ollama-num-parallel` value
- Track memory usage and detect OOM
- Export results with parallel setting in CSV

Show Grafana (`http://localhost:3000`) - TTFT and tokens/s panels.

## Benchmark Options

### Prompt Sizes

| Size | Description | Use Case |
|------|-------------|----------|
| `small` | Short response (~10 words) | Quick latency tests |
| `medium` | Medium response (3 paragraphs) | General benchmarking |
| `large` | Detailed architecture design | Stress testing, long generation |
| `xlarge` | Comprehensive system design | Maximum context/generation testing |

### Flags

| Flag | Description |
|------|-------------|
| `--concurrency` | Comma-separated concurrent request levels |
| `--prompt-sizes` | Comma-separated prompt sizes (small,medium,large,xlarge) |
| `--ollama-num-parallel` | Comma-separated OLLAMA_NUM_PARALLEL values to test |
| `--ollama-num-ctx` | Context window size (default: 32768 for qwen2.5:7b) |
| `--duration` | Seconds per scenario |
| `--vary-prompts` | Use different prompts to avoid KV cache benefits |

## Project Structure

```
├── server/           # Gateway (routing, metrics)
├── engines/          # Adapters (ollama, vllm stub, sglang stub)
├── bench/            # Load generator, Ollama manager, CLI
├── dashboards/       # Grafana JSON
├── prometheus/       # Prometheus config
├── tests/            # pytest tests
```

# Benchmark Results and Findings

- **Hardware**: Apple M4 Max, 64GB RAM
- **Model**: qwen2.5:7b-instruct
- **Context Window**: 4096 tokens
- **Test Duration**: 60s per scenario

**Abbreviations**:
- c=N: concurrency N (N simultaneous users)
- p=N: OLLAMA_NUM_PARALLEL=N (N parallel slots)
- TTFT: Time To First Token (latency until first token is received)

---

## Benchmark Results Executive Summary

| Metric | p=1 | p=4 | Winner |
|--------|-----|-----|--------|
| TTFT (c=4, large prompt) | 20,293ms | 118ms | **p=4** (99% better) |
| Tokens/sec (c=4) | 75 tok/s | 24 tok/s | p=1 (3x faster) |
| Throughput (c=4, small) | 9.24 req/s | 10.02 req/s | **p=4** (8% better) |
| Memory | 5,250 MB | 5,970 MB | p=1 (720MB less) |

---

## 1. The Queueing Problem

### TTFT at Concurrency=4 (4 simultaneous users)

| Prompt Size | p=1 (queue) | p=2 | p=4 (parallel) | Improvement |
|-------------|-------------|-----|----------------|-------------|
| small | 327ms | 240ms | **120ms** | 63% |
| medium | **8,876ms** | 5,003ms | **131ms** | **98.5%** |
| large | **20,293ms** | 11,070ms | **118ms** | **99.4%** |
| xlarge | **20,149ms** | 11,117ms | **119ms** | **99.4%** |

### Why does this happen?

With `OLLAMA_NUM_PARALLEL=1`, Ollama processes requests sequentially:

```
p=1, large prompt, 4 concurrent requests:
┌──────────────────────────────────────────────────────────────────────┐
│ Req1: [TTFT 60ms]======Generation 7s======                           │
│ Req2:              [wait 7s...][TTFT]======Generation======          │
│ Req3:                          [wait 14s......][TTFT]====Gen====     │
│ Req4:                                     [wait 21s.........][TTFT]  │
└──────────────────────────────────────────────────────────────────────┘
       ↑ Request 4 waits ~20 seconds before getting first token
```

With `OLLAMA_NUM_PARALLEL=4`, all requests start immediately:

```
p=4, large prompt, 4 concurrent requests:
┌──────────────────────────────────────────────────────────────────────┐
│ Req1: [TTFT 120ms]==========Generation (slower, ~21s)=========       │
│ Req2: [TTFT 120ms]==========Generation==========                     │
│ Req3: [TTFT 120ms]==========Generation==========                     │
│ Req4: [TTFT 120ms]==========Generation==========                     │
└──────────────────────────────────────────────────────────────────────┘
       ↑ All requests start within ~120ms
```

---

## 2. Single Request Performance (c=1)

When there's no contention, TTFT is identical regardless of parallel setting:

| Prompt Size | p=1 | p=2 | p=4 |
|-------------|-----|-----|-----|
| small | 57ms | 57ms | 57ms |
| medium | 59ms | 60ms | 59ms |
| large | 61ms | 61ms | 60ms |
| xlarge | 62ms | 61ms | 61ms |

**Takeaway**: Parallelism only matters under concurrent load.

---

## 3. Tokens/sec: The Parallelism Tradeoff between Speed and Throughput

Per-request generation speed drops as GPU is shared:

| Scenario (c=4) | p=1 | p=2 | p=4 | Slowdown |
|----------------|-----|-----|-----|----------|
| small | 86 tok/s | 52 tok/s | 30 tok/s | 2.9x |
| medium | 75 tok/s | 46 tok/s | 25 tok/s | 3.0x |
| large | 75 tok/s | 46 tok/s | 24 tok/s | 3.1x |
| xlarge | 76 tok/s | 46 tok/s | 24 tok/s | 3.2x |

At c=1 (single request), tokens/sec is consistent ~75-88 tok/s regardless of parallel setting.

### Why the slowdown for single requests?

GPU compute units are time-sliced across parallel requests. Each request gets ~1/N of the GPU attention.

### System Throughput

Despite slower per-request speed, total throughput improves with parallelism:

#### Small prompts (c=4)

| Parallel | Total Requests | Requests/sec |
|----------|----------------|--------------|
| p=1 | 562 | 9.24 |
| p=2 | 594 | 9.78 |
| p=4 | 608 | **10.02** |

### Why throughput improves?

Basically, while each request is slower, more requests are being processed simultaneously, leading to higher overall completion rates. For this test setup, we are not exhausting GPU capacity, so parallelism yields better utilization.

- p=1: GPU idles between request context switches
- p=4: GPU stays busy processing all requests simultaneously

---

## 5. Memory Scaling

| Parallel | Peak Memory | Delta from p=1 |
|----------|-------------|----------------|
| p=1 | ~5,250 MB | baseline |
| p=2 | ~5,500 MB | +250 MB |
| p=4 | ~5,970 MB | +720 MB |

Each parallel slot allocates additional KV cache memory (~250MB per slot).

### Model Weights + Runtime Overhead
- qwen2.5:7b weights (Q4_K_M quantized): 7B × (0.5 bytes + metadata overhead ~= 0.7 bytes) = ~4.8 GB
- Runtime overhead: ~0.2 GB

### KV Cache Size Calculation

#### Formula

```
KV Cache = 2 × num_layers × num_kv_heads × head_dim × context_length × bytes_per_value
```
- 2 for Key + Value
- bytes_per_value = 2 (fp16)

#### With qwen2.5:7b and 4096 context window (default of Ollama):

```= 2 × 28 × 4 × 128 × 4096 × 2
= 234,881,024 bytes
= 224 MB
```
With addtional overhead, rounds to ~250MB per slot. Matches observed memory usage.

---

## 6. Prompt Size Observations

### Large ≈ XLarge Performance

| Metric | large | xlarge |
|--------|-------|--------|
| Latency (c=1) | 6,936ms | 6,754ms |
| Latency (c=4, p=4) | 21,277ms | 21,339ms |
| Tokens/sec | 74-75 | 75-76 |

Both produce similar output lengths (~500 tokens). The xlarge prompt's longer input doesn't significantly affect generation.

### Output tokens by prompt size (estimated)

| Prompt | Latency (c=1) | Est. Output Tokens |
|--------|---------------|-------------------|
| small | 148ms | ~10-15 |
| medium | 2,866ms | ~200-250 |
| large | 6,936ms | ~500-550 |
| xlarge | 6,754ms | ~500-550 |

---

## Raw Data

### TTFT (p50) by Configuration

| Config | small | medium | large | xlarge |
|--------|-------|--------|-------|--------|
| c1_p1 | 57ms | 59ms | 61ms | 62ms |
| c4_p1 | 327ms | 8,876ms | 20,293ms | 20,149ms |
| c1_p2 | 57ms | 60ms | 61ms | 61ms |
| c4_p2 | 240ms | 5,003ms | 11,070ms | 11,117ms |
| c1_p4 | 57ms | 59ms | 60ms | 61ms |
| c4_p4 | 120ms | 131ms | 118ms | 119ms |

### Latency (p50) by Configuration

| Config | small | medium | large | xlarge |
|--------|-------|--------|-------|--------|
| c1_p1 | 148ms | 2,866ms | 6,936ms | 6,754ms |
| c4_p1 | 422ms | 11,759ms | 27,017ms | 26,843ms |
| c1_p2 | 149ms | 3,128ms | 6,963ms | 6,930ms |
| c4_p2 | 396ms | 9,929ms | 22,127ms | 22,137ms |
| c1_p4 | 149ms | 3,048ms | 6,858ms | 6,741ms |
| c4_p4 | 390ms | 9,864ms | 21,277ms | 21,339ms |

### Tokens/sec (mean) by Configuration

| Config | small | medium | large | xlarge |
|--------|-------|--------|-------|--------|
| c1_p1 | 88.7 | 83.8 | 74.4 | 75.9 |
| c4_p1 | 85.9 | 75.0 | 75.1 | 76.4 |
| c1_p2 | 87.9 | 77.1 | 73.1 | 74.6 |
| c4_p2 | 51.5 | 46.4 | 46.2 | 46.2 |
| c1_p4 | 87.3 | 77.0 | 75.5 | 76.6 |
| c4_p4 | 30.2 | 24.7 | 24.2 | 24.1 |

### Memory (MB) by Parallel Setting

| Parallel | Peak Memory |
|----------|-------------|
| p=1 | 5,138 - 5,265 |
| p=2 | 5,382 - 5,514 |
| p=4 | 5,827 - 5,976 |

---

## 9. KV Cache Deep Dive

### What is KV Cache?

In transformer models, generating each token requires attending to all previous tokens. This involves computing Key (K) and Value (V) matrices for the entire prompt. KV caching stores these matrices so they don't need to be recomputed.

### Isolated Cache Test Results

Test configuration: `bench cache-test --iterations 10 --prompt-size large`
- Concurrency: 1 (no queueing effects)
- Parallelism: 1 (single slot)

| Scenario | TTFT |
|----------|------|
| First request (model cold start) | 897ms |
| Warm cache (same prompt repeated) | 59ms |
| Cold cache (different prompts) | 106ms |
| **Cache benefit** | **47ms (44% faster)** |

### How Ollama's KV Cache Works

Ollama (via llama.cpp) uses **prefix-based cache reuse**, not just exact matching:

```
Request 1: "System: You are helpful. User: What is 2+2?"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           Full KV cache computed and stored in slot

Request 2: "System: You are helpful. User: What is 3+3?"
           ^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^
           Reuse this prefix KV      Only recompute suffix
```

Key behaviors:
- **`cache_prompt=true` (default)**: Re-uses KV cache from previous request when prefixes match
- **Per-slot caching**: Each parallel slot maintains its own KV cache state
- **Last-prompt caching**: Each slot caches only the most recent prompt (not multiple prompts)


### When KV Cache Helps

1. **Chat conversations**: System prompt + history is cached, only new user message needs processing
2. **Batch processing same prompt**: Running identical prompt multiple times
3. **Shared prefix workloads**: Multiple prompts with common system instructions
