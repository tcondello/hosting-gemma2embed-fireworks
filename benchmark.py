"""
High-Performance Benchmark Harness for Embedding Models on Fireworks AI.
Measures latency, throughput, dollars-per-token, and dollars-per-million-tokens
across varying batch sizes and concurrency levels.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import httpx


@dataclass
class BenchmarkRunResult:
    batch_size: int
    num_requests: int
    concurrency: int
    total_tokens: int
    elapsed_time_s: float
    requests_per_second: float
    tokens_per_second: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    dollars_per_token: float
    dollars_per_million_tokens: float
    gpu_hourly_rate: float


def load_api_key(cli_key: Optional[str] = None) -> Optional[str]:
    """Retrieve Fireworks API key from CLI flag, env, or fireworks config."""
    if cli_key:
        return cli_key
    env_key = os.environ.get("FIREWORKS_API_KEY")
    if env_key:
        return env_key

    # Check ~/.fireworks/auth.ini
    auth_path = os.path.expanduser("~/.fireworks/auth.ini")
    if os.path.exists(auth_path):
        try:
            with open(auth_path, "r") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "api_key" and v.strip():
                            return v.strip()
        except Exception:
            pass
    return None


def generate_synthetic_texts(count: int, approx_words_per_text: int = 120) -> List[str]:
    """Generate realistic text sequences of controlled token lengths."""
    vocab_seeds = [
        "Semantic search enables users to find documents based on conceptual meaning rather than exact keywords.",
        "Deep neural networks map high dimensional representations into compact dense latent spaces for similarity ranking.",
        "Large language models utilize transformer self-attention mechanisms to capture complex bidirectional context.",
        "Retrieval augmented generation connects vectorized knowledge repositories with generative reasoning models.",
        "Cost effective deployment on accelerated GPU infrastructure requires maximizing batch saturation and tensor core throughput.",
        "Matryoshka representation learning allows vector truncation without significant performance degradation.",
        "Fireworks AI provides high speed inference engines optimized for low latency and high batch prefill speeds.",
        "Embeddings serve as foundational building blocks for document clustering, classification, and vector database indexing.",
    ]
    texts = []
    for i in range(count):
        # Repeat seeds to reach target approximate word length
        repeats = max(1, approx_words_per_text // 15)
        text = " ".join([vocab_seeds[(i + j) % len(vocab_seeds)] for j in range(repeats)])
        texts.append(f"Document #{i+1}: {text}")
    return texts


async def query_embedding_endpoint(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    batch: List[str],
    mock: bool = False,
) -> Dict[str, Any]:
    """Sends a batch embedding request to Fireworks OpenAI-compatible API or simulates mock."""
    start_t = time.perf_counter()
    if mock:
        # Simulate realistic GPU inference scaling:
        # Base latency ~12ms + 0.15ms per sequence in batch
        simulated_delay = 0.012 + (0.00025 * len(batch))
        await asyncio.sleep(simulated_delay)
        approx_tokens = sum(len(t.split()) * 1.3 for t in batch)
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        return {
            "latency_ms": latency_ms,
            "total_tokens": int(approx_tokens),
            "status": "success",
        }

    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": batch,
    }

    resp = await client.post(url, headers=headers, json=payload, timeout=60.0)
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens") or usage.get("total_tokens")
    if prompt_tokens is None:
        prompt_tokens = int(sum(len(t.split()) * 1.3 for t in batch))

    return {
        "latency_ms": latency_ms,
        "total_tokens": prompt_tokens,
        "status": "success",
    }


async def run_benchmark_for_batch_size(
    batch_size: int,
    num_requests: int,
    concurrency: int,
    model: str,
    base_url: str,
    api_key: str,
    gpu_hourly_rate: float,
    mock: bool = False,
) -> BenchmarkRunResult:
    """Run a load test for a specific batch size across concurrent workers."""
    all_texts = generate_synthetic_texts(batch_size * num_requests)
    batches = [
        all_texts[i * batch_size : (i + 1) * batch_size]
        for i in range(num_requests)
    ]

    sem = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    total_tokens = 0

    limits = httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        async def worker(batch: List[str]):
            nonlocal total_tokens
            async with sem:
                res = await query_embedding_endpoint(
                    client=client,
                    base_url=base_url,
                    api_key=api_key or "",
                    model=model,
                    batch=batch,
                    mock=mock,
                )
                latencies.append(res["latency_ms"])
                total_tokens += res["total_tokens"]

        bench_start = time.perf_counter()
        tasks = [worker(b) for b in batches]
        await asyncio.gather(*tasks)
        total_wall_time = time.perf_counter() - bench_start

    latencies.sort()
    p50 = statistics.median(latencies)
    p90 = latencies[int(len(latencies) * 0.90)] if latencies else 0.0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0

    rps = num_requests / total_wall_time if total_wall_time > 0 else 0.0
    tps = total_tokens / total_wall_time if total_wall_time > 0 else 0.0

    # Dollars per token calculation:
    # Hourly GPU Rate / 3600 = Per-second cost
    # Per-second cost / Tokens per second = Cost per token ($/tok)
    cost_per_second = gpu_hourly_rate / 3600.0
    dollars_per_token = (cost_per_second / tps) if tps > 0 else 0.0
    dollars_per_million = dollars_per_token * 1_000_000.0

    return BenchmarkRunResult(
        batch_size=batch_size,
        num_requests=num_requests,
        concurrency=concurrency,
        total_tokens=total_tokens,
        elapsed_time_s=total_wall_time,
        requests_per_second=rps,
        tokens_per_second=tps,
        p50_latency_ms=p50,
        p90_latency_ms=p90,
        p99_latency_ms=p99,
        dollars_per_token=dollars_per_token,
        dollars_per_million_tokens=dollars_per_million,
        gpu_hourly_rate=gpu_hourly_rate,
    )


def print_results_table(results: List[BenchmarkRunResult]):
    """Pretty prints results in a clean table format."""
    print("\n" + "=" * 115)
    print("BENCHMARK RESULTS: GEMMA 2 EMBEDDING ON FIREWORKS AI (DOLLARS PER TOKEN)")
    print("=" * 115)
    header = (
        f"{'Batch':<6} | {'Reqs':<6} | {'Total Tok':<10} | {'Wall (s)':<8} | "
        f"{'Tokens/s':<11} | {'P50 (ms)':<9} | {'P99 (ms)':<9} | "
        f"{'$/Token':<16} | {'$/1M Tokens':<12}"
    )
    print(header)
    print("-" * 115)
    for r in results:
        row = (
            f"{r.batch_size:<6d} | {r.num_requests:<6d} | {r.total_tokens:<10,d} | {r.elapsed_time_s:<8.2f} | "
            f"{r.tokens_per_second:>10,.0f} | {r.p50_latency_ms:>8.1f} | {r.p99_latency_ms:>8.1f} | "
            f"${r.dollars_per_token:<15.10f} | ${r.dollars_per_million_tokens:>10.4f}"
        )
        print(row)
    print("=" * 115)


def generate_markdown_report(results: List[BenchmarkRunResult], model_name: str) -> str:
    """Generate markdown report for documentation and artifacts."""
    md = []
    md.append(f"# Benchmark & Cost Analysis: {model_name}\n")
    md.append(f"**GPU Hardware Basis:** $8.00 / hour ($0.00222 / second - 1x H100 80GB)\n")
    md.append("| Batch Size | Concurrency | Throughput (Tokens/s) | P50 Latency | P99 Latency | Cost / Token | Cost / 1M Tokens |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        md.append(
            f"| **{r.batch_size}** | {r.concurrency} | {r.tokens_per_second:,.0f} tok/s | "
            f"{r.p50_latency_ms:.1f} ms | {r.p99_latency_ms:.1f} ms | "
            f"`${r.dollars_per_token:.10f}` | **`${r.dollars_per_million_tokens:.4f}`** |"
        )
    md.append("\n### Key Takeaways:")
    best = min(results, key=lambda x: x.dollars_per_million_tokens)
    worst = max(results, key=lambda x: x.dollars_per_million_tokens)
    efficiency_gain = (worst.dollars_per_million_tokens / best.dollars_per_million_tokens) if best.dollars_per_million_tokens > 0 else 1.0

    md.append(f"- **Optimal Batch Size:** Batch size {best.batch_size} yielded **{best.tokens_per_second:,.0f} tokens/sec**, driving cost down to **`${best.dollars_per_million_tokens:.4f} per 1M tokens`**.")
    md.append(f"- **Batching Efficiency Gain:** Moving from batch size {worst.batch_size} to {best.batch_size} reduced cost per token by **{efficiency_gain:.1f}x**.")
    md.append(f"- **Serverless Breakeven:** Beats managed Qwen3-8B serverless ($0.10/1M tokens) as long as throughput exceeds ~22,222 tokens/sec.")
    return "\n".join(md)


async def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models on Fireworks AI")
    parser.add_argument("--model", type=str, default="accounts/fireworks/models/bge-multilingual-gemma2", help="Model name or endpoint ID")
    parser.add_argument("--api-key", type=str, default=None, help="Fireworks API key")
    parser.add_argument("--base-url", type=str, default="https://api.fireworks.ai/inference/v1", help="API base URL")
    parser.add_argument("--batch-sizes", type=str, default="1,8,16,32,64,128", help="Comma-separated list of batch sizes")
    parser.add_argument("--requests-per-batch", type=int, default=10, help="Number of requests to send per batch size")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent requests")
    parser.add_argument("--gpu-rate", type=float, default=8.00, help="GPU hourly cost in USD (default: $8.00 for H100)")
    parser.add_argument("--mock", action="store_true", help="Simulate inference without calling live endpoint")
    parser.add_argument("--output-json", type=str, default="results.json", help="Path to write JSON results")
    parser.add_argument("--output-report", type=str, default="benchmark_report.md", help="Path to write Markdown report")

    args = parser.parse_args()

    api_key = load_api_key(args.api_key)
    if not api_key and not args.mock:
        print("[!] No Fireworks API key found. Use --api-key <key>, export FIREWORKS_API_KEY, or run with --mock for simulation.")
        return

    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()]
    results: List[BenchmarkRunResult] = []

    print(f"\n[*] Starting benchmark for model: {args.model}")
    print(f"[*] Base URL: {args.base_url}")
    print(f"[*] GPU Hourly Rate: ${args.gpu_rate:.2f}/hr")
    print(f"[*] Testing Batch Sizes: {batch_sizes} (Concurrency: {args.concurrency})\n")

    for bs in batch_sizes:
        print(f"--> Running test for batch size {bs:>3d} ({args.requests_per_batch} requests)...", end="", flush=True)
        res = await run_benchmark_for_batch_size(
            batch_size=bs,
            num_requests=args.requests_per_batch,
            concurrency=args.concurrency,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key or "",
            gpu_hourly_rate=args.gpu_rate,
            mock=args.mock,
        )
        results.append(res)
        print(f" Done ({res.tokens_per_second:,.0f} tok/s, ${res.dollars_per_million_tokens:.4f}/1M tok)")

    print_results_table(results)

    # Save JSON results
    with open(args.output_json, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n[+] Raw results saved to {args.output_json}")

    # Save Markdown report
    report_md = generate_markdown_report(results, args.model)
    with open(args.output_report, "w") as f:
        f.write(report_md)
    print(f"[+] Markdown report saved to {args.output_report}\n")


if __name__ == "__main__":
    asyncio.run(main())
