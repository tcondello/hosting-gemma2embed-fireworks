"""
Cost and Efficiency Analyzer for Gemma 2 Embedding on Fireworks AI.
Calculates Dollars per Token ($/token) and Dollars per Million Tokens ($/1M tokens)
based on GPU instance rates, throughput, and batching efficiencies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GPUShape:
    name: str
    hourly_rate: float
    description: str

    @property
    def per_second_rate(self) -> float:
        return self.hourly_rate / 3600.0


# Standard Fireworks AI On-Demand GPU Rates
FIREWORKS_GPU_SHAPES: Dict[str, GPUShape] = {
    "H100_80GB": GPUShape(
        name="NVIDIA H100 80GB (1 GPU)",
        hourly_rate=8.00,
        description="Standard dedicated H100 80GB replica",
    ),
    "H200_141GB": GPUShape(
        name="NVIDIA H200 141GB (1 GPU)",
        hourly_rate=8.00,
        description="High-memory H200 replica for large batch inference",
    ),
    "B200_180GB": GPUShape(
        name="NVIDIA B200 180GB (1 GPU)",
        hourly_rate=13.00,
        description="Blackwell B200 for peak compute & FP4/FP8 speed",
    ),
}

# Market Baselines (Price per 1 Million tokens)
MARKET_BASELINES: Dict[str, float] = {
    "Fireworks Serverless Qwen3-8B": 0.10,
    "OpenAI text-embedding-3-large": 0.13,
    "Google text-embedding-004": 0.025,
    "Voyage-3-large": 0.12,
    "Fireworks Serverless Small (<350M)": 0.016,
}


def calculate_cost_per_token(hourly_rate: float, tokens_per_second: float) -> float:
    """Calculate cost in dollars per single token."""
    if tokens_per_second <= 0:
        return float("inf")
    per_second_rate = hourly_rate / 3600.0
    return per_second_rate / tokens_per_second


def calculate_cost_per_million(hourly_rate: float, tokens_per_second: float) -> float:
    """Calculate cost in dollars per 1 million tokens."""
    return calculate_cost_per_token(hourly_rate, tokens_per_second) * 1_000_000


def calculate_breakeven_throughput(hourly_rate: float, target_cost_per_million: float) -> float:
    """Calculate tokens/sec required to match a target price per 1M tokens."""
    if target_cost_per_million <= 0:
        return float("inf")
    target_cost_per_token = target_cost_per_million / 1_000_000
    per_second_rate = hourly_rate / 3600.0
    return per_second_rate / target_cost_per_token


def generate_cost_table(gpu_key: str = "H100_80GB") -> str:
    gpu = FIREWORKS_GPU_SHAPES.get(gpu_key, FIREWORKS_GPU_SHAPES["H100_80GB"])
    throughput_samples = [
        ("Low utilization (BS=1)", 2_500),
        ("Moderate batching (BS=16)", 15_000),
        ("Standard batching (BS=32)", 35_000),
        ("High batching (BS=64)", 75_000),
        ("Saturated batching (BS=128)", 125_000),
        ("Peak TensorCore / FP8 (BS=256+)", 220_000),
    ]

    lines = []
    lines.append("=" * 88)
    lines.append(f"COST EFFECTIVENESS MODEL: GEMMA-2 EMBEDDING ON FIREWORKS AI")
    lines.append(f"Hardware: {gpu.name} @ ${gpu.hourly_rate:.2f}/hr (${gpu.per_second_rate:.5f}/sec)")
    lines.append("=" * 88)
    lines.append(f"{'Operational Scenario':<32} | {'Tokens/sec':<12} | {'$/Token':<14} | {'$/1M Tokens':<12} | {'Comparison vs Qwen3-8B ($0.10)'}")
    lines.append("-" * 88)

    for scenario, tps in throughput_samples:
        cost_tok = calculate_cost_per_token(gpu.hourly_rate, tps)
        cost_1m = calculate_cost_per_million(gpu.hourly_rate, tps)
        diff_pct = ((cost_1m - 0.10) / 0.10) * 100
        comp = f"{diff_pct:+.1f}%" if diff_pct != 0 else "0.0%"
        status = "CHEAPER" if cost_1m < 0.10 else "MORE EXPENSIVE"
        lines.append(f"{scenario:<32} | {tps:>10,d} | ${cost_tok:.10f} | ${cost_1m:>10.4f} | {status} ({comp})")

    lines.append("=" * 88)
    lines.append("\nMARKET BREAKEVEN ANALYSIS (Required throughput on 1x H100 to beat serverless):")
    lines.append("-" * 88)
    for provider, price_1m in MARKET_BASELINES.items():
        req_tps = calculate_breakeven_throughput(gpu.hourly_rate, price_1m)
        lines.append(f"  • {provider:<35} (${price_1m:.3f}/1M tok): Requires >= {req_tps:>9,.0f} tokens/sec")

    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_cost_table("H100_80GB"))
