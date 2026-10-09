# Benchmark & Cost Analysis: accounts/fireworks/models/bge-multilingual-gemma2

**GPU Hardware Basis:** $8.00 / hour ($0.00222 / second - 1x H100 80GB)

| Batch Size | Concurrency | Throughput (Tokens/s) | P50 Latency | P99 Latency | Cost / Token | Cost / 1M Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 4 | 35,503 tok/s | 13.4 ms | 13.4 ms | `$0.0000000626` | **`$0.0626`** |
| **32** | 4 | 712,247 tok/s | 21.3 ms | 21.4 ms | `$0.0000000031` | **`$0.0031`** |
| **128** | 4 | 1,308,987 tok/s | 45.9 ms | 46.7 ms | `$0.0000000017` | **`$0.0017`** |

### Key Takeaways:
- **Optimal Batch Size:** Batch size 128 yielded **1,308,987 tokens/sec**, driving cost down to **`$0.0017 per 1M tokens`**.
- **Batching Efficiency Gain:** Moving from batch size 1 to 128 reduced cost per token by **36.9x**.
- **Serverless Breakeven:** Beats managed Qwen3-8B serverless ($0.10/1M tokens) as long as throughput exceeds ~22,222 tokens/sec.