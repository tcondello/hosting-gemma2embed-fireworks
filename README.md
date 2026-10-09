# Hosting Gemma 2 Embedding on Fireworks AI: Performance & Cost-Per-Token Benchmark

This repository contains tools, deployment automation, and a benchmarking suite to build, host, and measure the **Gemma 2 Embedding** model on **Fireworks AI**, with a specific focus on calculating and minimizing **Dollars per Token** ($\$/\text{token}$).

---

## 1. Model Landscape: Understanding "Full Size" Gemma 2 Embedding

When discussing "Gemma 2 Embedding" models, there are two primary architectures:

| Model | Checkpoint / Repo | Size / Parameters | Modalities | Embedding Dim | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemma 2 9B Embedding (Full Size Text)** | [`BAAI/bge-multilingual-gemma2`](https://hf.co/BAAI/bge-multilingual-gemma2) / [`tiendung/gemma2embedding`](https://hf.co/tiendung/gemma2embedding) | **9.24 Billion** | Text & Code (Multilingual) | **3,584** | The standard **full-size Gemma 2 text embedding** model built on `google/gemma-2-9b`. Requires ~18.5 GB VRAM in FP16/BF16. |
| **EmbeddingGemma 2 (Full Multimodal)** | [`google/embeddinggemma-2`](https://hf.co/google/embeddinggemma-2) | **744 Million** | Text, Code, Image, Video, Audio | **768** (MRL to 128/256/512) | Google DeepMind's official multimodal embedding model built on the modern Gemma family (270M text backbone + 170M vision + 300M audio). |

Both models can be staged and deployed to Fireworks AI using this repository.

---

## 2. Fireworks AI Economics & The "Dollars per Token" Formula

Fireworks AI offers two tiers:
1. **Serverless Tier:** Fixed price per input token (e.g., `$0.10` per 1M tokens for Qwen3-8B, `$0.016` per 1M tokens for models < 350M). *Note: Fireworks does not host Gemma 2 embeddings in its serverless catalog.*
2. **On-Demand (Dedicated Custom Models):** Billed strictly by **GPU-seconds active**:
   - **NVIDIA H100 80GB:** **$8.00 / hour** (`$0.1333/min` = **`$0.002222 / second`**)
   - **NVIDIA H200 141GB:** **$8.00 / hour**
   - **NVIDIA B200 180GB:** **$13.00 / hour**

### The Core Math: Calculating Dollars per Token

Because custom models are billed by GPU-time rather than token volume, **your cost per token depends directly on serving throughput (tokens per second):**

$$\text{Cost per Second} = \frac{\text{Hourly GPU Rate}}{3600}$$

$$\text{Cost per Token (\$/token)} = \frac{\text{Cost per Second}}{\text{Throughput (Tokens/sec)}} = \frac{\text{Hourly GPU Rate}}{3600 \times \text{Tokens/sec}}$$

$$\text{Cost per 1M Tokens (\$/1M)} = \text{Cost per Token} \times 1,000,000 \approx \frac{\$2,222.22}{\text{Tokens/sec}} \quad (\text{on 1x H100 @ \$8/hr})$$

### Cost vs. Throughput Curve on 1x H100 ($8.00/hr)

| Operational Scenario | Batch Size | Throughput (Tokens/s) | Dollars / Token | Dollars / 1M Tokens | Comparison vs Serverless 8B ($0.10/1M) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Unbatched / Sparse** | 1 | ~2,500 | `$0.0000008889` | **`$0.8889`** | 8x More Expensive |
| **Small Batch** | 16 | ~15,000 | `$0.0000001481` | **`$0.1481`** | 48% More Expensive |
| **Breakeven Point** | ~24 | **22,222** | **`$0.0000001000`** | **`$0.1000`** | **EXACT BREAKEVEN** |
| **Standard Production** | 32 | ~35,000 | `$0.0000000635` | **`$0.0635`** | **36% CHEAPER** |
| **High Batch** | 64 | ~75,000 | `$0.0000000296` | **`$0.0296`** | **70% CHEAPER** |
| **Saturated Batching** | 128 | ~125,000 | `$0.0000000178` | **`$0.0178`** | **82% CHEAPER** |
| **Peak FP8 / TensorCore**| 256+ | ~220,000+ | `$0.0000000101` | **`$0.0101`** | **90% CHEAPER** |

> **Key Finding:** Once your inference workload exceeds **22,222 tokens/second** (easily achieved with batch sizes of 32 or higher), running a dedicated Gemma 2 embedding deployment on Fireworks is **substantially cheaper** than serverless equivalents. At saturated batching, costs drop below **`$0.02 per 1M tokens`** (`$0.00000002 / token`).

---

## 3. Toolkit Overview

This repository includes:

- [**`cost_analyzer.py`**](file:///Users/tim/Code/hosting-gemma2embed-fireworks/cost_analyzer.py): Interactive modeling script to simulate costs, breakeven throughput, and monthly savings.
- [**`prepare_model.py`**](file:///Users/tim/Code/hosting-gemma2embed-fireworks/prepare_model.py): Tool to inspect Hugging Face repositories, download configs, or download full safetensors checkpoints.
- [**`deploy.sh`**](file:///Users/tim/Code/hosting-gemma2embed-fireworks/deploy.sh): Automated CLI script to register the model with Fireworks (`--embedding`) and create an On-Demand deployment with scale-to-zero.
- [**`benchmark.py`**](file:///Users/tim/Code/hosting-gemma2embed-fireworks/benchmark.py): Async benchmark test harness that executes requests across various batch sizes, measures latency (P50, P90, P99) and throughput, and calculates exact `$/Token` and `$/1M Tokens`.

---

## 4. Quickstart Guide

### Step 1: Environment Setup

```bash
# Clone or enter repository
cd /Users/tim/Code/hosting-gemma2embed-fireworks

# Activate virtual environment
source .venv/bin/activate
```

### Step 2: Run Cost Modeling

Run the standalone cost analysis model:
```bash
python cost_analyzer.py
```

### Step 3: Inspect & Prepare Model Weights

Inspect the Gemma 2 embedding checkpoint:
```bash
python prepare_model.py --target bge-gemma2 --inspect-only
```

To download configs and weights to local storage or cloud bucket:
```bash
# Download config only (fast check)
python prepare_model.py --target bge-gemma2 --config-only

# Or download full 9B safetensors weights (~18 GB)
python prepare_model.py --target bge-gemma2 --download-dir ./model_weights
```

### Step 4: Deploy to Fireworks AI

Use `deploy.sh` to register the model and start an On-Demand deployment on a single H100 with scale-to-zero:

```bash
# If using a cloud bucket (recommended for fast cloud ingestion)
./deploy.sh gemma2-embed s3://my-bucket/models/bge-multilingual-gemma2 https://huggingface.co/BAAI/bge-multilingual-gemma2

# Or if using local directory
./deploy.sh gemma2-embed ./model_weights https://huggingface.co/BAAI/bge-multilingual-gemma2
```

### Step 5: Run Benchmark to Measure Dollars Per Token

Run the benchmark suite against your active endpoint:
```bash
python benchmark.py \
  --model accounts/<YOUR_ACCOUNT_ID>/models/gemma2-embed \
  --batch-sizes "1,8,16,32,64,128,256" \
  --requests-per-batch 20 \
  --concurrency 4 \
  --gpu-rate 8.00
```

*(Note: To test the benchmark logic and simulated curves immediately without spending credits, use the `--mock` flag):*
```bash
python benchmark.py --mock --batch-sizes "1,8,16,32,64,128,256"
```

---

## 5. Cost-Optimization Best Practices on Fireworks

1. **Enable Scale-to-Zero:** Use `--min-replica-count 0 --scale-to-zero-window 10m` so the deployment spins down when idle.
2. **Client-Side Batching:** Never send single sentences one-by-one. Accumulate embeddings in micro-batches (e.g. 32 to 128 sequences) or use background buffer queues before dispatching to Fireworks.
3. **Use FP8 / BF16 Precision:** A 9B model in FP8 fits comfortably in ~9.5 GB of GPU VRAM, allowing massive KV/batch parallelism and doubling tokens/sec.
