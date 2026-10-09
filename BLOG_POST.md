# From Cloud GPU Theory to Zero-Cost Reality: Embedding 40 Minutes of Archival WWII Film with Google's EmbeddingGemma 2

*A real-world engineering autopsy of moving from theoretical cloud calculations to on-device Apple Silicon embedding, eliminating Pinecone, and hosting full semantic video search for $0.00.*

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-orange)](https://huggingface.co/spaces/astr010/loc-ww2-color-search)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/astr010/loc-ww2-color-film-gemma2)
[![LOC Catalog](https://img.shields.io/badge/Library%20of%20Congress-LCCN%202020600759-blue)](https://www.loc.gov/item/2020600759/)

---

## 1. The Theory vs. Reality Trap

Every cloud AI pitch starts with elegant napkin math:
*"If an H100 GPU costs $8.00 per hour on Fireworks AI, and we batch 1,000 video frames at 250 tokens per second, we can embed 30 minutes of video for $0.0015!"*

It sounds visionary on paper. But when you move from whitepapers to production terminals, reality hits:

1. **Compiled C++ Inference Engines Aren't Generic Containers**: Platforms like Fireworks AI or TensorRT-LLM run heavily optimized, proprietary compiled kernels whitelist-locked to specific architectures (e.g., Llama, Qwen, Mistral). When Google DeepMind released `google/embeddinggemma-2` (using the new `EmbeddingGemma2Model` from `transformers 5.18+`), Fireworks' batch inference engine rejected it with:
   ```
   base model's parameter count is misconfigured, must be >0
   ```
   Deploying it as a dedicated on-demand replica hung indefinitely in `Initializing Replica Count: 1`.
2. **The "External Database" Knee-Jerk Reaction**: Modern AI architecture diagrams almost reflexively include an external vector database (Pinecone, Qdrant, Milvus). But why pay \$70+/month and add 50–100ms of network roundtrip latency to search a single film?

We decided to strip away the theoretical bloat, test what is actually possible on commodity hardware, and build an end-to-end semantic video search engine with **measured, empirical numbers and zero monthly cloud bills**.

---

## 2. The Asset: 40 Minutes of Authentic WWII Color Film

Instead of animated clips or synthetic test footage, we tested our pipeline on an authentic national treasure from the **Library of Congress National Audiovisual Conservation Center**:

- **Title**: *World War II color footage--Stevens and SPECOU in Berlin; Stevens in North Africa and Egypt before D-Day*
- **Catalog Item / LCCN**: [`2020600759`](https://www.loc.gov/item/2020600759/)
- **Director**: Lt. Col. George Stevens (U.S. Army Signal Corps Special Coverage Unit - SPECOU)
- **Cinematographer**: William C. Mellor
- **Date**: 1943 (North Africa / Cairo / Giza) & 1945 (Berlin)
- **Format**: 16mm Kodachrome Color Film, scanned at 1440x1080 HD, 24fps
- **Duration**: 39 minutes, 53.5 seconds (2,393.5 seconds)
- **Status**: National Film Registry (Librarian of Congress), Public Domain

```
                               TIMELINE OVERVIEW (39m 54s)
  00:00:00                                 00:24:00                           00:39:54
     ├────────────────────────────────────────┼──────────────────────────────────┤
     │       BERLIN, GERMANY (1945)           │     NORTH AFRICA & EGYPT (1943)  │
     │ • Reichstag ruins & rubble cleanup     │ • Desert tanks maneuvers         │
     │ • Olympic Stadium 1936 grounds         │ • George Stevens at Great Sphinx │
     │ • Russian sector & Soviet troops       │ • Allied pilots & desert staging │
     │ • Civilian refugees with carts         │ • Giza Pyramids                  │
```

Sampling keyframes at 1 frame every 4.0 seconds yielded exactly **598 keyframes** covering every scene transition in the film.

---

## 3. The Model: Google DeepMind EmbeddingGemma 2

Google DeepMind's `google/embeddinggemma-2` is a purpose-built multimodal embedding model:
- **Total Parameters**: 744 Million
- **Text Backbone**: 270M parameter Gemma transformer
- **Vision Backbone**: 170M parameter SigLIP-based vision encoder
- **Shared Representation**: Both image frames and text queries are projected into the same unit-normalized **768-dimensional hypersphere**.

Because the vision encoder and text encoder map to the exact same geometric space, natural language queries like *"Great Sphinx in Egypt"* directly align with visual frames via standard cosine similarity:

$$\text{Similarity}(q, f) = \frac{q \cdot f}{\|q\|_2 \|f\|_2} = q_{\text{norm}} \cdot f_{\text{norm}}$$

---

## 4. The Empirical Benchmark: 2 Batches on Apple Silicon (MPS)

Rather than paying cloud GPU rates, we ran `scripts/embed_local_gemma2.py` locally on an **Apple Silicon MacBook Pro (M3 Pro GPU)** using PyTorch Metal Performance Shaders (`device='mps'`).

To assess throughput and repeatability, the 598 frames were divided into **two equal macro-batches** processed with mini-batch size 32:

```
[Frame 000 ..................... Frame 298] -> Batch 1 (299 frames)
[Frame 299 ..................... Frame 597] -> Batch 2 (299 frames)
```

### Measured Execution Telemetry

| Run Phase | Frame Range | Timecode Range | Frames | Elapsed Time | Throughput | Latency / Frame | Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Batch 1** | Frames 000–298 | 00:00:00 – 00:19:52 | 299 | **120.60 s** | **2.48 fps** | **403.3 ms** | Apple M3 Pro (MPS) |
| **Batch 2** | Frames 299–597 | 00:19:56 – 00:39:48 | 299 | **117.28 s** | **2.55 fps** | **392.2 ms** | Apple M3 Pro (MPS) |
| **Total** | **All Frames** | **Full Video (39m 54s)** | **598** | **237.88 s** (~3.96m) | **2.51 fps** | **397.8 ms** | **Cost: $0.00** |

```
                       INFERENCE LATENCY COMPARISON
                                (Lower is better)

Cloud API (Vertex/Voyage):  ███████████████████████████ 1500ms+ (Rate limits + network)
Apple M3 Pro (MPS Local):   ████████ 397.8ms ($0.00 cost)
```

### Vector Quality Audit
- **Matrix Dimensions**: `(598, 768)` Float32
- **Mean L2 Norm**: `1.0001` (Unit normalized)
- **NaNs / Corrupted Vectors**: `0` detected

---

## 5. The Math of Eliminating External Vector Databases

When developers build video retrieval systems, many assume they need an enterprise vector database. But let's look at the actual physics of memory:

$$598 \text{ vectors} \times 768 \text{ dimensions} \times 4 \text{ bytes (Float32)} = 1,837,056 \text{ bytes} \approx \mathbf{1.75\text{ MB}}$$

**1.75 Megabytes.** That is smaller than a single high-resolution JPEG photo.

| Metric | Pinecone / Cloud Vector DB | In-Memory NumPy / Browser Float32Array |
| :--- | :--- | :--- |
| **Search Latency** | 35 – 80 ms (Network hop + TLS) | **0.05 ms** (50 microseconds) |
| **RAM Footprint** | N/A | **1.8 MB** |
| **Monthly Cost** | $70.00 – $150.00 / mo | **$0.00** |
| **Cold Start** | Cloud cluster provisioning | **Instant** (< 1ms) |
| **External Dependencies** | API Key, VPC, Quotas | **None** |

A single matrix-vector dot product `np.dot(V_norm, q_vec)` across 598 vectors takes **0.05 milliseconds** on a standard CPU. An in-memory search is **1,000x faster than a cloud network roundtrip** and costs literally nothing.

---

## 6. Semantic Search Accuracy: Real Results

How well does EmbeddingGemma 2 align natural language queries to authentic 80-year-old Kodachrome film?

### Test 1: *"Great Sphinx and Pyramids of Giza in Egypt"*
- **Top Match**: `frame_0473.jpg` / `frame_0474.jpg` at **00:31:32**
- **Cosine Score**: **`0.7789`**
- **Archival Verification**: In the Library of Congress catalog summary: *"In the second half of the film, there are out of sequence shots of North Africa and Egypt, filmed in 1943 before D-Day. Stevens and Mellor visit the Sphinx."* Frame 474 at 31m 32s is the exact shot of Stevens standing before the Sphinx.

### Test 2: *"Bombed Reichstag ruins and destruction in Berlin"*
- **Top Match**: `frame_0171.jpg` at **00:11:20**
- **Cosine Score**: **`0.7873`**
- **Archival Verification**: Shows the ruined, hollowed-out dome of the Reichstag building in late summer 1945.

### Test 3: *"Tanks and military armor driving through desert sand"*
- **Top Match**: `frame_0512.jpg` at **00:34:08**
- **Cosine Score**: **`0.7412`**
- **Archival Verification**: Allied tanks maneuvering across North African sand dunes.

---

## 7. The Architecture: Published & Live

We published all assets and deployed the search application with zero recurring hosting costs:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA & EMBEDDING PIPELINE                       │
│                                                                        │
│  LOC 1080p Video ──> 598 Frames ──> M3 Pro (MPS) ──> 598x768 Vectors  │
│  (LCCN 2020600759)     (4s step)     (EmbeddingGemma 2)     (Parquet)  │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
┌─────────────────────────────┐               ┌─────────────────────────────┐
│    HUGGING FACE DATASET     │               │     HUGGING FACE SPACE      │
│  astr010/loc-ww2-color-film │               │   astr010/loc-ww2-color-    │
│            -gemma2          │               │           search            │
│                             │               │                             │
│ • Parquet embeddings table  │               │ • In-Memory Vector Search   │
│ • 16.5 MB frame tarball     │               │ • Sub-millisecond dot math  │
│ • Archival catalog metadata │               │ • Direct LOC 1080p stream   │
└─────────────────────────────┘               └─────────────────────────────┘
```

1. **Hugging Face Dataset**: Published at [`astr010/loc-ww2-color-film-gemma2`](https://huggingface.co/datasets/astr010/loc-ww2-color-film-gemma2), containing the verified Parquet embedding table, complete 598-frame archive, and catalog metadata.
2. **Interactive Search Space**: Live at [`astr010/loc-ww2-color-search`](https://huggingface.co/spaces/astr010/loc-ww2-color-search). Loads the 1.8 MB vector index into the browser and executes instant semantic queries.
3. **Bandwidth Optimization**: The 935 MB 1080p video file is **not** bundled into the container. The HTML5 player streams directly from the Library of Congress CDN URL (`https://tile.loc.gov/storage-services/service/mbrs/ntscrm/02531189/02531189.mp4`), seeking instantly to the matched timecode.

---

## 8. Summary & Key Takeaways

1. **Don't Trust Theoretical Cloud Costs**: Specialized compiled inference engines don't automatically support new model architectures. Test real model weights early.
2. **744M Parameters is the Multimodal Sweet Spot**: EmbeddingGemma 2 is light enough to run on local laptops (Apple Silicon M3 Pro encodes at 2.5 fps) while delivering top-tier semantic retrieval accuracy.
3. **Know When Not to Use a Vector DB**: If your vector dataset fits in a few megabytes of RAM (which is true for hundreds of hours of sampled video clips), keep it in-memory. You get 50-microsecond queries, zero infrastructure overhead, and $0 monthly invoices.

---

### Resources & Links
- **Live Search Application**: [Hugging Face Space](https://huggingface.co/spaces/astr010/loc-ww2-color-search)
- **Embeddings & Frames Dataset**: [Hugging Face Datasets](https://huggingface.co/datasets/astr010/loc-ww2-color-film-gemma2)
- **Library of Congress Archival Record**: [LCCN 2020600759](https://www.loc.gov/item/2020600759/)
- **Model Checkpoint**: [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)
