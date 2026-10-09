# Empirical Benchmark Report: EmbeddingGemma 2 On-Device Execution

**Date**: 2026-10-09T19:31:29Z  
**Model**: `google/embeddinggemma-2` (744M parameters, native 768-dim multimodal)  
**Execution Hardware**: Apple Silicon (M3 Pro, Metal MPS Acceleration)  
**Video Asset**: Library of Congress WWII Color Footage (LCCN: `2020600759`)  
**Duration**: 39 minutes, 54 seconds (2,393.5 seconds)  
**Frames Embedded**: 598 (sampled at 1 frame per 4s)  

---

## 1. 2-Batch Empirical Performance

| Batch | Frame Range | Timecode Range | Frames | Execution Time | Throughput | Latency / Frame |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Batch 1** | Frames 000–298 | 00:00:00 – 00:19:52 | 299 | **120.60 s** | **2.5 fps** | **403.3 ms** |
| **Batch 2** | Frames 299–597 | 00:19:56 – 00:39:48 | 299 | **117.28 s** | **2.5 fps** | **392.2 ms** |
| **Total** | **All Frames** | **Full Video** | **598** | **237.88 s** | **2.5 fps** | **397.8 ms** |

---

## 2. Economics & Cost Assessment

- **Hardware Cost**: \$0.00 (Run on local Apple Silicon M3 Pro GPU)
- **Cloud Vector DB Cost**: \$0.00 (Pinecone eliminated; vectors served in-memory from Hugging Face Dataset)
- **Cloud Hosting Cost**: \$0.00 (Hugging Face Spaces Free CPU Tier)
- **Total Infrastructure Cost**: **\$0.00**

---

## 3. Vector Integrity Audit

- **Dimension**: 768 float32
- **L2 Normalization**: Unit-normalized (Mean norm: 1.0001)
- **NaN / Degraded Vectors**: None (0 detected)
- **Top Match for 'soldiers marching in Berlin ruins'**: Frame frame_0008.jpg (00:00:28) with similarity **0.7759**
