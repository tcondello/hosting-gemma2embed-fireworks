# Video Analysis & Multimodal Embedding Plan

## 1. Video Profile

- **Filename:** `Rick and Morty ｜ Season 9 Battle Scenes ｜ adult swim.webm`
- **Duration:** **30m 4s** (1804.2 seconds)
- **Resolution:** **854 x 480**
- **Native Framerate:** 23.98 fps (~43,257 total native frames)
- **Codecs:** Video: `av1` | Audio: `opus`
- **File Size:** 122.6 MB

## 2. Comparison of Embedding Strategies

| Metric | Plan A (8s Action Clips) | Plan B (2s Storyboard) | Plan C (15s Video + Audio) |
| :--- | :--- | :--- | :--- |
| **Embedding Target** | Temporal Action Retrieval | Fine-Grained Keyframes | Visual + Audio Speech |
| **Total Segments / Units** | **226 clips** | **903 keyframes** | **121 clips** |
| **Frames per Unit** | 8 frames | 1 frame | 15 frames |
| **Tokens per Unit** | 1,120 tokens | 280 tokens | 2,475 tokens |
| **Total Token Footprint** | **253,120 tokens** | **252,840 tokens** | **299,475 tokens** |
| **Batches at BS=256** | **1 batch** | **4 batches** | **1 batch** |
| **GPU Inference Time** | **0.70 seconds** | **0.70 seconds** | **0.83 seconds** |
| **Cost (Fireworks H100)** | **`$0.00156`** | **`$0.00156`** | **`$0.00185`** |
| **Cost (Google Vertex AI)** | `$0.0601` | `$0.2258` | `$0.0601` |
| **Cost (Voyage Multimodal)**| `$0.0304` | `$0.0303` | `$0.0359` |
| **Fireworks Cost Advantage**| **97.4% cheaper** | **99.3% cheaper** | **96.9% cheaper** |

## 3. Extracted Sample Keyframes

Extracted 9 sample keyframes across the 30-minute episode for visual verification:

- `keyframe_0010s.jpg` (68.0 KB)
- `keyframe_0060s.jpg` (84.2 KB)
- `keyframe_0180s.jpg` (104.4 KB)
- `keyframe_0300s.jpg` (90.4 KB)
- `keyframe_0600s.jpg` (72.3 KB)
- `keyframe_0900s.jpg` (34.7 KB)
- `keyframe_1200s.jpg` (98.9 KB)
- `keyframe_1500s.jpg` (121.9 KB)
- `keyframe_1750s.jpg` (83.6 KB)

## 4. Key Takeaways & Recommended Implementation Plan

1. **The Entire 30-Minute Video Fits in 1 Batch:** Under Plan A (8-second clips), the entire 30-minute episode generates **226 clips** (`253,120 tokens`), which fits entirely inside a single batch request at **Batch Size = 256**!
2. **Incredible Economics:** Processing the whole 30 minutes on a dedicated H100 takes **under 1 second of GPU time** and costs **`$0.00156` (approx. 0.16 cents)**.
3. **Commercial Discrepancy:** Compared to Google Vertex AI Multimodal Embedding (`$0.060`), dedicated Gemma 2 on Fireworks is **~38x cheaper** for this video.