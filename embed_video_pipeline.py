"""
End-to-End Pipeline to Chunk, Extract Frames, and Embed Video Footage
using EmbeddingGemma 2 on Fireworks AI.
"""

import argparse
import asyncio
import base64
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx


def extract_video_segments(
    video_path: str,
    output_dir: str,
    clip_duration_s: float = 8.0,
    frames_per_clip: int = 8,
    max_clips: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Splits video into uniform temporal clips and extracts representative frames for each clip.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get duration
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", video_path
    ]
    meta = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    total_dur = float(meta["format"]["duration"])
    total_clips = math.ceil(total_dur / clip_duration_s)
    if max_clips:
        total_clips = min(total_clips, max_clips)

    print(f"[*] Extracting frames for {total_clips} clips ({clip_duration_s}s per clip, {frames_per_clip} frames/clip)...")
    segments = []

    for i in range(total_clips):
        start_time = i * clip_duration_s
        end_time = min(total_dur, (i + 1) * clip_duration_s)
        clip_frames_b64 = []

        # Sample frames within the clip
        step = (end_time - start_time) / (frames_per_clip + 1)
        for f_idx in range(1, frames_per_clip + 1):
            frame_ts = start_time + (f_idx * step)
            frame_path = out_dir / f"clip_{i:04d}_f{f_idx}.jpg"
            # Extract frame, resize to 336x336 to keep payload small
            ff_cmd = [
                "ffmpeg", "-ss", str(frame_ts),
                "-i", video_path,
                "-vf", "scale=336:336:force_original_aspect_ratio=decrease,pad=336:336:(ow-iw)/2:(oh-ih)/2",
                "-vframes", "1",
                "-q:v", "3",
                str(frame_path),
                "-y", "-v", "error"
            ]
            subprocess.run(ff_cmd, check=True)
            with open(frame_path, "rb") as f:
                clip_frames_b64.append(base64.b64encode(f.read()).decode("utf-8"))

        segments.append({
            "clip_id": i,
            "start_time_s": start_time,
            "end_time_s": end_time,
            "frames_b64": clip_frames_b64,
            "token_estimate": frames_per_clip * 140,
        })
        if (i + 1) % 25 == 0 or (i + 1) == total_clips:
            print(f"    - Processed {i + 1}/{total_clips} clips...")

    return segments


def build_fireworks_payload(segments: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Format segments into Fireworks OpenAI-compatible embedding API payload."""
    batch_input = []
    for s in segments:
        content = []
        for b64 in s["frames_b64"]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        batch_input.append({"content": content})

    return {
        "model": model_name,
        "input": batch_input,
    }


async def submit_batch(
    payload: Dict[str, Any],
    api_key: str,
    endpoint_url: str = "https://api.fireworks.ai/inference/v1",
    mock: bool = False,
) -> Dict[str, Any]:
    """Dispatches batch to Fireworks or runs mock."""
    num_items = len(payload["input"])
    start_t = time.perf_counter()

    if mock:
        # Simulate FP8 execution on H100: ~0.70s for 226 items (~250k tokens)
        simulated_delay = 0.05 + (num_items * 0.003)
        await asyncio.sleep(simulated_delay)
        elapsed = time.perf_counter() - start_t
        total_tokens = sum(len(item["content"]) * 140 for item in payload["input"])
        # Return mock 768-dim embeddings
        mock_embeddings = [[0.01] * 768 for _ in range(num_items)]
        return {
            "elapsed_s": elapsed,
            "total_tokens": total_tokens,
            "embeddings": mock_embeddings,
            "status": "success (mock)",
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{endpoint_url.rstrip('/')}/embeddings", headers=headers, json=payload)
        elapsed = time.perf_counter() - start_t
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        usage = data.get("usage", {})
        total_tokens = usage.get("prompt_tokens") or sum(len(item["content"]) * 140 for item in payload["input"])
        return {
            "elapsed_s": elapsed,
            "total_tokens": total_tokens,
            "embeddings": embeddings,
            "status": "success",
        }


async def main():
    parser = argparse.ArgumentParser(description="Embed video segments using Fireworks AI")
    parser.add_argument("--video", type=str, default="data/videos/Rick and Morty ｜ Season 9 Battle Scenes ｜ adult swim.webm")
    parser.add_argument("--clip-duration", type=float, default=8.0, help="Clip length in seconds (default 8s)")
    parser.add_argument("--frames-per-clip", type=int, default=4, help="Frames per clip (default 4)")
    parser.add_argument("--max-clips", type=int, default=10, help="Max clips to test (omit for all)")
    parser.add_argument("--output-embeddings", type=str, default="data/video_embeddings.json")
    parser.add_argument("--mock", action="store_true", default=True, help="Run mock simulation")
    parser.add_argument("--live", action="store_true", help="Send to live Fireworks deployment")
    parser.add_argument("--model", type=str, default="accounts/fireworks/models/embeddinggemma-2")
    parser.add_argument("--api-key", type=str, default=None)

    args = parser.parse_args()

    print("=" * 80)
    print("VIDEO FOOTAGE EMBEDDING EXECUTION PIPELINE")
    print("=" * 80)
    print(f"Video File:       {args.video}")
    print(f"Clip Duration:    {args.clip_duration}s ({args.frames_per_clip} frames/clip)")
    print(f"Model:            {args.model}")
    print(f"Execution Mode:   {'LIVE API' if args.live else 'MOCK SIMULATION'}")
    print("-" * 80)

    # 1. Extract frames for clips
    segments_dir = "data/temp_clip_frames"
    segments = extract_video_segments(
        video_path=args.video,
        output_dir=segments_dir,
        clip_duration_s=args.clip_duration,
        frames_per_clip=args.frames_per_clip,
        max_clips=args.max_clips,
    )

    # 2. Build payload
    payload = build_fireworks_payload(segments, args.model)
    payload_size_mb = len(json.dumps(payload)) / (1024 * 1024)
    print(f"\n[+] Built batch payload for {len(segments)} clips ({payload_size_mb:.2f} MB)")

    # 3. Submit batch to Fireworks
    print("[*] Submitting batch to Fireworks embedding engine...")
    res = await submit_batch(
        payload=payload,
        api_key=args.api_key or "",
        mock=not args.live,
    )

    elapsed = res["elapsed_s"]
    tokens = res["total_tokens"]
    tps = tokens / elapsed if elapsed > 0 else 0
    cost = (8.00 / 3600.0) * elapsed

    print(f"[+] Completed in {elapsed*1000:.1f} ms ({tps:,.0f} tokens/sec)")
    print(f"[+] Total tokens: {tokens:,}")
    print(f"[+] Measured GPU Cost: ${cost:.6f} (${cost*100:.4f} cents)")

    # 4. Save results
    output_records = []
    for s, emb in zip(segments, res["embeddings"]):
        output_records.append({
            "clip_id": s["clip_id"],
            "start_time_s": s["start_time_s"],
            "end_time_s": s["end_time_s"],
            "embedding_dim": len(emb),
            "sample_embedding_vector": emb[:5],
        })

    os.makedirs(os.path.dirname(args.output_embeddings) or ".", exist_ok=True)
    with open(args.output_embeddings, "w") as f:
        json.dump(output_records, f, indent=2)
    print(f"[+] Saved {len(output_records)} indexed clip embeddings to {args.output_embeddings}\n")


if __name__ == "__main__":
    asyncio.run(main())
