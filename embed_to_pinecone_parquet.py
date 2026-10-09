"""
2-Batch 4-Second Video Embedding Pipeline with Pinecone-Compliant Parquet Export.
Splits 30-minute video into 452 4-second clips (Batch 1: 256, Batch 2: 196).
Exports embeddings to Pinecone serverless bulk-import Parquet schema:
  - id: String
  - values: List[Float32] (768-dim)
  - metadata: String (JSON formatted)
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
import pyarrow as pa
import pyarrow.parquet as pq


def format_timestamp(seconds: float) -> str:
    """Converts seconds float to HH:MM:SS format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def extract_4s_keyframes(
    video_path: str,
    output_dir: str,
    interval_s: float = 4.0,
) -> List[Dict[str, Any]]:
    """
    Extracts 1 keyframe at the midpoint of each 4-second interval.
    For a 1804.2s video, this produces 452 clips.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Probe duration
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", video_path,
    ]
    meta = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    total_dur = float(meta["format"]["duration"])
    total_clips = math.ceil(total_dur / interval_s)

    print(f"[*] Extracting 1 keyframe every {interval_s}s across {total_dur:.1f}s video ({total_clips} total clips)...")
    clips = []

    for i in range(total_clips):
        start_t = i * interval_s
        end_t = min(total_dur, (i + 1) * interval_s)
        mid_t = start_t + (end_t - start_t) / 2.0

        frame_filename = f"clip_{i:04d}_mid.jpg"
        frame_path = out_dir / frame_filename

        # Extract and resize frame to 336x336
        ff_cmd = [
            "ffmpeg",
            "-ss", str(mid_t),
            "-i", video_path,
            "-vf", "scale=336:336:force_original_aspect_ratio=decrease,pad=336:336:(ow-iw)/2:(oh-ih)/2",
            "-vframes", "1",
            "-q:v", "3",
            str(frame_path),
            "-y",
            "-v", "error",
        ]
        subprocess.run(ff_cmd, check=True)

        with open(frame_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        clips.append({
            "clip_index": i,
            "id": f"rm_s9_battle_{i:04d}",
            "start_time_s": round(start_t, 2),
            "end_time_s": round(end_t, 2),
            "mid_time_s": round(mid_t, 2),
            "timestamp_label": f"{format_timestamp(start_t)} - {format_timestamp(end_t)}",
            "frame_b64": b64_data,
        })

        if (i + 1) % 100 == 0 or (i + 1) == total_clips:
            print(f"    - Extracted {i + 1}/{total_clips} keyframes...")

    return clips


def build_batch_payload(clips: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Builds Fireworks OpenAI-compatible embedding JSON payload for a list of clips."""
    batch_input = []
    for c in clips:
        batch_input.append({
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{c['frame_b64']}"}
                }
            ]
        })
    return {
        "model": model_name,
        "input": batch_input,
    }


async def send_batch_request(
    payload: Dict[str, Any],
    api_key: str,
    endpoint_url: str = "https://api.fireworks.ai/inference/v1",
    mock: bool = False,
) -> List[List[float]]:
    """Sends one batch to Fireworks or generates mock 768-dim vectors."""
    num_items = len(payload["input"])
    if mock:
        # Simulate ~150ms H100 execution at FP8
        await asyncio.sleep(0.15)
        # Generate normalized dummy 768-dim float vectors
        import numpy as np
        vecs = np.random.randn(num_items, 768).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        return vecs.tolist()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{endpoint_url.rstrip('/')}/embeddings", headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        return [item["embedding"] for item in data["data"]]


def export_to_pinecone_parquet(
    clips: List[Dict[str, Any]],
    embeddings: List[List[float]],
    output_parquet_path: str,
    video_id: str = "9Rul9N1LREQ",
    video_title: str = "Rick and Morty | Season 9 Battle Scenes | adult swim",
):
    """
    Constructs and writes a Pinecone-compliant Parquet file.
    Schema matches Pinecone serverless bulk-import specs:
      - id: string
      - values: list<float32>
      - metadata: string (JSON string)
    """
    ids = []
    values = []
    metadatas = []

    for c, emb in zip(clips, embeddings):
        ids.append(c["id"])
        values.append(emb)

        # Pinecone metadata: key-values must be string, number, bool, or list of strings
        meta_dict = {
            "video_id": video_id,
            "title": video_title,
            "start_time_s": float(c["start_time_s"]),
            "end_time_s": float(c["end_time_s"]),
            "timestamp": c["timestamp_label"],
            "clip_index": int(c["clip_index"]),
            "media_type": "video_clip_4s",
            "youtube_url": f"https://youtu.be/{video_id}?t={int(c['start_time_s'])}",
        }
        metadatas.append(json.dumps(meta_dict))

    # Define strict PyArrow Schema for Pinecone
    schema = pa.schema([
        ("id", pa.string()),
        ("values", pa.list_(pa.float32())),
        ("metadata", pa.string()),
    ])

    table = pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.string()),
            pa.array(values, type=pa.list_(pa.float32())),
            pa.array(metadatas, type=pa.string()),
        ],
        schema=schema,
    )

    out_file = Path(output_parquet_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_file), compression="snappy")
    print(f"\n[+] Exported Pinecone Parquet file to: {out_file}")
    print(f"    - Total Records: {len(table):,}")
    print(f"    - Columns: {table.column_names}")
    print(f"    - Vector Dimension: {len(values[0])}")
    print(f"    - File Size: {out_file.stat().st_size / (1024 * 1024):.2f} MB")
    return out_file


async def run_pipeline(
    video_path: str,
    output_parquet: str,
    mock: bool = True,
    endpoint_url: str = "https://api.fireworks.ai/inference/v1",
    model_name: str = "accounts/fireworks/models/embeddinggemma-2",
    api_key: Optional[str] = None,
):
    print("=" * 85)
    print("2-BATCH 4-SECOND VIDEO EMBEDDING & PINECONE PARQUET EXPORTER")
    print("=" * 85)
    print(f"Video Path:     {video_path}")
    print(f"Interval:       4.0 seconds (1 frame per clip)")
    print(f"Target Schema:  Pinecone Bulk Import Parquet (id, values, metadata)")
    print(f"Execution Mode: {'MOCK SIMULATION' if mock else 'LIVE FIREWORKS API'}")
    print("-" * 85)

    # 1. Extract 452 keyframes
    keyframes_dir = "data/temp_4s_frames"
    clips = extract_4s_keyframes(video_path, keyframes_dir, interval_s=4.0)
    total_clips = len(clips)

    # 2. Split into 2 Batches
    batch_1 = clips[:256]
    batch_2 = clips[256:]
    print(f"\n[*] Divided into 2 Batches for BS=256:")
    print(f"    - Batch 1: {len(batch_1)} items (covering 00:00:00 to {batch_1[-1]['timestamp_label'].split(' - ')[1]})")
    print(f"    - Batch 2: {len(batch_2)} items (covering {batch_2[0]['timestamp_label'].split(' - ')[0]} to {batch_2[-1]['timestamp_label'].split(' - ')[1]})")

    # 3. Process Batch 1
    print("\n--> Submitting Batch 1 (256 items)...", end="", flush=True)
    t0 = time.perf_counter()
    p1 = build_batch_payload(batch_1, model_name)
    emb_1 = await send_batch_request(p1, api_key or "", endpoint_url, mock=mock)
    dur_1 = time.perf_counter() - t0
    print(f" Done in {dur_1*1000:.1f} ms ({len(batch_1)} vectors)")

    # 4. Process Batch 2
    print("--> Submitting Batch 2 (196 items)...", end="", flush=True)
    t1 = time.perf_counter()
    p2 = build_batch_payload(batch_2, model_name)
    emb_2 = await send_batch_request(p2, api_key or "", endpoint_url, mock=mock)
    dur_2 = time.perf_counter() - t1
    print(f" Done in {dur_2*1000:.1f} ms ({len(batch_2)} vectors)")

    all_embeddings = emb_1 + emb_2
    total_latency_s = dur_1 + dur_2
    total_gpu_cost = (8.00 / 3600.0) * total_latency_s

    print("-" * 85)
    print(f"[+] Total Vectors Generated: {len(all_embeddings):,}")
    print(f"[+] Total Processing Time:   {total_latency_s*1000:.1f} ms")
    print(f"[+] Total H100 Cost:         ${total_gpu_cost:.6f} ({total_gpu_cost*100:.4f} cents)")

    # 5. Export to Pinecone Parquet
    export_to_pinecone_parquet(
        clips=clips,
        embeddings=all_embeddings,
        output_parquet_path=output_parquet,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-batch video embedding to Pinecone Parquet")
    parser.add_argument(
        "--video",
        type=str,
        default="data/videos/Rick and Morty ｜ Season 9 Battle Scenes ｜ adult swim.webm",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default="data/pinecone_export/default/rick_and_morty_s9_4s_embeddings.parquet",
    )
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", type=str, default="accounts/fireworks/models/embeddinggemma-2")
    parser.add_argument("--api-key", type=str, default=None)

    args = parser.parse_args()
    asyncio.run(
        run_pipeline(
            video_path=args.video,
            output_parquet=args.output_parquet,
            mock=not args.live,
            model_name=args.model,
            api_key=args.api_key,
        )
    )
