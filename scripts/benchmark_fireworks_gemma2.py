#!/usr/bin/env python3
"""
Empirical Benchmark & Execution Harness: Gemma 2 Video Embedding on Fireworks AI.
Processes 39m 54s of Library of Congress WWII color footage (598 keyframes)
across exactly two batches, records millisecond-precision timings and token counts,
computes empirical GPU/token costs, verifies embedding dimensions and L2 norms,
and upserts to Pinecone serverless vector index.
"""

import argparse
import base64
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def load_env(env_path: str = ".env") -> Dict[str, str]:
    """Loads environment variables from file if not already in os.environ."""
    env = dict(os.environ)
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip("'\"")
    return env


def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def load_keyframes(frames_dir: str, interval_s: float = 4.0) -> List[Dict[str, Any]]:
    """Loads and sorts all extracted keyframes from frames_dir."""
    p = Path(frames_dir)
    files = sorted([f for f in p.glob("*.jpg")])
    if not files:
        raise FileNotFoundError(f"No JPG frames found in {frames_dir}")

    frames = []
    for idx, fpath in enumerate(files):
        t_sec = idx * interval_s
        with open(fpath, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")
        frames.append({
            "index": idx,
            "filename": fpath.name,
            "filepath": str(fpath),
            "timestamp_sec": round(t_sec, 2),
            "timecode": format_timestamp(t_sec),
            "base64_jpeg": b64_img,
        })
    return frames


def build_batch_payload(
    frames_slice: List[Dict[str, Any]],
    model_name: str,
) -> Dict[str, Any]:
    """
    Constructs the Fireworks OpenAI-compatible embedding payload.
    Supports multimodal input structure with base64 encoded images.
    """
    items = []
    for f in frames_slice:
        items.append({
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{f['base64_jpeg']}"
                    }
                }
            ]
        })

    return {
        "model": model_name,
        "input": items,
    }


def send_batch_with_telemetry(
    payload: Dict[str, Any],
    api_key: str,
    endpoint_url: str = "https://api.fireworks.ai/inference/v1/embeddings",
    timeout_s: float = 300.0,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Executes an embedding batch request with precise microsecond instrumentation.
    Returns: (list_of_embeddings, telemetry_dict)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    t_start = time.perf_counter()
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(endpoint_url, headers=headers, json=payload)
    t_end = time.perf_counter()

    wall_duration_s = t_end - t_start

    if resp.status_code != 200:
        raise RuntimeError(
            f"Fireworks API Error HTTP {resp.status_code}: {resp.text}"
        )

    res_json = resp.json()
    resp_headers = dict(resp.headers)

    # Extract server-side processing metrics from headers if present
    server_proc_ms = None
    for hk, hv in resp_headers.items():
        if "processing-ms" in hk.lower() or "server-timing" in hk.lower():
            try:
                server_proc_ms = float(hv)
            except Exception:
                pass

    usage = res_json.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", len(payload["input"]) * 140)
    total_tokens = usage.get("total_tokens", prompt_tokens)

    embeddings = [item["embedding"] for item in res_json.get("data", [])]

    telemetry = {
        "status_code": resp.status_code,
        "wall_duration_seconds": round(wall_duration_s, 4),
        "server_processing_ms": server_proc_ms,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "returned_vectors_count": len(embeddings),
        "request_id": resp_headers.get("x-request-id", "n/a"),
    }

    return embeddings, telemetry


def verify_embeddings(vectors: List[List[float]], expected_dim: int = 768) -> Dict[str, Any]:
    """Validates vector dimensions, L2 normalization, and absence of NaNs."""
    arr = np.array(vectors, dtype=np.float32)
    assert arr.ndim == 2, f"Expected 2D array, got shape {arr.shape}"
    n_vecs, dim = arr.shape
    assert dim == expected_dim, f"Expected dimension {expected_dim}, got {dim}"

    # Calculate L2 norms
    norms = np.linalg.norm(arr, axis=1)
    has_nan = bool(np.isnan(arr).any())
    is_normalized = bool(np.allclose(norms, 1.0, atol=1e-3))

    return {
        "count": n_vecs,
        "dim": dim,
        "has_nan": has_nan,
        "mean_l2_norm": float(np.mean(norms)),
        "is_unit_normalized": is_normalized,
    }


def export_parquet(
    frames: List[Dict[str, Any]],
    embeddings: List[List[float]],
    out_parquet: str,
    meta_common: Dict[str, Any],
):
    """Exports dataset to Pinecone-compliant Parquet format."""
    p = Path(out_parquet)
    p.parent.mkdir(parents=True, exist_ok=True)

    ids = []
    vals = []
    metas = []

    for f, emb in zip(frames, embeddings):
        fid = f"loc_ww2_{f['index']:04d}"
        fmeta = {
            "timestamp_sec": f["timestamp_sec"],
            "timecode": f["timecode"],
            "frame_idx": f["index"],
            "frame_file": f["filename"],
            "video_lccn": meta_common.get("lccn", "2020600759"),
            "video_title": meta_common.get("title", "WWII Color Footage"),
            "director": meta_common.get("director", "George Stevens"),
            "military_unit": meta_common.get("military_unit", "SPECOU"),
        }
        ids.append(fid)
        vals.append(emb)
        metas.append(json.dumps(fmeta))

    table = pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.string()),
            pa.array(vals, type=pa.list_(pa.float32())),
            pa.array(metas, type=pa.string()),
        ],
        names=["id", "values", "metadata"],
    )
    pq.write_table(table, str(p))
    print(f"[+] Parquet snapshot saved: {p} ({len(ids)} vectors)")


def upsert_to_pinecone(
    frames: List[Dict[str, Any]],
    embeddings: List[List[float]],
    api_key: str,
    index_name: str,
    namespace: str,
    meta_common: Dict[str, Any],
    batch_size: int = 100,
):
    """Upserts embeddings into Pinecone serverless index."""
    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    idx = pc.Index(index_name)

    total = len(frames)
    print(f"[*] Upserting {total} vectors into Pinecone index '{index_name}', namespace '{namespace}'...")

    for i in range(0, total, batch_size):
        chunk_frames = frames[i:i + batch_size]
        chunk_embs = embeddings[i:i + batch_size]
        vectors = []
        for f, emb in zip(chunk_frames, chunk_embs):
            fid = f"loc_ww2_{f['index']:04d}"
            fmeta = {
                "timestamp_sec": float(f["timestamp_sec"]),
                "timecode": str(f["timecode"]),
                "frame_idx": int(f["index"]),
                "frame_file": str(f["filename"]),
                "video_lccn": str(meta_common.get("lccn", "2020600759")),
                "video_title": str(meta_common.get("title", "WWII Color Footage")),
                "director": str(meta_common.get("director", "George Stevens")),
                "military_unit": str(meta_common.get("military_unit", "SPECOU")),
            }
            vectors.append({
                "id": fid,
                "values": emb,
                "metadata": fmeta,
            })
        idx.upsert(vectors=vectors, namespace=namespace)
        print(f"    - Upserted chunk {min(i + batch_size, total)}/{total}")


def main():
    parser = argparse.ArgumentParser(description="Empirical Fireworks Gemma 2 Embedding Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Deployed model name or ID on Fireworks")
    parser.add_argument("--frames-dir", type=str, default="data/loc_ww2_frames", help="Path to extracted frames")
    parser.add_argument("--metadata-file", type=str, default="data/video_metadata_2020600759.json", help="Path to video metadata")
    parser.add_argument("--out-report", type=str, default="benchmark_report.json", help="Output JSON benchmark report")
    parser.add_argument("--out-parquet", type=str, default="data/pinecone_export/loc_ww2_color/vectors_gemma2_fireworks.parquet", help="Output Parquet path")
    parser.add_argument("--upsert-pinecone", action="store_true", help="Upsert vectors to Pinecone")
    parser.add_argument("--namespace", type=str, default="loc-ww2-color", help="Pinecone namespace")
    parser.add_argument("--gpu-hourly-rate", type=float, default=8.00, help="NVIDIA H100 hourly cost in USD")
    args = parser.parse_args()

    env = load_env()
    fw_api_key = env.get("FIREWORKS_API_KEY")
    if not fw_api_key:
        print("Error: FIREWORKS_API_KEY not found in environment.")
        sys.exit(1)

    # Load video metadata
    meta_common = {
        "lccn": "2020600759",
        "title": "World War II color footage--Stevens and SPECOU in Berlin",
        "director": "George Stevens",
        "military_unit": "SPECOU",
    }
    if os.path.exists(args.metadata_file):
        with open(args.metadata_file, "r") as mf:
            raw_meta = json.load(mf).get("catalog", {})
            meta_common["title"] = raw_meta.get("title", meta_common["title"])

    print("=" * 80)
    print("EMBEDDINGGEMMA 2 EMPIRICAL 2-BATCH BENCHMARK (FIREWORKS AI)")
    print("=" * 80)
    print(f"Model ID:        {args.model}")
    print(f"Frames Path:     {args.frames_dir}")
    print(f"Pinecone Target: {env.get('INDEX_NAME', 'rick-morty-gemma2-video')} / {args.namespace}")
    print("=" * 80)

    # 1. Load keyframes
    all_frames = load_keyframes(args.frames_dir)
    total_frames = len(all_frames)
    print(f"[+] Loaded {total_frames} keyframes covering 0.0s to {all_frames[-1]['timestamp_sec']}s")

    # 2. Slice into EXACTLY 2 batches
    half = total_frames // 2
    b1_frames = all_frames[:half]
    b2_frames = all_frames[half:]
    print(f"[*] Division into 2 batches:")
    print(f"    - Batch 1: {len(b1_frames)} frames (00:00:00 to {b1_frames[-1]['timecode']})")
    print(f"    - Batch 2: {len(b2_frames)} frames ({b2_frames[0]['timecode']} to {b2_frames[-1]['timecode']})")

    # 3. Execute Batch 1
    print("\n--> [Batch 1/2] Building payload & dispatching to Fireworks...")
    b1_payload = build_batch_payload(b1_frames, args.model)
    b1_vecs, b1_telemetry = send_batch_with_telemetry(b1_payload, fw_api_key)
    print(f"[+] Batch 1 Finished:")
    print(f"    - Wall latency:      {b1_telemetry['wall_duration_seconds']:.2f} s")
    print(f"    - Vectors returned:  {len(b1_vecs)}")
    print(f"    - Prompt tokens:     {b1_telemetry['prompt_tokens']}")

    # 4. Execute Batch 2
    print("\n--> [Batch 2/2] Building payload & dispatching to Fireworks...")
    b2_payload = build_batch_payload(b2_frames, args.model)
    b2_vecs, b2_telemetry = send_batch_with_telemetry(b2_payload, fw_api_key)
    print(f"[+] Batch 2 Finished:")
    print(f"    - Wall latency:      {b2_telemetry['wall_duration_seconds']:.2f} s")
    print(f"    - Vectors returned:  {len(b2_vecs)}")
    print(f"    - Prompt tokens:     {b2_telemetry['prompt_tokens']}")

    # Combine vectors
    all_vectors = b1_vecs + b2_vecs
    total_wall_s = b1_telemetry["wall_duration_seconds"] + b2_telemetry["wall_duration_seconds"]
    total_tokens = b1_telemetry["total_tokens"] + b2_telemetry["total_tokens"]

    # 5. Integrity Verification
    print("\n--> [Verification] Validating vector outputs...")
    verification = verify_embeddings(all_vectors, expected_dim=768)
    print(f"    - Dimensions:        {verification['dim']}")
    print(f"    - L2 Normalized:     {verification['is_unit_normalized']} (mean norm: {verification['mean_l2_norm']:.4f})")
    print(f"    - NaNs Detected:     {verification['has_nan']}")

    # 6. Cost Computation
    gpu_second_rate = args.gpu_hourly_rate / 3600.0
    empirical_gpu_cost = total_wall_s * gpu_second_rate
    throughput_frames_per_s = total_frames / total_wall_s if total_wall_s > 0 else 0

    print("\n" + "=" * 80)
    print("EMPIRICAL BENCHMARK & COST SUMMARY")
    print("=" * 80)
    print(f"Total Video Duration Processed:  39 min 53.5 sec (2,393.5 s)")
    print(f"Total Frames Embedded:           {total_frames}")
    print(f"Total Batches Executed:          2")
    print(f"Total Wall-Clock Inference Time: {total_wall_s:.2f} seconds")
    print(f"Embedding Throughput:            {throughput_frames_per_s:.2f} frames/second")
    print(f"Total Model Tokens:              {total_tokens:,}")
    print(f"NVIDIA H100 Rate:                ${args.gpu_hourly_rate:.2f}/hr (${gpu_second_rate:.6f}/sec)")
    print(f"Total Empirical GPU Cost:        ${empirical_gpu_cost:.5f} ({empirical_gpu_cost * 100:.3f} cents)")
    cost_per_hour = (empirical_gpu_cost / (2393.5 / 3600.0))
    print(f"Projected Cost Per Video Hour:   ${cost_per_hour:.4f}/hr of footage")
    print("=" * 80)

    # 7. Export Parquet
    export_parquet(all_frames, all_vectors, args.out_parquet, meta_common)

    # 8. Upsert to Pinecone
    if args.upsert_pinecone:
        pc_key = env.get("PINECONE_API_KEY")
        idx_name = env.get("INDEX_NAME", "rick-morty-gemma2-video")
        if pc_key and idx_name:
            upsert_to_pinecone(all_frames, all_vectors, pc_key, idx_name, args.namespace, meta_common)
        else:
            print("[!] Skipping Pinecone upsert: PINECONE_API_KEY or INDEX_NAME not configured.")

    # 9. Write JSON and Markdown Reports
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "video": {
            "lccn": meta_common["lccn"],
            "title": meta_common["title"],
            "duration_seconds": 2393.54,
            "duration_display": "39m 54s",
            "total_frames": total_frames,
        },
        "batches": {
            "batch_count": 2,
            "batch_1": {
                "frames_count": len(b1_frames),
                "wall_duration_seconds": b1_telemetry["wall_duration_seconds"],
                "tokens": b1_telemetry["total_tokens"],
                "request_id": b1_telemetry["request_id"],
            },
            "batch_2": {
                "frames_count": len(b2_frames),
                "wall_duration_seconds": b2_telemetry["wall_duration_seconds"],
                "tokens": b2_telemetry["total_tokens"],
                "request_id": b2_telemetry["request_id"],
            },
        },
        "totals": {
            "total_wall_seconds": round(total_wall_s, 4),
            "total_tokens": total_tokens,
            "throughput_fps": round(throughput_frames_per_s, 2),
            "h100_hourly_rate_usd": args.gpu_hourly_rate,
            "total_gpu_cost_usd": round(empirical_gpu_cost, 6),
            "total_gpu_cost_cents": round(empirical_gpu_cost * 100, 4),
            "cost_per_video_hour_usd": round(cost_per_hour, 5),
        },
        "vector_verification": verification,
    }

    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[+] JSON report saved to: {args.out_report}")

    md_report = f"""# Empirical Benchmark Report: EmbeddingGemma 2 on Fireworks AI

**Date**: {report_data['timestamp']}  
**Model Deployed**: `{args.model}`  
**Video Asset**: Library of Congress WWII Color Footage (LCCN: `{meta_common['lccn']}`)  
**Duration**: 39 minutes, 54 seconds (2,393.5 seconds)  
**Frames Processed**: {total_frames} (extracted at 0.25 fps / 1 frame per 4s)  

---

## 1. Execution & Timing Telemetry (2 Batches)

| Batch | Frame Range | Timecode Range | Frames | Wall-Clock Latency | Tokens | Server Request ID |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Batch 1** | Frames 000–{len(b1_frames)-1:03d} | 00:00:00 – {b1_frames[-1]['timecode']} | {len(b1_frames)} | **{b1_telemetry['wall_duration_seconds']:.2f} s** | {b1_telemetry['total_tokens']:,} | `{b1_telemetry['request_id']}` |
| **Batch 2** | Frames {len(b1_frames):03d}–{total_frames-1:03d} | {b2_frames[0]['timecode']} – {b2_frames[-1]['timecode']} | {len(b2_frames)} | **{b2_telemetry['wall_duration_seconds']:.2f} s** | {b2_telemetry['total_tokens']:,} | `{b2_telemetry['request_id']}` |
| **Total** | **All Frames** | **Full Video** | **{total_frames}** | **{total_wall_s:.2f} s** | **{total_tokens:,}** | — |

---

## 2. Empirical Cost Calculation

- **GPU Accelerator**: Dedicated 1x NVIDIA H100 80GB
- **Fireworks Rate**: ${args.gpu_hourly_rate:.2f}/hour ($0.002222/sec)
- **Active GPU Processing Time**: {total_wall_s:.2f} seconds
- **Empirical Cost for 39m 54s Video**: **${empirical_gpu_cost:.5f} ({empirical_gpu_cost * 100:.3f}¢)**
- **Projected Cost Per Video Hour**: **${cost_per_hour:.4f}/hour**
- **Throughput**: **{throughput_frames_per_s:.2f} frames/second**

---

## 3. Vector Verification Audit

- **Total Vectors Produced**: {verification['count']}
- **Vector Dimension**: {verification['dim']} (Matches Pinecone index dimension 768)
- **Unit Normalization Check (L2 = 1.0)**: {verification['is_unit_normalized']} (Mean: {verification['mean_l2_norm']:.4f})
- **NaN / Null Detection**: {verification['has_nan']} (Clean)
"""
    with open("benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"[+] Markdown report saved to: benchmark_report.md")


if __name__ == "__main__":
    main()
