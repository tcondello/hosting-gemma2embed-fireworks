#!/usr/bin/env python3
"""
Automated Library of Congress (LOC) Video Ingestion Pipeline.

Fetches any digitized historical video from loc.gov, extracts keyframes at a configurable
interval, generates 768-dimensional normalized embeddings via EmbeddingGemma 2 on Apple Silicon
or CPU, and appends the new frames and vectors to the local database and Hugging Face Space index.

Usage:
    python scripts/ingest_loc_video.py --item-id 2015600171 --interval 4.0
    python scripts/ingest_loc_video.py --url https://www.loc.gov/item/2015600171/ --max-frames 50
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


def format_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def fetch_loc_item_metadata(item_id_or_url: str) -> Dict[str, Any]:
    """Queries the LOC JSON API to extract catalog metadata and direct MP4 URL."""
    clean_id = item_id_or_url.strip().rstrip("/")
    if "loc.gov/item/" in clean_id:
        clean_id = clean_id.split("loc.gov/item/")[-1].split("/")[0]

    api_url = f"https://www.loc.gov/item/{clean_id}/?fo=json"
    headers = {"User-Agent": "LOC-Video-Pipeline/2.0 (Historical Archive Search)"}
    print(f"[*] Querying LOC metadata API: {api_url}")

    data = None
    try:
        with httpx.Client(timeout=45.0, follow_redirects=True) as client:
            resp = client.get(api_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
    except Exception as e:
        print(f"[*] httpx timed out or failed ({e}), falling back to curl...")

    if not data:
        curl_cmd = ["curl", "-s", "-L", api_url]
        res = subprocess.run(curl_cmd, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout:
            data = json.loads(res.stdout)
        else:
            raise RuntimeError(f"Failed to fetch metadata for LOC item {clean_id}")

    item_info = data.get("item", {})
    raw_title = item_info.get("title") or data.get("title") or f"LOC Video {clean_id}"
    if isinstance(raw_title, list):
        raw_title = raw_title[0]
    title = raw_title.strip("[]")

    date = item_info.get("date") or data.get("date") or "Unknown"
    description_list = item_info.get("notes", []) or data.get("description", [])
    summary = " ".join(description_list) if isinstance(description_list, list) else str(description_list)
    summary = re.sub(r"\s+", " ", summary).strip()

    # Search resources for direct MP4 stream
    mp4_url = None
    resources = data.get("resources", [])
    for res in resources:
        for file_group in res.get("files", []):
            for f in file_group:
                url = f.get("url", "")
                mtype = f.get("mimetype", "")
                if mtype == "video/mp4" or url.endswith(".mp4"):
                    mp4_url = url
                    break
            if mp4_url:
                break
        if mp4_url:
            break

    # If no explicit MP4 in resources, check for ntscrm identifier pattern
    if not mp4_url:
        for res in resources:
            for file_group in res.get("files", []):
                for f in file_group:
                    url = f.get("url", "")
                    m = re.search(r"/ntscrm/(\d+)/", url)
                    if m:
                        candidate = f"https://tile.loc.gov/storage-services/service/mbrs/ntscrm/{m.group(1)}/{m.group(1)}.mp4"
                        # Verify candidate via curl HEAD
                        check = subprocess.run(["curl", "-s", "-I", candidate], capture_output=True, text=True)
                        if "200" in check.stdout or "302" in check.stdout:
                            mp4_url = candidate
                            print(f"[+] Discovered active MP4 stream via ntscrm registry: {candidate}")
                            break
                if mp4_url:
                    break

    if not mp4_url:
        raise ValueError(f"Could not locate a direct MP4 stream URL for LOC item {clean_id}")

    return {
        "id": clean_id,
        "title": title,
        "date": date,
        "summary": summary[:600] if len(summary) > 600 else summary,
        "loc_item_url": f"https://www.loc.gov/item/{clean_id}/",
        "stream_url": mp4_url,
        "archive": "Library of Congress (LOC)",
    }


def fetch_nara_item_metadata(item_id_or_url: str) -> Dict[str, Any]:
    """Queries the National Archives (NARA) API to extract metadata and direct MP4 URL."""
    clean_id = item_id_or_url.strip().rstrip("/")
    if "catalog.archives.gov/id/" in clean_id:
        clean_id = clean_id.split("catalog.archives.gov/id/")[-1].split("/")[0]

    api_url = f"https://catalog.archives.gov/proxy/v3/records/search?naId={clean_id}"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) (Historical Research)"}
    print(f"[*] Querying NARA Catalog API: {api_url}")

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.get(api_url, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"NARA API responded with HTTP {resp.status_code}")
        data = resp.json()

    hits = data.get("body", {}).get("hits", {}).get("hits", [])
    if not hits:
        raise ValueError(f"No NARA record found for NAID {clean_id}")

    hit = hits[0].get("_source", {})
    record = hit.get("record", {})
    title = record.get("title", f"NARA Record {clean_id}")
    summary = record.get("scopeAndContentNote") or "National Archives and Records Administration historic motion picture."
    date = "1945"

    mp4_url = None
    for dobj in record.get("digitalObjects", []):
        obj_url = dobj.get("objectUrl", "")
        if obj_url.endswith(".mp4") or "mp4" in dobj.get("objectType", "").lower():
            mp4_url = obj_url
            break

    if not mp4_url:
        raise ValueError(f"Could not locate an MP4 stream for NARA item {clean_id}")

    return {
        "id": f"nara_{clean_id}",
        "title": title,
        "date": date,
        "summary": summary[:600] if len(summary) > 600 else summary,
        "loc_item_url": f"https://catalog.archives.gov/id/{clean_id}",
        "stream_url": mp4_url,
        "archive": "National Archives (NARA)",
    }


def extract_keyframes_ffmpeg(
    stream_url: str,
    output_dir: Path,
    interval_s: float = 4.0,
    max_frames: Optional[int] = None,
) -> List[Tuple[Path, float]]:
    """Extracts keyframes directly from the LOC MP4 stream URL using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fps_val = 1.0 / interval_s
    pattern = str(output_dir / "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "warning",
        "-i", stream_url,
        "-vf", f"fps={fps_val:.4f},scale=640:-1",
        "-q:v", "3",
    ]
    if max_frames:
        cmd.extend(["-vframes", str(max_frames)])
    cmd.append(pattern)

    print(f"[*] Extracting keyframes (1 every {interval_s}s) via FFmpeg...")
    print(f"    Command: {' '.join(cmd[:6])} ...")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg failed (code {res.returncode}): {res.stderr}")

    jpg_files = sorted(output_dir.glob("frame_*.jpg"))
    print(f"[+] Extracted {len(jpg_files)} keyframes in {time.time() - t0:.1f}s")

    result = []
    for idx, f in enumerate(jpg_files):
        sec = round(idx * interval_s, 2)
        result.append((f, sec))
    return result


def embed_keyframes_gemma2(
    keyframes: List[Tuple[Path, float]],
    meta: Dict[str, Any],
    model: SentenceTransformer,
    batch_size: int = 32,
    device: str = "mps",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generates 768-dim normalized embeddings directly from keyframe images using Gemma 2."""
    metadata_list = []
    clean_title = meta["title"].replace('"', "'")
    date_str = meta.get("date", "")
    summary_str = meta.get("summary", "")
    archive_str = meta.get("archive", "Historical Film Archive")

    for idx, (frame_path, start_s) in enumerate(keyframes):
        timecode = format_timestamp(start_s)
        metadata_list.append({
            "t": timecode,
            "s": start_s,
            "vid": meta["id"],
            "vtitle": clean_title,
            "stream": meta["stream_url"],
            "loc_url": meta["loc_item_url"],
            "c": f"{clean_title} ({date_str}) — Recorded at {timecode} • {archive_str}",
        })

    print(f"[*] Generating Gemma 2 visual embeddings for {len(keyframes)} frames (batch_size={batch_size})...")
    t0 = time.perf_counter()
    all_embeddings = []

    for i in range(0, len(keyframes), batch_size):
        chunk = keyframes[i : i + batch_size]
        images = [Image.open(f).convert("RGB") for f, _ in chunk]
        emb = model.encode(
            images,
            batch_size=len(images),
            normalize_embeddings=True,
            show_progress_bar=False,
            device=device,
        )
        all_embeddings.append(emb)

    embeddings_np = np.vstack(all_embeddings).astype(np.float32)
    elapsed = time.perf_counter() - t0
    rate = len(keyframes) / elapsed if elapsed > 0 else 0
    print(f"[+] Encoded {len(keyframes)} image frames in {elapsed:.2f}s ({rate:.1f} frames/sec)")
    return embeddings_np, metadata_list


def append_to_index(
    new_embeddings: np.ndarray,
    new_metadata: List[Dict[str, Any]],
    keyframes: List[Tuple[Path, float]],
    item_id: str,
    data_dir: Path = Path("data"),
    hf_dir: Path = Path("hf_space"),
):
    """Appends new vectors and metadata to both local data/ and hf_space/."""
    vec_bin_path = data_dir / "vectors.bin"
    meta_json_path = data_dir / "frames_metadata.json"
    hf_vec_bin = hf_dir / "data" / "vectors.bin"
    hf_meta_json = hf_dir / "data" / "frames_metadata.json"
    hf_frames_dir = hf_dir / "frames"

    # 1. Load existing metadata
    if meta_json_path.exists():
        with open(meta_json_path, "r") as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = []

    current_idx = len(existing_metadata)

    # 2. Copy frames into hf_space/frames with unique prefix
    hf_frames_dir.mkdir(parents=True, exist_ok=True)
    for (src_path, _), frame_meta in zip(keyframes, new_metadata):
        unique_name = f"{item_id}_{src_path.name}"
        dest_path = hf_frames_dir / unique_name
        shutil.copy2(src_path, dest_path)
        frame_meta["i"] = current_idx
        frame_meta["f"] = unique_name
        current_idx += 1

    # 3. Update metadata JSON
    combined_metadata = existing_metadata + new_metadata
    for target in [meta_json_path, hf_meta_json]:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump(combined_metadata, f)
    print(f"[+] Updated metadata index: now {len(combined_metadata)} total frames")

    # 4. Append to vectors.bin
    new_bytes = new_embeddings.tobytes()
    for target in [vec_bin_path, hf_vec_bin]:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "ab") as f:
            f.write(new_bytes)
        file_size = target.stat().st_size
        total_vectors = file_size // (768 * 4)
        print(f"[+] Updated {target}: {file_size / (1024*1024):.2f} MB ({total_vectors} vectors)")


def update_registry(meta: Dict[str, Any], frame_count: int, duration_s: float):
    """Updates data/films_registry.json with the new film."""
    for reg_path in [Path("data/films_registry.json"), Path("hf_space/data/films_registry.json")]:
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        if reg_path.exists():
            with open(reg_path, "r") as f:
                registry = json.load(f)
        else:
            registry = []

        # Remove if exists
        registry = [r for r in registry if r.get("id") != meta["id"]]
        registry.append({
            "id": meta["id"],
            "title": meta["title"],
            "year": meta.get("date", "Unknown"),
            "duration_sec": round(duration_s, 1),
            "frame_count": frame_count,
            "loc_item_url": meta["loc_item_url"],
            "stream_url": meta["stream_url"],
            "archive": meta.get("archive", "National Archives / LOC"),
        })

        with open(reg_path, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"[+] Updated films registry at {reg_path} ({len(registry)} films registered)")


def main():
    parser = argparse.ArgumentParser(description="Ingest LOC / NARA historical films into EmbeddingGemma 2 index.")
    parser.add_argument("--item-id", type=str, help="Item ID, LCCN, or NAID (e.g. 483301384)")
    parser.add_argument("--url", type=str, help="Catalog item URL (e.g. https://catalog.archives.gov/id/483301384)")
    parser.add_argument("--interval", type=float, default=4.0, help="Seconds between sampled keyframes (default: 4.0)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max frames to extract for testing")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('mps', 'cuda', 'cpu')")
    args = parser.parse_args()

    target = args.item_id or args.url
    if not target:
        parser.print_help()
        print("\nExample: python scripts/ingest_loc_video.py --url https://catalog.archives.gov/id/483301384 --interval 1.0")
        sys.exit(1)

    print("=" * 80)
    print("HISTORICAL ARCHIVES MULTI-VIDEO INGESTION PIPELINE")
    print("=" * 80)

    # 1. Fetch metadata
    if "archives.gov" in target or (target.isdigit() and len(target) == 9):
        meta = fetch_nara_item_metadata(target)
    else:
        meta = fetch_loc_item_metadata(target)
    print(f"\n[Film Found: {meta.get('archive', 'Archive')}]")
    print(f"  Title:      {meta['title']}")
    print(f"  Date:       {meta['date']}")
    print(f"  Stream URL: {meta['stream_url']}")
    print(f"  Catalog:    {meta['loc_item_url']}")

    # 2. Extract keyframes
    temp_dir = Path(f"data/temp_extract_{meta['id']}")
    local_video = Path(f"data/temp_{meta['id']}.mp4")
    video_source = str(local_video) if local_video.exists() else meta["stream_url"]
    print(f"[*] Decoding video source: {video_source}")

    keyframes = extract_keyframes_ffmpeg(
        video_source,
        temp_dir,
        interval_s=args.interval,
        max_frames=args.max_frames,
    )
    if not keyframes:
        print("[-] No keyframes extracted. Exiting.")
        sys.exit(1)

    # 3. Load model
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[*] Loading EmbeddingGemma 2 on {device.upper()}...")
    model = SentenceTransformer("model_weights/embeddinggemma-2", device=device)

    # 4. Generate embeddings
    embeddings, new_metadata = embed_keyframes_gemma2(
        keyframes, meta, model, batch_size=32, device=device
    )

    # 5. Append to database and HF Space
    print("\n[*] Updating vector index and metadata...")
    append_to_index(embeddings, new_metadata, keyframes, meta["id"])

    # 6. Update registry
    duration_s = len(keyframes) * args.interval
    update_registry(meta, len(keyframes), duration_s)

    # 7. Clean up temporary extract folder
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"[+] Cleaned up temporary directory: {temp_dir}")

    print("\n" + "=" * 80)
    print("INGESTION COMPLETE!")
    print(f"Added {len(keyframes)} frames from '{meta['title']}' to the search index.")
    print("To sync to Hugging Face Spaces, run:")
    print("  .venv/bin/python3 -c \"from huggingface_hub import HfApi; api = HfApi(); api.upload_folder(folder_path='hf_space', repo_id='astr010/loc-ww2-color-search', repo_type='space')\"")
    print("=" * 80)


if __name__ == "__main__":
    main()
