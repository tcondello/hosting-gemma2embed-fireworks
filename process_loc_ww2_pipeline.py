"""
End-to-End Pipeline for Library of Congress WWII Color Footage (Item 2020600759).
1. Uses extracted 4-second frames from data/loc_ww2_frames/ (598 frames).
2. Generates dense visual captions via gpt-4o-mini (detail: low) with concurrency and caching.
3. Generates 768-dimensional normalized embeddings via text-embedding-3-small (dimensions=768).
4. Exports Pinecone Serverless bulk-import Parquet file into data/pinecone_export/loc_ww2_color/.
5. Upserts/imports vectors directly into Pinecone index under namespace 'loc-ww2-color'.
"""

import asyncio
import base64
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI, OpenAI
from pinecone import Pinecone

# Load environment
def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

load_env()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "rick-morty-gemma2-video")
NAMESPACE = "loc-ww2-color"

VIDEO_ID = "2020600759"
VIDEO_TITLE = "World War II color footage -- Stevens and SPECOU in Berlin; Stevens in North Africa and Egypt before D-Day"
LOC_ITEM_URL = "https://www.loc.gov/item/2020600759/"
LOC_MP4_URL = "https://tile.loc.gov/storage-services/service/mbrs/ntscrm/02531189/02531189.mp4"

FRAMES_DIR = Path("data/loc_ww2_frames")
CAPTIONS_CACHE_FILE = Path("data/loc_ww2_captions.json")
PARQUET_OUTPUT_PATH = Path("data/pinecone_export/loc_ww2_color/ww2_color_stevens_embeddings.parquet")


def format_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


async def caption_single_frame(
    client: AsyncOpenAI,
    frame_path: Path,
    frame_idx: int,
    start_s: float,
    end_s: float,
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with sem:
        with open(frame_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            "Describe this historical WWII video frame in 1-2 concise, highly factual sentences. "
            "Identify key visual elements: soldiers, uniforms, ranks, weapons, tanks, vehicles, aircraft, "
            "historic landmarks (Sphinx, Pyramids, Reichstag, Olympiastadion), ruins, rubble, civilians, refugees, or activities."
        )

        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"}}
                    ]
                }],
                max_tokens=70,
            )
            caption = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[!] Error on frame {frame_idx}: {e}")
            caption = "Historical World War II archival footage captured by George Stevens and Army Signal Corps."

        return {
            "clip_index": frame_idx,
            "id": f"ww2_stevens_{frame_idx:04d}",
            "frame_filename": frame_path.name,
            "start_time_s": round(start_s, 2),
            "end_time_s": round(end_s, 2),
            "mid_time_s": round((start_s + end_s) / 2.0, 2),
            "timestamp": f"{format_timestamp(start_s)} - {format_timestamp(end_s)}",
            "caption": caption,
        }


async def generate_all_captions(frames: List[Path], interval_s: float = 4.0) -> List[Dict[str, Any]]:
    cached = {}
    if CAPTIONS_CACHE_FILE.exists():
        with open(CAPTIONS_CACHE_FILE) as f:
            try:
                cached_list = json.load(f)
                cached = {item["frame_filename"]: item for item in cached_list}
                print(f"[*] Loaded {len(cached)} cached captions from {CAPTIONS_CACHE_FILE}")
            except Exception:
                cached = {}

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    sem = asyncio.Semaphore(25)  # 25 concurrent requests

    tasks = []
    need_fetch_indices = []

    for i, fpath in enumerate(frames):
        fname = fpath.name
        start_s = i * interval_s
        end_s = (i + 1) * interval_s
        if fname in cached:
            continue
        need_fetch_indices.append((i, fpath, start_s, end_s))
        tasks.append(caption_single_frame(client, fpath, i + 1, start_s, end_s, sem))

    if tasks:
        print(f"[*] Fetching visual captions for {len(tasks)} frames with concurrency 25...")
        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks)
        print(f"[+] Captions generated in {time.perf_counter() - t0:.2f} seconds.")
        for r in results:
            cached[r["frame_filename"]] = r

    # Build ordered list
    ordered = []
    for i, fpath in enumerate(frames):
        fname = fpath.name
        start_s = i * interval_s
        end_s = (i + 1) * interval_s
        if fname in cached:
            item = cached[fname]
            item["clip_index"] = i + 1
            item["id"] = f"ww2_stevens_{i+1:04d}"
            item["start_time_s"] = round(start_s, 2)
            item["end_time_s"] = round(end_s, 2)
            item["timestamp"] = f"{format_timestamp(start_s)} - {format_timestamp(end_s)}"
            ordered.append(item)

    # Save cache
    CAPTIONS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CAPTIONS_CACHE_FILE, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"[+] Saved all {len(ordered)} captions to {CAPTIONS_CACHE_FILE}")

    return ordered


def generate_embeddings_768(captions: List[Dict[str, Any]]) -> List[List[float]]:
    """Embeds captions using OpenAI text-embedding-3-small with dimensions=768 in batches."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    texts = [c["caption"] for c in captions]
    batch_size = 100
    all_embeddings = []

    print(f"[*] Generating 768-dim embeddings for {len(texts)} captions in batches of {batch_size}...")
    t0 = time.perf_counter()

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small",
            dimensions=768,
        )
        chunk_embeddings = [item.embedding for item in resp.data]
        all_embeddings.extend(chunk_embeddings)
        print(f"    - Processed {min(i + batch_size, len(texts))}/{len(texts)} embeddings...")

    print(f"[+] Generated {len(all_embeddings)} embeddings in {time.perf_counter() - t0:.2f}s.")
    return all_embeddings


def export_parquet(captions: List[Dict[str, Any]], embeddings: List[List[float]], output_path: Path):
    """Exports Pinecone Serverless bulk-import compliant Parquet file."""
    ids = []
    values = []
    metadatas = []

    for c, emb in zip(captions, embeddings):
        ids.append(c["id"])
        values.append(emb)

        meta_dict = {
            "video_id": VIDEO_ID,
            "title": VIDEO_TITLE,
            "start_time_s": float(c["start_time_s"]),
            "end_time_s": float(c["end_time_s"]),
            "timestamp": c["timestamp"],
            "clip_index": int(c["clip_index"]),
            "media_type": "historical_video_clip_4s",
            "local_frame_file": c["frame_filename"],
            "caption": c["caption"],
            "loc_url": LOC_ITEM_URL,
            "stream_url": LOC_MP4_URL,
        }
        metadatas.append(json.dumps(meta_dict))

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path), compression="snappy")
    print(f"\n[+] Exported Pinecone Parquet file to: {output_path}")
    print(f"    - Records: {len(table):,}")
    print(f"    - Dimension: {len(values[0])}")
    print(f"    - Size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
    return table


def import_into_pinecone(captions: List[Dict[str, Any]], embeddings: List[List[float]], namespace: str = NAMESPACE):
    """Imports vectors into Pinecone under the dedicated namespace."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    records = []
    for c, emb in zip(captions, embeddings):
        meta = {
            "video_id": VIDEO_ID,
            "title": VIDEO_TITLE,
            "start_time_s": float(c["start_time_s"]),
            "end_time_s": float(c["end_time_s"]),
            "timestamp": c["timestamp"],
            "clip_index": int(c["clip_index"]),
            "media_type": "historical_video_clip_4s",
            "local_frame_file": c["frame_filename"],
            "caption": c["caption"],
            "loc_url": LOC_ITEM_URL,
            "stream_url": LOC_MP4_URL,
        }
        records.append({
            "id": c["id"],
            "values": emb,
            "metadata": meta,
        })

    batch_size = 100
    print(f"\n[*] Upserting {len(records)} records into Pinecone namespace '{namespace}' (batches of {batch_size})...")
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        index.upsert(vectors=chunk, namespace=namespace)
        print(f"    - Upserted {min(i + batch_size, len(records))}/{len(records)} to namespace '{namespace}'...")

    time.sleep(3)
    stats = index.describe_index_stats()
    print(f"\n[+] Pinecone Index Stats Updated:")
    print(f"    - Total Vector Count: {stats.total_vector_count}")
    print(f"    - Namespaces: {stats.namespaces}")


async def main():
    print("=" * 85)
    print("LIBRARY OF CONGRESS WWII COLOR FOOTAGE EMBEDDING & PINECONE PIPELINE")
    print("=" * 85)
    print(f"Video:       {VIDEO_TITLE}")
    print(f"Source Item: {LOC_ITEM_URL}")
    print(f"Namespace:   {NAMESPACE}")
    print("-" * 85)

    frames = sorted(list(FRAMES_DIR.glob("*.jpg")), key=lambda p: int(p.stem.split("_")[1]))
    print(f"[+] Found {len(frames)} extracted frames in {FRAMES_DIR}")
    if not frames:
        raise RuntimeError("No frames found. Please run ffmpeg extraction first.")

    # 1. Generate captions
    captions = await generate_all_captions(frames, interval_s=4.0)

    # 2. Embed captions in 768 dimensions
    embeddings = generate_embeddings_768(captions)

    # 3. Export Parquet
    export_parquet(captions, embeddings, PARQUET_OUTPUT_PATH)

    # 4. Import into Pinecone in own namespace
    import_into_pinecone(captions, embeddings, namespace=NAMESPACE)

    print("\n" + "=" * 85)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    asyncio.run(main())
