#!/usr/bin/env python3
"""
Uploads the WWII Color Film Embeddings Parquet dataset and frame assets
to Hugging Face Dataset repository 'astr010/loc-ww2-color-film-gemma2'.
"""

import os
import tarfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "astr010/loc-ww2-color-film-gemma2"

def main():
    api = HfApi()

    print(f"[*] Ensuring public Hugging Face Dataset repository exists: {REPO_ID}")
    create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, private=False)

    # 1. Package frames into tar.gz for efficient single-file download in Spaces
    tar_path = Path("data/loc_ww2_frames.tar.gz")
    if not tar_path.exists():
        print("[*] Creating compressed archive of 598 keyframe JPEGs...")
        with tarfile.open(tar_path, "w:gz") as tar:
            for f in sorted(Path("data/loc_ww2_frames").glob("*.jpg")):
                tar.add(f, arcname=f"loc_ww2_frames/{f.name}")
        print(f"[+] Created {tar_path} ({tar_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # 2. Upload Parquet dataset
    parquet_path = "data/loc_ww2_color_embeddings.parquet"
    print(f"[*] Uploading {parquet_path}...")
    api.upload_file(
        path_or_fileobj=parquet_path,
        path_in_repo="loc_ww2_color_embeddings.parquet",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    # 3. Upload frames archive
    print(f"[*] Uploading {tar_path}...")
    api.upload_file(
        path_or_fileobj=str(tar_path),
        path_in_repo="assets/loc_ww2_frames.tar.gz",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    # 4. Upload video metadata JSON
    meta_path = "data/video_metadata_2020600759.json"
    if os.path.exists(meta_path):
        print(f"[*] Uploading {meta_path}...")
        api.upload_file(
            path_or_fileobj=meta_path,
            path_in_repo="metadata/video_metadata_2020600759.json",
            repo_id=REPO_ID,
            repo_type="dataset",
        )

    # 5. Create Dataset README
    readme_content = """---
license: apache-2.0
task_categories:
  - feature-extraction
  - sentence-similarity
tags:
  - multimodal
  - video
  - embeddinggemma-2
  - library-of-congress
  - world-war-2
size_categories:
  - n<1K
---

# Library of Congress WWII Color Film - EmbeddingGemma 2 Multimodal Dataset

**LCCN**: [2020600759](https://www.loc.gov/item/2020600759/)  
**Title**: World War II color footage--Stevens and SPECOU in Berlin; Stevens in North Africa and Egypt before D-Day  
**Director / Unit**: George Stevens (Lt. Col., U.S. Army Signal Corps) / SPECOU  
**Dates**: 1943 - 1945  
**National Film Registry**: Selected as an essential visual record of World War II  
**Duration**: 39 minutes, 53.5 seconds (2,393.5 seconds)  
**Total Keyframes**: 598 frames (extracted at 0.25 fps / 1 frame per 4 seconds)  
**Embedding Model**: [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (744M parameters, 768 dimensions)  

## Dataset Structure
- `loc_ww2_color_embeddings.parquet`: 598 rows containing `id`, `values` (float32[768] vector), and `metadata` (JSON with timestamp, timecode, archival metadata, and frame references).
- `assets/loc_ww2_frames.tar.gz`: Archive containing all 598 high-resolution 480x360 keyframe JPEGs.
- `metadata/video_metadata_2020600759.json`: Full cataloging and technical metadata from the Library of Congress.
"""
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    print(f"\n[+] Dataset successfully published to: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()
