"""
Helper script to inspect, download, and prepare Gemma 2 embedding model weights
for deployment to Fireworks AI.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

SUPPORTED_TARGETS = {
    "bge-gemma2": {
        "repo_id": "BAAI/bge-multilingual-gemma2",
        "description": "Full-size Gemma 2 text embedding model (9.24B parameters, based on Gemma 2 9B)",
        "params": "9.24B",
        "dim": 3584,
        "recommended_shape": "NVIDIA_H100_80GB",
    },
    "gemma2embedding-fp16": {
        "repo_id": "tiendung/gemma2embedding",
        "description": "Full-size Gemma 2 text embedding FP16 checkpoint (9.24B parameters)",
        "params": "9.24B",
        "dim": 3584,
        "recommended_shape": "NVIDIA_H100_80GB",
    },
    "embeddinggemma-2": {
        "repo_id": "google/embeddinggemma-2",
        "description": "Google DeepMind Multimodal Embedding model (744M parameters total)",
        "params": "744M",
        "dim": 768,
        "recommended_shape": "NVIDIA_H100_80GB",
    },
}


def inspect_model(repo_id: str):
    """Check repository details and file tree on Hugging Face."""
    api = HfApi()
    print(f"\n[*] Inspecting repository: {repo_id}")
    try:
        info = api.model_info(repo_id)
        print(f"    - Downloads: {info.downloads:,}")
        print(f"    - Likes: {info.likes}")
        print(f"    - Pipeline Tag: {info.pipeline_tag}")

        files = api.list_repo_files(repo_id)
        safetensors = [f for f in files if f.endswith(".safetensors")]
        configs = [f for f in files if "config" in f or "tokenizer" in f]

        print(f"    - Weight files: {len(safetensors)} safetensors found")
        print(f"    - Config files: {len(configs)} found ({', '.join(configs[:5])}...)")
        return True
    except Exception as e:
        print(f"    [!] Error fetching repo details: {e}")
        return False


def download_model(repo_id: str, output_dir: str, config_only: bool = False):
    """Download weights or config files to target directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ignore_patterns = ["*.pt", "*.bin"]
    if config_only:
        ignore_patterns.append("*.safetensors")
        print(f"\n[*] Downloading CONFIG-ONLY for {repo_id} to {output_dir}...")
    else:
        print(f"\n[*] Downloading FULL CHECKPOINT for {repo_id} to {output_dir} (this may take several minutes)...")

    downloaded = snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_path),
        ignore_patterns=ignore_patterns,
    )
    print(f"[+] Download complete: {downloaded}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Prepare Gemma 2 embedding models for Fireworks deployment")
    parser.add_argument(
        "--target",
        choices=list(SUPPORTED_TARGETS.keys()),
        default="bge-gemma2",
        help="Model preset to prepare",
    )
    parser.add_argument("--repo-id", type=str, default=None, help="Custom HF repository ID")
    parser.add_argument("--download-dir", type=str, default="./model_weights", help="Directory to save downloaded files")
    parser.add_argument("--config-only", action="store_true", help="Download only configs (skip multi-GB weight files)")
    parser.add_argument("--inspect-only", action="store_true", help="Only inspect repo metadata without downloading")

    args = parser.parse_args()

    target_info = SUPPORTED_TARGETS.get(args.target, {})
    repo_id = args.repo_id or target_info.get("repo_id", "BAAI/bge-multilingual-gemma2")

    print("=" * 70)
    print("GEMMA 2 EMBEDDING PREPARATION TOOL FOR FIREWORKS AI")
    print("=" * 70)
    print(f"Target Preset:       {args.target}")
    print(f"HF Repository:       {repo_id}")
    if target_info:
        print(f"Model Description:   {target_info['description']}")
        print(f"Parameters:          {target_info['params']}")
        print(f"Recommended GPU:     {target_info['recommended_shape']}")
    print("=" * 70)

    success = inspect_model(repo_id)
    if not success or args.inspect_only:
        return

    download_model(repo_id, args.download_dir, config_only=args.config_only)

    print("\nNext step: Upload and deploy to Fireworks using deploy.sh:")
    print(f"  ./deploy.sh {args.target} {args.download_dir} https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
