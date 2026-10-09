"""
Local Zero-Cost Text Query Encoder and Vector Search.
Generates 768-dimensional text embeddings locally (or via WASM/ONNX)
using the 270M text backbone of EmbeddingGemma 2, matching the video vectors
stored in the Pinecone Parquet dataset with $0.00 GPU spend.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pyarrow.parquet as pq


def load_parquet_index(parquet_path: str):
    """Loads vectors and metadata from Pinecone Parquet export."""
    table = pq.read_table(parquet_path)
    ids = table["id"].to_pylist()
    vectors = np.array(table["values"].to_pylist(), dtype=np.float32)
    metadatas = [json.loads(m) for m in table["metadata"].to_pylist()]
    return ids, vectors, metadatas


def generate_local_text_embedding(query_text: str, mock: bool = True) -> np.ndarray:
    """
    Encodes query text using EmbeddingGemma 2's SearchQuery prefix:
    'task: search result | query: {query}'

    Uses the 270M text-only encoder (config_kwargs={'vision_config': None, 'audio_config': None}).
    """
    prefixed_query = f"task: search result | query: {query_text}"

    if mock:
        # Mock normalized 768-dim vector for testing
        np.random.seed(abs(hash(query_text)) % (2**32))
        vec = np.random.randn(768).astype(np.float32)
        return vec / np.linalg.norm(vec)

    # Real local execution via sentence-transformers (CPU / Apple Silicon Metal)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    emb = model.encode(prefixed_query, prompt_name="SearchQuery", normalize_embeddings=True)
    return np.array(emb, dtype=np.float32)


def search_video_clips(
    query_text: str,
    parquet_path: str,
    top_k: int = 5,
    mock: bool = True,
):
    print("=" * 80)
    print("LOCAL TEXT-TO-VIDEO SEARCH (ZERO CLOUD GPU SPEND)")
    print("=" * 80)
    print(f"Query:        '{query_text}'")
    print(f"Index File:   {parquet_path}")

    # 1. Load Parquet database
    ids, vectors, metadatas = load_parquet_index(parquet_path)
    print(f"Loaded:       {len(ids)} video clips (768-dim dense vectors)")

    # 2. Encode text locally
    t0 = time.perf_counter()
    query_vec = generate_local_text_embedding(query_text, mock=mock)
    encode_time = (time.perf_counter() - t0) * 1000.0

    print(f"Encoded in:   {encode_time:.1f} ms (executed locally on CPU/WASM)")
    print(f"GPU Cost:     $0.000000 (100% Free - No server spin-up)")
    print("-" * 80)

    # 3. Compute Cosine Similarities: dot product since vectors are L2-normalized
    similarities = np.dot(vectors, query_vec)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    print(f"TOP {top_k} MATCHING VIDEO SEGMENTS:")
    print("-" * 80)
    for rank, idx in enumerate(top_indices, 1):
        score = similarities[idx]
        meta = metadatas[idx]
        print(f"#{rank} [Score: {score:.4f}] {meta['timestamp']}")
        print(f"    - Clip ID:   {ids[idx]}")
        print(f"    - Direct URL: {meta['youtube_url']}")
        print(f"    - Interval:  {meta['start_time_s']}s - {meta['end_time_s']}s")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local text query search against video parquet index")
    parser.add_argument("--query", type=str, default="Rick shooting alien lasers in space battle")
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/pinecone_export/default/rick_and_morty_s9_4s_embeddings.parquet",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--live", action="store_true", help="Load real model weights locally")

    args = parser.parse_args()
    search_video_clips(
        query_text=args.query,
        parquet_path=args.parquet,
        top_k=args.top_k,
        mock=not args.live,
    )
