"""
FastAPI Server & Video Search UI for Library of Congress WWII Color Film Archive.
Powered by Google DeepMind EmbeddingGemma 2 (744M parameters) multimodal embeddings,
running 100% self-contained with in-memory NumPy cosine similarity search (zero external vector database, zero API keys),
and streaming 1440x1080 HD archival video directly from the Library of Congress.
"""

import json
import os
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# --- DIRECTORIES & ASSET CONFIGURATION ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FRAMES_DIR = DATA_DIR / "loc_ww2_frames"
LOCAL_VIDEO_FILE = DATA_DIR / "videos" / "ww2_color_stevens_2020600759.mp4"
METADATA_FILE = DATA_DIR / "video_metadata_2020600759.json"
EMBEDDINGS_FILE = DATA_DIR / "loc_ww2_color_embeddings.parquet"
FRAMES_TARBALL = DATA_DIR / "loc_ww2_frames.tar.gz"

HF_DATASET_REPO = "astr010/loc-ww2-color-film-gemma2"
LOC_CDN_STREAM_URL = "https://tile.loc.gov/storage-services/service/mbrs/ntscrm/02531189/02531189.mp4"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def ensure_data_assets():
    """Ensures embeddings, metadata, and frames exist locally, downloading from HF Dataset if needed."""
    # 1. Video Metadata
    if not METADATA_FILE.exists():
        try:
            print(f"[*] Downloading metadata from Hugging Face ({HF_DATASET_REPO})...")
            dl_path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename="metadata/video_metadata_2020600759.json",
                repo_type="dataset",
            )
            with open(dl_path, "r") as f_in, open(METADATA_FILE, "w") as f_out:
                f_out.write(f_in.read())
            print("[+] Video metadata downloaded.")
        except Exception as e:
            print(f"[!] Warning downloading metadata: {e}")

    # 2. Parquet Embeddings
    if not EMBEDDINGS_FILE.exists():
        try:
            print(f"[*] Downloading embeddings from Hugging Face ({HF_DATASET_REPO})...")
            dl_path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename="loc_ww2_color_embeddings.parquet",
                repo_type="dataset",
            )
            with open(dl_path, "rb") as f_in, open(EMBEDDINGS_FILE, "wb") as f_out:
                f_out.write(f_in.read())
            print("[+] Parquet embeddings downloaded.")
        except Exception as e:
            print(f"[!] Warning downloading embeddings: {e}")

    # 3. Keyframes Archive
    existing_frames = list(FRAMES_DIR.glob("*.jpg"))
    if len(existing_frames) < 500:
        if FRAMES_TARBALL.exists():
            print(f"[*] Extracting existing tarball: {FRAMES_TARBALL}...")
            with tarfile.open(FRAMES_TARBALL, "r:gz") as tar:
                tar.extractall(path=DATA_DIR)
        else:
            try:
                print(f"[*] Downloading frames archive from Hugging Face ({HF_DATASET_REPO})...")
                dl_tar = hf_hub_download(
                    repo_id=HF_DATASET_REPO,
                    filename="assets/loc_ww2_frames.tar.gz",
                    repo_type="dataset",
                )
                with tarfile.open(dl_tar, "r:gz") as tar:
                    tar.extractall(path=DATA_DIR)
            except Exception as e:
                print(f"[!] Warning extracting frames archive: {e}")
        print(f"[+] Total keyframes ready: {len(list(FRAMES_DIR.glob('*.jpg')))}")


ensure_data_assets()

# --- LOAD METADATA & PARQUET EMBEDDINGS ---
VIDEO_METADATA: Dict[str, Any] = {}
if METADATA_FILE.exists():
    with open(METADATA_FILE, "r") as f:
        VIDEO_METADATA = json.load(f)

if not EMBEDDINGS_FILE.exists():
    raise RuntimeError(f"Embeddings file {EMBEDDINGS_FILE} could not be loaded or downloaded.")

print(f"[*] Loading embeddings table from {EMBEDDINGS_FILE}...")
df_embeddings = pd.read_parquet(EMBEDDINGS_FILE)
raw_matrix = np.array(df_embeddings["values"].tolist(), dtype=np.float32)

# L2-normalize vectors so cosine similarity is a fast single matrix-vector dot product
row_norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
V_NORM = raw_matrix / np.maximum(row_norms, 1e-9)

METADATA_ROWS: List[Dict[str, Any]] = [
    json.loads(meta_str) if isinstance(meta_str, str) else meta_str
    for meta_str in df_embeddings["metadata"]
]
TOTAL_VECTORS = len(METADATA_ROWS)
print(f"[+] Loaded {TOTAL_VECTORS} vectors (shape {V_NORM.shape}, {V_NORM.nbytes / 1024:.1f} KB in RAM).")

# --- INITIALIZE EMBEDDINGGEMMA 2 MODEL ---
# Prioritize local staged directory if available, otherwise fetch from Hugging Face
LOCAL_WEIGHTS_DIR = BASE_DIR / "model_weights" / "embeddinggemma-2"
if LOCAL_WEIGHTS_DIR.exists() and (LOCAL_WEIGHTS_DIR / "model.safetensors").exists():
    MODEL_SOURCE = str(LOCAL_WEIGHTS_DIR)
    print(f"[*] Loading EmbeddingGemma 2 from local weights: {MODEL_SOURCE}")
else:
    MODEL_SOURCE = "google/embeddinggemma-2"
    print(f"[*] Loading EmbeddingGemma 2 from Hugging Face Hub: {MODEL_SOURCE}")

t0_model = time.perf_counter()
model = SentenceTransformer(MODEL_SOURCE, device="cpu")
print(f"[+] Model loaded in {time.perf_counter() - t0_model:.2f}s on CPU.")

# Warm up query encoder
_ = model.encode("task: search result | query: warm up", device="cpu", normalize_embeddings=True)

# --- FASTAPI APPLICATION ---
app = FastAPI(
    title="Library of Congress WWII Color Footage - Semantic Video Search",
    description="Natural-language video search powered by Google DeepMind EmbeddingGemma 2 and in-memory cosine similarity.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local keyframe images
if FRAMES_DIR.exists():
    app.mount("/frames/ww2", StaticFiles(directory=str(FRAMES_DIR)), name="frames_ww2")


@app.get("/video/stream")
@app.head("/video/stream")
async def stream_video(range: Optional[str] = Header(None)):
    """
    Streams local video if available on disk (with HTTP 206 Partial Content range seeking),
    otherwise seamlessly redirects to the official high-speed Library of Congress CDN MP4 stream.
    """
    if not LOCAL_VIDEO_FILE.exists():
        return RedirectResponse(url=LOC_CDN_STREAM_URL, status_code=307)

    file_size = LOCAL_VIDEO_FILE.stat().st_size
    start = 0
    end = file_size - 1

    if range:
        bytes_part = range.replace("bytes=", "").split("-")
        start = int(bytes_part[0]) if bytes_part[0] else 0
        end = int(bytes_part[1]) if len(bytes_part) > 1 and bytes_part[1] else file_size - 1

    chunk_size = (end - start) + 1

    def iter_file():
        with open(LOCAL_VIDEO_FILE, "rb") as f:
            f.seek(start)
            bytes_left = chunk_size
            while bytes_left > 0:
                read_bytes = min(1024 * 512, bytes_left)
                data = f.read(read_bytes)
                if not data:
                    break
                bytes_left -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Content-Type": "video/mp4",
    }
    return StreamingResponse(iter_file(), status_code=206 if range else 200, headers=headers)


@app.get("/api/metadata")
async def get_metadata():
    """Returns the comprehensive Library of Congress cataloging and technical video stream metadata."""
    if not VIDEO_METADATA:
        raise HTTPException(status_code=404, detail="Metadata not found.")
    return VIDEO_METADATA


@app.get("/api/stats")
async def get_stats():
    """Returns in-memory vector index stats, model details, and archival metadata."""
    catalog = VIDEO_METADATA.get("catalog", {})
    technical = VIDEO_METADATA.get("technical_stream", {})
    frame_count = len(list(FRAMES_DIR.glob("*.jpg"))) if FRAMES_DIR.exists() else 0

    return {
        "model": "google/embeddinggemma-2",
        "model_architecture": "EmbeddingGemma2Model (744M multimodal / 270M text)",
        "dimension": 768,
        "total_vectors": TOTAL_VECTORS,
        "metric": "cosine",
        "storage": f"In-Memory NumPy array ({V_NORM.nbytes / 1024:.1f} KB)",
        "hosting": "Hugging Face Spaces (CPU tier)",
        "external_database": "None (Self-contained in-memory search)",
        "video": {
            "lccn": catalog.get("lccn", "2020600759"),
            "title": catalog.get("title", "World War II color footage"),
            "director": "George Stevens (Lt. Col., U.S. Army Signal Corps)",
            "unit": "Special Coverage Unit (SPECOU)",
            "cinematographer": "William C. Mellor",
            "dates": catalog.get("date_display", "1943 - 1945"),
            "national_film_registry": catalog.get("national_film_registry", True),
            "duration": technical.get("duration_formatted", "39m 54s"),
            "resolution": "1440x1080 (HD 4:3)",
            "codec": "H.264 / AAC 24fps",
            "granularity": "1 frame every 4.0 seconds (598 clips)",
            "loc_url": catalog.get("item_url", "https://www.loc.gov/item/2020600759/"),
            "cdn_stream_url": LOC_CDN_STREAM_URL,
            "local_frames_available": frame_count,
            "has_local_video": LOCAL_VIDEO_FILE.exists(),
        },
    }


@app.get("/api/search")
async def search_video(
    q: str = Query(..., description="Natural language search query"),
    top_k: int = Query(9, ge=1, le=48),
):
    """
    1. Encodes query using EmbeddingGemma 2 on CPU with task prefix
    2. Performs instant in-memory NumPy matrix dot-product (cosine similarity)
    3. Returns top-k matching scenes with timecodes, LOC stream seeking, and archival metadata
    """
    cleaned_q = q.strip()
    if not cleaned_q:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    formatted_query = f"task: search result | query: {cleaned_q}"

    # 1. Embed query
    t0 = time.perf_counter()
    try:
        q_vec = model.encode(formatted_query, device="cpu", normalize_embeddings=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")
    embed_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # 2. In-memory cosine similarity
    t1 = time.perf_counter()
    sims = np.dot(V_NORM, q_vec)
    top_indices = np.argsort(sims)[::-1][:top_k]
    search_time_ms = round((time.perf_counter() - t1) * 1000.0, 3)
    total_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # 3. Format matches
    matches = []
    for rank, idx in enumerate(top_indices):
        rec = METADATA_ROWS[idx]
        t_sec = float(rec.get("timestamp_sec", 0.0))
        timecode = rec.get("timecode", "00:00:00")
        frame_file = rec.get("frame_file") or f"frame_{idx:04d}.jpg"
        clip_idx = int(rec.get("frame_idx", idx))

        # Archival contextual note based on historical timeline
        if t_sec < 1440.0:  # First 24 minutes: Berlin 1945
            location_tag = "Berlin, Germany (Summer 1945) — SPECOU Coverage"
        else:  # Second half: North Africa & Egypt 1943
            location_tag = "North Africa & Egypt (1943) — Desert Maneuvers & Giza"

        matches.append({
            "id": f"loc_ww2_{clip_idx:04d}",
            "score": round(float(sims[idx]), 4),
            "clip_index": clip_idx,
            "timestamp": timecode,
            "start_time_s": t_sec,
            "end_time_s": round(t_sec + 4.0, 2),
            "frame_url": f"/frames/ww2/{frame_file}",
            "frame_filename": frame_file,
            "source_link": rec.get("loc_url", "https://www.loc.gov/item/2020600759/"),
            "director": rec.get("director", "George Stevens"),
            "military_unit": rec.get("military_unit", "U.S. Army Signal Corps SPECOU"),
            "resolution": "1440x1080 Kodachrome Color",
            "historical_context": location_tag,
            "stream_url": LOC_CDN_STREAM_URL,
        })

    return {
        "query": cleaned_q,
        "model": "google/embeddinggemma-2",
        "timing": {
            "embed_time_ms": embed_time_ms,
            "search_time_ms": search_time_ms,
            "total_time_ms": total_time_ms,
        },
        "total_matches": len(matches),
        "results": matches,
    }


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the interactive Video Search UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Library of Congress WWII Color Video Search | EmbeddingGemma 2</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
        body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #080c15; color: #f1f5f9; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .amber-glow { box-shadow: 0 0 35px rgba(245, 158, 11, 0.2); }
        .card-hover:hover { transform: translateY(-4px); box-shadow: 0 12px 30px rgba(245, 158, 11, 0.15); }
        .score-pill { background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%); }
    </style>
</head>
<body class="min-h-screen flex flex-col justify-between">
    <!-- Navbar -->
    <header class="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-amber-500/20 border border-amber-500/50 flex items-center justify-center text-amber-400 text-lg font-bold shadow-md">
                    <i class="fa-solid fa-film"></i>
                </div>
                <div>
                    <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                        LOC Moving Image Semantic Search
                        <span class="text-xs px-2.5 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30">EmbeddingGemma 2 • In-Memory</span>
                    </h1>
                    <p class="text-xs text-slate-400">Library of Congress • WWII Color Archival Footage (LCCN 2020600759)</p>
                </div>
            </div>
            
            <!-- Controls & Metadata Trigger -->
            <div class="flex items-center space-x-3 text-xs">
                <!-- Archival Metadata Button -->
                <button onclick="openMetadataModal()" 
                    class="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-amber-400 font-semibold border border-amber-500/30 transition flex items-center gap-1.5 shadow">
                    <i class="fa-solid fa-circle-info"></i>
                    <span>Film Metadata</span>
                </button>

                <div class="bg-slate-800/80 px-3 py-2 rounded-lg border border-slate-700 flex items-center gap-2">
                    <i class="fa-solid fa-bolt text-amber-400"></i>
                    <span id="statVectors" class="mono font-semibold">598 Vectors In-Memory</span>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-6 py-8 flex-1 w-full">
        <!-- Film Quick Metadata Banner -->
        <div class="max-w-4xl mx-auto mb-8 bg-slate-900/60 border border-slate-800/90 rounded-2xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs">
            <div class="flex items-center gap-3">
                <span class="px-2.5 py-1 rounded-md bg-amber-500/20 text-amber-300 font-bold border border-amber-500/40 mono">LCCN 2020600759</span>
                <span class="text-slate-300"><i class="fa-solid fa-user-tie text-amber-400 mr-1"></i> Dir: <b>George Stevens</b> (SPECOU)</span>
                <span class="text-slate-400 hidden sm:inline">•</span>
                <span class="text-slate-300 hidden sm:inline"><i class="fa-solid fa-camera text-blue-400 mr-1"></i> <b>William C. Mellor</b></span>
                <span class="text-slate-400 hidden md:inline">•</span>
                <span class="text-slate-300 hidden md:inline"><i class="fa-solid fa-calendar text-emerald-400 mr-1"></i> <b>1943–1945</b></span>
            </div>
            <div class="flex items-center gap-2">
                <span class="px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700 font-medium">1440x1080 HD Kodachrome</span>
                <span class="px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-medium">National Film Registry</span>
                <a href="https://www.loc.gov/item/2020600759/" target="_blank" class="text-amber-400 hover:underline flex items-center gap-1 font-semibold">
                    <span>LOC.gov</span>
                    <i class="fa-solid fa-arrow-up-right-from-square text-[10px]"></i>
                </a>
            </div>
        </div>

        <!-- Hero Section -->
        <div class="text-center max-w-3xl mx-auto mb-10">
            <h2 class="text-4xl font-extrabold tracking-tight mb-3 text-white">
                Search 40 Minutes of <span class="text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-orange-400">WWII Color Footage</span>
            </h2>
            <p class="text-slate-400 text-sm mb-6">
                Direct natural-language visual search with Google DeepMind <b>EmbeddingGemma 2</b> and zero-latency in-memory vector search.
            </p>

            <!-- Search Form -->
            <form id="searchForm" class="relative max-w-2xl mx-auto mb-4">
                <div class="relative flex items-center">
                    <i class="fa-solid fa-magnifying-glass absolute left-4 text-amber-400 text-lg"></i>
                    <input type="text" id="searchInput" 
                        class="w-full pl-12 pr-32 py-4 bg-slate-900 border border-slate-700 rounded-2xl text-white placeholder-slate-500 focus:outline-none focus:border-amber-500 amber-glow transition text-base"
                        placeholder="e.g. Great Sphinx in Egypt with military personnel..." 
                        value="Great Sphinx and pyramids in Egypt" required>
                    <button type="submit" id="searchBtn"
                        class="absolute right-2 px-6 py-2.5 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-slate-950 font-bold rounded-xl transition flex items-center gap-2 text-sm shadow-lg shadow-amber-500/20">
                        <span>Search</span>
                        <i class="fa-solid fa-arrow-right"></i>
                    </button>
                </div>
            </form>

            <!-- Suggestion Pills -->
            <div id="chipsContainer" class="flex flex-wrap items-center justify-center gap-2 text-xs">
                <span class="text-slate-500">Curated queries:</span>
                <button onclick="setQuery('Great Sphinx and Pyramids of Giza in Egypt')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-amber-500 text-slate-300 hover:text-amber-400 transition">🏛️ Great Sphinx & Pyramids</button>
                <button onclick="setQuery('Bombed Reichstag ruins and destruction in Berlin')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-amber-500 text-slate-300 hover:text-amber-400 transition">💥 Reichstag Ruins</button>
                <button onclick="setQuery('Tanks and military armor driving through desert sand')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-amber-500 text-slate-300 hover:text-amber-400 transition">🛡️ Desert Tanks</button>
                <button onclick="setQuery('American pilots and aircraft on airfield')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-amber-500 text-slate-300 hover:text-amber-400 transition">✈️ Airfield Pilots</button>
                <button onclick="setQuery('Civilians and refugees carrying luggage and carts')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-amber-500 text-slate-300 hover:text-amber-400 transition">🚶 Berlin Refugees</button>
                <button onclick="setQuery('Olympiastadion Berlin stadium empty grounds')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-amber-500 text-slate-300 hover:text-amber-400 transition">🏟️ Berlin Olympic Stadium</button>
                <button onclick="setQuery('Russian Soviet soldiers marching in Berlin')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-amber-500 text-slate-300 hover:text-amber-400 transition">🎖️ Soviet Troops</button>
            </div>
        </div>

        <!-- Query Performance Timing Bar -->
        <div id="timingBar" class="hidden max-w-3xl mx-auto mb-8 bg-slate-900/90 border border-slate-800 rounded-xl px-4 py-2.5 text-xs flex items-center justify-between text-slate-400 shadow-md">
            <div class="flex items-center gap-4">
                <span><i class="fa-solid fa-bolt text-amber-400 mr-1"></i> Total: <b id="timeTotal" class="text-white mono">-</b></span>
                <span>• Gemma 2 Embed: <b id="timeEmbed" class="text-slate-300 mono">-</b></span>
                <span>• In-Memory Dot: <b id="timeQuery" class="text-slate-300 mono">-</b></span>
                <span>• Engine: <b class="text-amber-400 mono">NumPy float32</b></span>
            </div>
            <div class="text-amber-400 font-bold" id="matchesCount">0 scenes found</div>
        </div>

        <!-- Loading Spinner -->
        <div id="loading" class="hidden text-center py-16">
            <div class="inline-block animate-spin text-4xl text-amber-400 mb-3"><i class="fa-solid fa-circle-notch"></i></div>
            <p class="text-slate-400 text-sm">Encoding query with EmbeddingGemma 2 & running in-memory search...</p>
        </div>

        <!-- Video Results Grid -->
        <div id="resultsGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"></div>

        <!-- Empty State -->
        <div id="emptyState" class="hidden text-center py-20">
            <i class="fa-solid fa-film text-slate-700 text-6xl mb-4"></i>
            <h3 class="text-lg font-bold text-slate-300">No matching scenes found</h3>
            <p class="text-slate-500 text-sm">Try broader keywords or another search phrase.</p>
        </div>
    </main>

    <!-- Modal for Video Player & Scene Details -->
    <div id="modal" class="fixed inset-0 bg-black/85 backdrop-blur-md z-50 hidden flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl max-w-3xl w-full p-6 shadow-2xl relative">
            <button onclick="closeModal()" class="absolute top-4 right-4 text-slate-400 hover:text-white text-xl z-10">
                <i class="fa-solid fa-xmark"></i>
            </button>
            
            <div class="mb-4 flex items-center justify-between pr-8">
                <div>
                    <span id="modalTimestamp" class="text-xs px-2.5 py-1 rounded-md bg-amber-500/20 text-amber-400 font-bold border border-amber-500/30 mono">00:00:00</span>
                    <span id="modalScore" class="text-xs text-slate-400 ml-2 mono">Cosine: 0.000</span>
                </div>
                <div class="flex items-center gap-2">
                    <button onclick="toggleModalMedia('video')" id="btnViewVideo" class="px-3 py-1 text-xs rounded-lg bg-amber-500 text-slate-950 font-bold">Play Video</button>
                    <button onclick="toggleModalMedia('frame')" id="btnViewFrame" class="px-3 py-1 text-xs rounded-lg bg-slate-800 text-slate-300 font-semibold hover:bg-slate-700">Still Frame</button>
                </div>
            </div>
            
            <!-- Video & Frame Viewers -->
            <div id="videoContainer" class="mb-4 aspect-video w-full rounded-xl overflow-hidden border border-slate-800 bg-black relative">
                <!-- HTML5 Native Player -->
                <video id="nativePlayer" controls class="w-full h-full object-contain" preload="metadata">
                    <source id="videoSource" src="/video/stream" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>

            <div id="frameContainer" class="hidden mb-4">
                <img id="modalImg" src="" alt="Scene Frame" class="w-full rounded-xl border border-slate-800 object-cover max-h-96">
            </div>
            
            <!-- Scene Caption & Metadata -->
            <div class="bg-slate-950/80 border border-slate-800 rounded-xl p-4 mb-4">
                <div class="flex items-center justify-between mb-1">
                    <div class="text-xs text-amber-400 font-bold flex items-center gap-2">
                        <i class="fa-solid fa-landmark"></i>
                        <span>Historical Timeline Context</span>
                    </div>
                    <span id="modalMetaTag" class="text-[11px] text-slate-400 mono">George Stevens • SPECOU • 1440x1080 Color</span>
                </div>
                <p id="modalCaption" class="text-xs text-slate-200 leading-relaxed mb-3">Historical scene description...</p>
                
                <div class="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
                    <span>Captured on 16mm Kodachrome by U.S. Army Signal Corps Special Coverage Unit</span>
                    <span class="text-emerald-400"><i class="fa-solid fa-circle-check mr-1"></i> Public Domain (LOC)</span>
                </div>
            </div>

            <div class="flex items-center justify-between">
                <a id="modalSourceLink" href="https://www.loc.gov/item/2020600759/" target="_blank" 
                    class="px-5 py-2.5 bg-slate-800 hover:bg-slate-700 text-amber-400 font-bold rounded-xl text-sm flex items-center gap-2 transition border border-slate-700">
                    <i class="fa-solid fa-building-columns"></i>
                    <span>View Catalog on Library of Congress (LOC.gov)</span>
                </a>
                <button onclick="closeModal()" class="px-5 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium rounded-xl text-sm transition">
                    Close
                </button>
            </div>
        </div>
    </div>

    <!-- Archival & Technical Metadata Modal -->
    <div id="metadataModal" class="fixed inset-0 bg-black/85 backdrop-blur-md z-50 hidden flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto p-6 shadow-2xl relative">
            <button onclick="closeMetadataModal()" class="absolute top-4 right-4 text-slate-400 hover:text-white text-xl z-10">
                <i class="fa-solid fa-xmark"></i>
            </button>

            <div class="flex items-center gap-3 mb-6">
                <div class="w-10 h-10 rounded-xl bg-amber-500/20 border border-amber-500/50 flex items-center justify-center text-amber-400 text-lg">
                    <i class="fa-solid fa-landmark"></i>
                </div>
                <div>
                    <h3 class="text-lg font-bold text-white">Library of Congress Archival & Technical Metadata</h3>
                    <p class="text-xs text-slate-400">Complete Cataloging & Video Stream Encoding Properties</p>
                </div>
            </div>

            <!-- Metadata Grid -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 text-xs">
                <div class="bg-slate-950/60 border border-slate-800 p-4 rounded-xl">
                    <h4 class="font-bold text-amber-400 mb-2 uppercase tracking-wider text-[11px]">Archival Provenance</h4>
                    <ul class="space-y-1.5 text-slate-300">
                        <li><span class="text-slate-500">Title:</span> World War II color footage</li>
                        <li><span class="text-slate-500">LCCN / Item ID:</span> <span class="mono text-amber-300">2020600759</span></li>
                        <li><span class="text-slate-500">Director:</span> Lt. Col. George Stevens</li>
                        <li><span class="text-slate-500">Military Unit:</span> U.S. Army Signal Corps SPECOU</li>
                        <li><span class="text-slate-500">Cinematographer:</span> William C. Mellor</li>
                        <li><span class="text-slate-500">Date Range:</span> 1943 (Egypt) & 1945 (Berlin)</li>
                        <li><span class="text-slate-500">Collection:</span> George Stevens Jr. Collection</li>
                        <li><span class="text-slate-500">Registry:</span> National Film Registry (Librarian of Congress)</li>
                        <li><span class="text-slate-500">Rights:</span> Public Domain (U.S. Government Work)</li>
                    </ul>
                </div>

                <div class="bg-slate-950/60 border border-slate-800 p-4 rounded-xl">
                    <h4 class="font-bold text-amber-400 mb-2 uppercase tracking-wider text-[11px]">Technical Stream Specs</h4>
                    <ul class="space-y-1.5 text-slate-300">
                        <li><span class="text-slate-500">Medium:</span> 16mm Kodachrome Color Film</li>
                        <li><span class="text-slate-500">Scan Resolution:</span> 1440x1080 (HD 4:3)</li>
                        <li><span class="text-slate-500">Frame Rate:</span> 24.0 fps progressive</li>
                        <li><span class="text-slate-500">Duration:</span> 39m 53.5s (2,393.5 seconds)</li>
                        <li><span class="text-slate-500">Video Codec:</span> H.264 / AVC Main Profile</li>
                        <li><span class="text-slate-500">Audio Codec:</span> AAC (48,000 Hz, stereo)</li>
                        <li><span class="text-slate-500">Embedding Model:</span> Google DeepMind EmbeddingGemma 2</li>
                        <li><span class="text-slate-500">Vector Count:</span> 598 clips (1 frame / 4 sec)</li>
                        <li><span class="text-slate-500">Search Engine:</span> In-Memory NumPy float32 Dot Product</li>
                    </ul>
                </div>
            </div>

            <!-- Archival Summary & Historical Context -->
            <div class="bg-slate-950/60 border border-slate-800 p-4 rounded-xl mb-6 text-xs text-slate-300">
                <h4 class="font-bold text-amber-400 mb-1 uppercase tracking-wider text-[11px]">Library of Congress Catalog Summary</h4>
                <p class="leading-relaxed mb-3">
                    "In late summer 1945, Stevens and the SPECOU visit Berlin. They visit various buildings and historic sites, including the Olympic Stadium. Berlin is heavily damaged. Civilians clean up the rubble. Refugees are leaving with their belongings. Hamilton and Morse visit the Russian Sector. Multiple groups of Russian troops walk and march through town. Some recreational activities are shown--relaxing at the beach, golfing, visiting the racetrack. In the second half of the film, there are out of sequence shots of North Africa and Egypt, filmed in 1943 before D-Day. Stevens and Mellor visit the Sphinx. American military action is staged in the African desert. Unidentified troops march in the desert."
                </p>
                <div class="pt-2 border-t border-slate-800/80 text-[11px] text-slate-400">
                    <span class="text-slate-500">Production Background:</span> Ordered in late 1943 by General Dwight D. Eisenhower, George Stevens assembled 45 Hollywood cameramen and technicians (the "Stevens Irregulars") to accompany Allied soldiers from North Africa through D-Day, the liberation of Paris, and the fall of Berlin.
                </div>
            </div>

            <!-- Subject Tags -->
            <div class="mb-4">
                <h4 class="font-bold text-amber-400 mb-2 uppercase tracking-wider text-[11px]">Cataloged Historical Subjects (Click to Search)</h4>
                <div class="flex flex-wrap gap-1.5 text-xs">
                    <button onclick="setQueryAndCloseModal('Great Sphinx and Pyramids of Giza in Egypt')" class="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-amber-500 hover:text-slate-950 text-slate-300 transition">Egypt -- Giza (Sphinx)</button>
                    <button onclick="setQueryAndCloseModal('Bombed Reichstag ruins and destruction in Berlin')" class="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-amber-500 hover:text-slate-950 text-slate-300 transition">Berlin -- Reichstag Ruins</button>
                    <button onclick="setQueryAndCloseModal('Olympiastadion Berlin stadium empty grounds')" class="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-amber-500 hover:text-slate-950 text-slate-300 transition">Olympiastadion (Berlin)</button>
                    <button onclick="setQueryAndCloseModal('Tanks and military armor driving through desert sand')" class="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-amber-500 hover:text-slate-950 text-slate-300 transition">North Africa Desert Tanks</button>
                    <button onclick="setQueryAndCloseModal('Civilians and refugees carrying luggage and carts')" class="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-amber-500 hover:text-slate-950 text-slate-300 transition">Civilians in War & Refugees</button>
                    <button onclick="setQueryAndCloseModal('Russian troops marching in Soviet sector Berlin')" class="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-amber-500 hover:text-slate-950 text-slate-300 transition">Soviet Sector Troops</button>
                    <button onclick="setQueryAndCloseModal('American pilots and aircraft on airfield')" class="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-amber-500 hover:text-slate-950 text-slate-300 transition">Airfield Pilots & Aircraft</button>
                </div>
            </div>

            <div class="flex items-center justify-between pt-4 border-t border-slate-800">
                <a href="https://lccn.loc.gov/2020600759" target="_blank" class="text-amber-400 hover:underline text-xs flex items-center gap-1 font-semibold">
                    <i class="fa-solid fa-up-right-from-square"></i>
                    <span>Official LCCN Permalink (lccn.loc.gov/2020600759)</span>
                </a>
                <button onclick="closeMetadataModal()" class="px-5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold rounded-xl text-xs transition">
                    Close
                </button>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer class="border-t border-slate-800 py-6 text-center text-xs text-slate-500">
        Library of Congress Catalog Item 2020600759 • Powered by Google DeepMind EmbeddingGemma 2 • In-Memory Vector Search
    </footer>

    <script>
        async function loadStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                const total = data.total_vectors || 598;
                document.getElementById('statVectors').innerText = `${total} Vectors In-Memory`;
            } catch(e) {
                console.error("Failed to load stats", e);
            }
        }
        loadStats();

        function setQuery(text) {
            document.getElementById('searchInput').value = text;
            executeSearch();
        }

        function setQueryAndCloseModal(text) {
            closeMetadataModal();
            setQuery(text);
        }

        function openMetadataModal() {
            document.getElementById('metadataModal').classList.remove('hidden');
        }

        function closeMetadataModal() {
            document.getElementById('metadataModal').classList.add('hidden');
        }

        document.getElementById('searchForm').addEventListener('submit', (e) => {
            e.preventDefault();
            executeSearch();
        });

        async function executeSearch() {
            const query = document.getElementById('searchInput').value.trim();
            if (!query) return;

            const loading = document.getElementById('loading');
            const grid = document.getElementById('resultsGrid');
            const timingBar = document.getElementById('timingBar');
            const emptyState = document.getElementById('emptyState');

            grid.innerHTML = '';
            timingBar.classList.add('hidden');
            emptyState.classList.add('hidden');
            loading.classList.remove('hidden');

            try {
                const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&top_k=9`);
                const data = await res.json();
                loading.classList.add('hidden');

                if (data.results && data.results.length > 0) {
                    timingBar.classList.remove('hidden');
                    document.getElementById('timeTotal').innerText = `${data.timing.total_time_ms} ms`;
                    document.getElementById('timeEmbed').innerText = `${data.timing.embed_time_ms} ms`;
                    document.getElementById('timeQuery').innerText = `${data.timing.search_time_ms} ms`;
                    document.getElementById('matchesCount').innerText = `${data.total_matches} scenes found`;

                    data.results.forEach((clip, idx) => {
                        const card = document.createElement('div');
                        card.className = "bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden card-hover transition duration-200 flex flex-col justify-between";
                        
                        const frameSrc = clip.frame_url || 'https://via.placeholder.com/336x336?text=No+Frame';

                        card.innerHTML = `
                            <div class="relative cursor-pointer group" onclick="openModal('${frameSrc}', '${clip.timestamp}', '${clip.score}', '${clip.source_link}', '${clip.frame_filename}', '${clip.start_time_s}', '${encodeURIComponent(clip.historical_context)}', '${clip.director}', '${clip.resolution}')">
                                <img src="${frameSrc}" alt="Clip ${clip.clip_index}" class="w-full h-52 object-cover group-hover:scale-105 transition duration-300">
                                <div class="absolute inset-0 bg-gradient-to-t from-slate-950 via-slate-950/20 to-transparent"></div>
                                <div class="absolute top-3 left-3 bg-black/75 backdrop-blur px-2.5 py-1 rounded-md text-xs font-bold text-white border border-white/10 mono">
                                    <i class="fa-solid fa-clock text-amber-400 mr-1"></i> ${clip.timestamp}
                                </div>
                                <div class="absolute top-3 right-3 score-pill text-white px-2.5 py-0.5 rounded-full text-xs font-bold shadow-md mono">
                                    #${idx + 1} • ${clip.score.toFixed(3)}
                                </div>
                                <div class="absolute bottom-3 left-3 right-3 flex items-center justify-between text-xs text-slate-300">
                                    <span class="mono">Frame #${clip.clip_index}</span>
                                    <span class="text-amber-400 group-hover:underline flex items-center gap-1 font-semibold">
                                        <span>Play & Inspect</span>
                                        <i class="fa-solid fa-play text-[10px]"></i>
                                    </span>
                                </div>
                            </div>
                            <div class="p-4 bg-slate-950/60 border-t border-slate-800/80 flex flex-col justify-between flex-1">
                                <div class="flex items-center gap-2 text-[11px] text-amber-400/90 font-semibold mb-1">
                                    <i class="fa-solid fa-clapperboard text-[10px]"></i>
                                    <span>${clip.director} • ${clip.resolution}</span>
                                </div>
                                <p class="text-xs text-slate-300 line-clamp-2 mb-3 leading-relaxed">${clip.historical_context}</p>
                                <div class="flex items-center justify-between pt-2 border-t border-slate-900 text-xs">
                                    <span class="text-slate-500 mono">${clip.start_time_s}s - ${clip.end_time_s}s</span>
                                    <button onclick="openModal('${frameSrc}', '${clip.timestamp}', '${clip.score}', '${clip.source_link}', '${clip.frame_filename}', '${clip.start_time_s}', '${encodeURIComponent(clip.historical_context)}', '${clip.director}', '${clip.resolution}')"
                                        class="text-amber-400 hover:text-amber-300 font-semibold flex items-center gap-1 transition">
                                        <i class="fa-solid fa-circle-play"></i>
                                        <span>Watch Scene</span>
                                    </button>
                                </div>
                            </div>
                        `;
                        grid.appendChild(card);
                    });
                } else {
                    emptyState.classList.remove('hidden');
                }
            } catch (err) {
                loading.classList.add('hidden');
                alert("Search error: " + err.message);
            }
        }

        function toggleModalMedia(view) {
            const frameC = document.getElementById('frameContainer');
            const videoC = document.getElementById('videoContainer');
            const btnF = document.getElementById('btnViewFrame');
            const btnV = document.getElementById('btnViewVideo');

            if (view === 'frame') {
                frameC.classList.remove('hidden');
                videoC.classList.add('hidden');
                btnF.className = "px-3 py-1 text-xs rounded-lg bg-amber-500 text-slate-950 font-bold";
                btnV.className = "px-3 py-1 text-xs rounded-lg bg-slate-800 text-slate-300 font-semibold";
            } else {
                frameC.classList.add('hidden');
                videoC.classList.remove('hidden');
                btnV.className = "px-3 py-1 text-xs rounded-lg bg-amber-500 text-slate-950 font-bold";
                btnF.className = "px-3 py-1 text-xs rounded-lg bg-slate-800 text-slate-300 font-semibold";
            }
        }

        function openModal(imgSrc, timestamp, score, sourceLink, frameFile, startTimeSec, rawContext, director, res) {
            document.getElementById('modalImg').src = imgSrc;
            document.getElementById('modalTimestamp').innerText = timestamp;
            document.getElementById('modalScore').innerText = `Cosine: ${score}`;
            document.getElementById('modalSourceLink').href = sourceLink;
            document.getElementById('modalCaption').innerText = rawContext ? decodeURIComponent(rawContext) : "World War II 16mm Kodachrome Color Footage";
            document.getElementById('modalMetaTag').innerText = `${director || 'George Stevens'} • ${res || '1440x1080 Color'}`;

            const nativePlayer = document.getElementById('nativePlayer');
            const startSec = Math.max(0, Math.floor(parseFloat(startTimeSec) || 0));

            // Seek native player
            nativePlayer.currentTime = startSec;
            nativePlayer.play().catch(e => console.log("Autoplay waiting for user gesture"));

            toggleModalMedia('video');
            document.getElementById('modal').classList.remove('hidden');
        }

        function closeModal() {
            document.getElementById('modal').classList.add('hidden');
            const nativePlayer = document.getElementById('nativePlayer');
            nativePlayer.pause();
        }

        // Run default search on load
        executeSearch();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
