"""
FastAPI Server & Video Search UI for Library of Congress WWII Color Film Archive.
Connects Pinecone Serverless Index (rick-morty-gemma2-video) namespace 'loc-ww2-color' (598 vectors)
with OpenAI 768-dim Embeddings, local frame visualization, and HTML5 video streaming with timestamp seeking.
(Legacy namespace 'default' for Rick and Morty is preserved).
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pinecone import Pinecone

# --- CONFIGURATION (Loaded from Environment or .env) ---
def load_env_file():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

load_env_file()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "rick-morty-gemma2-video")

BASE_DIR = Path(__file__).parent
WW2_FRAMES_DIR = BASE_DIR / "data" / "loc_ww2_frames"
RM_FRAMES_DIR = BASE_DIR / "data" / "temp_4s_frames"
WW2_VIDEO_FILE = BASE_DIR / "data" / "videos" / "ww2_color_stevens_2020600759.mp4"

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY or OPENAI_API_KEY in environment or .env file.")

# Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Library of Congress WWII Color Footage - Semantic Video Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local frames directories
if WW2_FRAMES_DIR.exists():
    app.mount("/frames/ww2", StaticFiles(directory=str(WW2_FRAMES_DIR)), name="frames_ww2")
if RM_FRAMES_DIR.exists():
    app.mount("/frames/rm", StaticFiles(directory=str(RM_FRAMES_DIR)), name="frames_rm")


@app.get("/video/stream")
@app.head("/video/stream")
async def stream_video(range: Optional[str] = Header(None)):
    """Streams the local WWII video file with full HTTP Range request seeking support."""
    if not WW2_VIDEO_FILE.exists():
        raise HTTPException(status_code=404, detail="Video file not found on local disk.")

    file_size = WW2_VIDEO_FILE.stat().st_size
    start = 0
    end = file_size - 1

    if range:
        bytes_part = range.replace("bytes=", "").split("-")
        start = int(bytes_part[0]) if bytes_part[0] else 0
        end = int(bytes_part[1]) if len(bytes_part) > 1 and bytes_part[1] else file_size - 1

    chunk_size = (end - start) + 1

    def iter_file():
        with open(WW2_VIDEO_FILE, "rb") as f:
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


@app.get("/api/stats")
async def get_stats():
    """Returns Pinecone index statistics and available namespace datasets."""
    try:
        stats = index.describe_index_stats()
        ww2_frames_count = len(list(WW2_FRAMES_DIR.glob("*.jpg"))) if WW2_FRAMES_DIR.exists() else 0
        rm_frames_count = len(list(RM_FRAMES_DIR.glob("*.jpg"))) if RM_FRAMES_DIR.exists() else 0

        namespaces_info = {}
        for ns_name, ns_stat in stats.namespaces.items():
            namespaces_info[ns_name] = ns_stat.vector_count

        return {
            "index_name": INDEX_NAME,
            "dimension": stats.dimension,
            "total_vectors": stats.total_vector_count,
            "metric": stats.metric,
            "namespaces": namespaces_info,
            "active_namespace": "loc-ww2-color",
            "active_video": {
                "title": "World War II color footage -- Stevens and SPECOU in Berlin, North Africa and Egypt before D-Day",
                "item_id": "2020600759",
                "duration": "39m 54s (2,393.5 seconds)",
                "granularity": "1 frame every 4.0 seconds (598 clips)",
                "loc_url": "https://www.loc.gov/item/2020600759/",
                "local_frames_available": ww2_frames_count,
                "has_local_video": WW2_VIDEO_FILE.exists(),
            },
            "legacy_video": {
                "title": "Rick and Morty | Season 9 Battle Scenes",
                "namespace": "default",
                "local_frames_available": rm_frames_count,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search")
async def search_video(
    q: str = Query(..., description="Text query to search within video footage"),
    top_k: int = Query(9, ge=1, le=24),
    namespace: str = Query("loc-ww2-color", description="Pinecone namespace to query"),
):
    """
    1. Embeds query into 768-dimensional space using OpenAI text-embedding-3-small (dimensions=768)
    2. Queries Pinecone index under the specified namespace
    3. Resolves local frame images, metadata captions, and timestamps
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    t0 = time.perf_counter()

    # 1. Embed query
    try:
        emb_res = openai_client.embeddings.create(
            input=q.strip(),
            model="text-embedding-3-small",
            dimensions=768,
        )
        query_vector = emb_res.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI embedding error: {e}")

    embed_time_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    # 2. Query Pinecone
    t1 = time.perf_counter()
    try:
        pinecone_res = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query error: {e}")

    query_time_ms = round((time.perf_counter() - t1) * 1000.0, 1)
    total_time_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    # 3. Format matches
    matches = []
    is_ww2 = (namespace == "loc-ww2-color")

    for m in pinecone_res.matches:
        meta = m.metadata or {}
        clip_index = int(meta.get("clip_index", 0))

        if is_ww2:
            frame_filename = meta.get("local_frame_file") or f"frame_{clip_index:04d}.jpg"
            frame_url = f"/frames/ww2/{frame_filename}"
            caption = meta.get("caption", "Archival WWII footage captured by George Stevens.")
            source_link = meta.get("loc_url", "https://www.loc.gov/item/2020600759/")
        else:
            frame_filename = meta.get("local_frame_file") or f"clip_{clip_index:04d}_mid.jpg"
            frame_url = f"/frames/rm/{frame_filename}"
            caption = f"Rick and Morty Battle Scenes clip #{clip_index}"
            source_link = meta.get("youtube_url", "https://youtu.be/9Rul9N1LREQ")

        matches.append({
            "id": m.id,
            "score": round(float(m.score), 4),
            "clip_index": clip_index,
            "start_time_s": meta.get("start_time_s", 0.0),
            "end_time_s": meta.get("end_time_s", 0.0),
            "timestamp": meta.get("timestamp", "00:00:00"),
            "caption": caption,
            "frame_url": frame_url,
            "frame_filename": frame_filename,
            "source_link": source_link,
            "namespace": namespace,
        })

    return {
        "query": q,
        "namespace": namespace,
        "timing": {
            "embed_time_ms": embed_time_ms,
            "query_time_ms": query_time_ms,
            "total_time_ms": total_time_ms,
        },
        "total_matches": len(matches),
        "results": matches,
    }


@app.get("/api/ai-describe")
async def ai_describe(frame_file: str, namespace: str = "loc-ww2-color", query: str = ""):
    """Uses OpenAI GPT-4o-mini to inspect the frame and describe historical details."""
    folder = WW2_FRAMES_DIR if namespace == "loc-ww2-color" else RM_FRAMES_DIR
    frame_path = folder / frame_file
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found on local disk.")

    with open(frame_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "You are an archival film historian analyzing an authentic WWII color video frame recorded by George Stevens "
        "and the Army Signal Corps Special Coverage Unit. Describe the visual details in 2-3 sentences. "
    )
    if query:
        prompt += f"Highlight how this visual frame matches the search query: '{query}'."

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                ],
            }],
            max_tokens=220,
        )
        return {"description": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the interactive Video Search UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Library of Congress WWII Color Video Search | Pinecone Serverless</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
        body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #090d16; color: #f1f5f9; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .amber-glow { box-shadow: 0 0 35px rgba(245, 158, 11, 0.2); }
        .card-hover:hover { transform: translateY(-4px); box-shadow: 0 12px 30px rgba(245, 158, 11, 0.15); }
        .score-pill { background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%); }
    </style>
</head>
<body class="min-h-screen flex flex-col justify-between">
    <!-- Navbar -->
    <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-amber-500/20 border border-amber-500/50 flex items-center justify-center text-amber-400 text-lg font-bold shadow-md">
                    <i class="fa-solid fa-film"></i>
                </div>
                <div>
                    <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                        LOC Moving Image Semantic Search
                        <span class="text-xs px-2.5 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30">Pinecone Serverless</span>
                    </h1>
                    <p class="text-xs text-slate-400">Library of Congress • WWII Color Archival Footage (Item 2020600759)</p>
                </div>
            </div>
            
            <!-- Namespace & Stats Controls -->
            <div class="flex items-center space-x-3 text-xs">
                <!-- Namespace Switcher -->
                <div class="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800">
                    <button id="tabWw2" onclick="switchNamespace('loc-ww2-color')" 
                        class="px-3 py-1.5 rounded-lg font-semibold bg-amber-500 text-slate-950 shadow transition flex items-center gap-1.5">
                        <i class="fa-solid fa-monument text-xs"></i>
                        <span>WWII Color (Active)</span>
                    </button>
                    <button id="tabRm" onclick="switchNamespace('default')" 
                        class="px-3 py-1.5 rounded-lg font-semibold text-slate-400 hover:text-white transition flex items-center gap-1.5">
                        <i class="fa-solid fa-atom text-xs"></i>
                        <span>Rick & Morty (Legacy)</span>
                    </button>
                </div>

                <div class="bg-slate-800/80 px-3 py-2 rounded-lg border border-slate-700 flex items-center gap-2">
                    <i class="fa-solid fa-database text-amber-400"></i>
                    <span id="statVectors" class="mono font-semibold">1,050 Vectors</span>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-6 py-8 flex-1 w-full">
        <!-- Hero Section -->
        <div class="text-center max-w-3xl mx-auto mb-10">
            <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-semibold mb-4">
                <i class="fa-solid fa-circle-check"></i>
                <span id="heroBadge">George Stevens SPECOU in Berlin & North Africa (1943–1945) • 39m 54s</span>
            </div>
            <h2 class="text-4xl font-extrabold tracking-tight mb-3 text-white" id="heroTitle">
                Search 40 Minutes of <span class="text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-orange-400">WWII Color Footage</span>
            </h2>
            <p class="text-slate-400 text-sm mb-6" id="heroDesc">
                Natural-language visual search directly into authentic 16mm/35mm Kodachrome color film digitized by the Library of Congress.
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
            </div>
        </div>

        <!-- Query Performance Timing Bar -->
        <div id="timingBar" class="hidden max-w-3xl mx-auto mb-8 bg-slate-900/90 border border-slate-800 rounded-xl px-4 py-2.5 text-xs flex items-center justify-between text-slate-400 shadow-md">
            <div class="flex items-center gap-4">
                <span><i class="fa-solid fa-bolt text-amber-400 mr-1"></i> Total: <b id="timeTotal" class="text-white mono">-</b></span>
                <span>• Embed: <b id="timeEmbed" class="text-slate-300 mono">-</b></span>
                <span>• Pinecone: <b id="timeQuery" class="text-slate-300 mono">-</b></span>
                <span>• Namespace: <b id="activeNsLabel" class="text-amber-400 mono">loc-ww2-color</b></span>
            </div>
            <div class="text-amber-400 font-bold" id="matchesCount">0 scenes found</div>
        </div>

        <!-- Loading Spinner -->
        <div id="loading" class="hidden text-center py-16">
            <div class="inline-block animate-spin text-4xl text-amber-400 mb-3"><i class="fa-solid fa-circle-notch"></i></div>
            <p class="text-slate-400 text-sm">Embedding query and searching Pinecone index...</p>
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
                <!-- HTML5 Native Player for WW2 -->
                <video id="nativePlayer" controls class="w-full h-full object-contain" preload="metadata">
                    <source id="videoSource" src="/video/stream" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <!-- YouTube Iframe for Rick & Morty -->
                <iframe id="modalIframe" class="w-full h-full hidden" src="" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
            </div>

            <div id="frameContainer" class="hidden mb-4">
                <img id="modalImg" src="" alt="Scene Frame" class="w-full rounded-xl border border-slate-800 object-cover max-h-96">
            </div>
            
            <!-- Scene Caption & AI Description Box -->
            <div class="bg-slate-950/80 border border-slate-800 rounded-xl p-4 mb-4">
                <div class="text-xs text-amber-400 font-bold mb-1 flex items-center gap-2">
                    <i class="fa-solid fa-quote-left"></i>
                    <span>Indexed Scene Caption</span>
                </div>
                <p id="modalCaption" class="text-xs text-slate-200 leading-relaxed mb-3 italic">Loading caption...</p>
                
                <div class="pt-2 border-t border-slate-800/80">
                    <div class="flex items-center gap-2 text-xs font-bold text-slate-400 mb-1">
                        <i class="fa-solid fa-brain text-amber-400"></i>
                        <span>Archival Deep Dive</span>
                    </div>
                    <p id="aiDescription" class="text-xs text-slate-300 leading-relaxed">Analyzing frame context...</p>
                </div>
            </div>

            <div class="flex items-center justify-between">
                <a id="modalSourceLink" href="https://www.loc.gov/item/2020600759/" target="_blank" 
                    class="px-5 py-2.5 bg-slate-800 hover:bg-slate-700 text-amber-400 font-bold rounded-xl text-sm flex items-center gap-2 transition border border-slate-700">
                    <i class="fa-solid fa-building-columns"></i>
                    <span id="modalSourceText">View on Library of Congress (LOC.gov)</span>
                </a>
                <button onclick="closeModal()" class="px-5 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium rounded-xl text-sm transition">
                    Close
                </button>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer class="border-t border-slate-800 py-6 text-center text-xs text-slate-500">
        Library of Congress Catalog Item 2020600759 • Hosted on Fireworks AI + Pinecone Serverless
    </footer>

    <script>
        let currentNamespace = 'loc-ww2-color';

        async function loadStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                const total = data.total_vectors || 1050;
                document.getElementById('statVectors').innerText = `${total} Vectors Indexed`;
            } catch(e) {
                console.error("Failed to load stats", e);
            }
        }
        loadStats();

        function switchNamespace(ns) {
            currentNamespace = ns;
            const tabWw2 = document.getElementById('tabWw2');
            const tabRm = document.getElementById('tabRm');
            const chips = document.getElementById('chipsContainer');
            const heroBadge = document.getElementById('heroBadge');
            const heroTitle = document.getElementById('heroTitle');
            const heroDesc = document.getElementById('heroDesc');
            const searchInput = document.getElementById('searchInput');

            if (ns === 'loc-ww2-color') {
                tabWw2.className = "px-3 py-1.5 rounded-lg font-semibold bg-amber-500 text-slate-950 shadow transition flex items-center gap-1.5";
                tabRm.className = "px-3 py-1.5 rounded-lg font-semibold text-slate-400 hover:text-white transition flex items-center gap-1.5";
                heroBadge.innerText = "George Stevens SPECOU in Berlin & North Africa (1943–1945) • 39m 54s";
                heroTitle.innerHTML = 'Search 40 Minutes of <span class="text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-orange-400">WWII Color Footage</span>';
                heroDesc.innerText = "Natural-language visual search directly into authentic 16mm/35mm Kodachrome color film digitized by the Library of Congress.";
                searchInput.value = "Great Sphinx and pyramids in Egypt";
                
                chips.innerHTML = `
                    <span class="text-slate-500">Curated queries:</span>
                    <button onclick="setQuery('Great Sphinx and Pyramids of Giza in Egypt')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-amber-500 text-slate-300 hover:text-amber-400 transition">🏛️ Great Sphinx & Pyramids</button>
                    <button onclick="setQuery('Bombed Reichstag ruins and destruction in Berlin')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-amber-500 text-slate-300 hover:text-amber-400 transition">💥 Reichstag Ruins</button>
                    <button onclick="setQuery('Tanks and military armor driving through desert sand')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-amber-500 text-slate-300 hover:text-amber-400 transition">🛡️ Desert Tanks</button>
                    <button onclick="setQuery('American pilots and aircraft on airfield')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-amber-500 text-slate-300 hover:text-amber-400 transition">✈️ Airfield Pilots</button>
                    <button onclick="setQuery('Civilians and refugees carrying luggage and carts')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-amber-500 text-slate-300 hover:text-amber-400 transition">🚶 Berlin Refugees</button>
                `;
            } else {
                tabRm.className = "px-3 py-1.5 rounded-lg font-semibold bg-emerald-500 text-slate-950 shadow transition flex items-center gap-1.5";
                tabWw2.className = "px-3 py-1.5 rounded-lg font-semibold text-slate-400 hover:text-white transition flex items-center gap-1.5";
                heroBadge.innerText = "Rick and Morty Season 9 Battle Scenes • 30m 4s (Legacy Namespace)";
                heroTitle.innerHTML = 'Search Rick and Morty <span class="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">Battle Scenes</span>';
                heroDesc.innerText = "Legacy namespace search across 452 cartoon battle scene frames.";
                searchInput.value = "space dogfight battle with laser guns";

                chips.innerHTML = `
                    <span class="text-slate-500">Legacy queries:</span>
                    <button onclick="setQuery('space dogfight battle with laser guns')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-emerald-500 text-slate-300 hover:text-emerald-400 transition">🚀 Space Battle</button>
                    <button onclick="setQuery('green portal gun opening')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-emerald-500 text-slate-300 hover:text-emerald-400 transition">🌀 Portal Gun</button>
                    <button onclick="setQuery('giant explosion in outer space')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-emerald-500 text-slate-300 hover:text-emerald-400 transition">💥 Massive Explosion</button>
                `;
            }
            executeSearch();
        }

        function setQuery(text) {
            document.getElementById('searchInput').value = text;
            executeSearch();
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
                const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&namespace=${currentNamespace}&top_k=9`);
                const data = await res.json();
                loading.classList.add('hidden');

                if (data.results && data.results.length > 0) {
                    timingBar.classList.remove('hidden');
                    document.getElementById('timeTotal').innerText = `${data.timing.total_time_ms} ms`;
                    document.getElementById('timeEmbed').innerText = `${data.timing.embed_time_ms} ms`;
                    document.getElementById('timeQuery').innerText = `${data.timing.query_time_ms} ms`;
                    document.getElementById('activeNsLabel').innerText = data.namespace;
                    document.getElementById('matchesCount').innerText = `${data.total_matches} scenes found`;

                    data.results.forEach((clip, idx) => {
                        const card = document.createElement('div');
                        card.className = "bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden card-hover transition duration-200 flex flex-col justify-between";
                        
                        const frameSrc = clip.frame_url || 'https://via.placeholder.com/336x336?text=No+Frame';
                        const isWw2 = (data.namespace === 'loc-ww2-color');

                        card.innerHTML = `
                            <div class="relative cursor-pointer group" onclick="openModal('${frameSrc}', '${clip.timestamp}', '${clip.score}', '${clip.source_link}', '${clip.frame_filename}', '${clip.start_time_s}', '${encodeURIComponent(clip.caption)}')">
                                <img src="${frameSrc}" alt="Clip ${clip.clip_index}" class="w-full h-52 object-cover group-hover:scale-105 transition duration-300">
                                <div class="absolute inset-0 bg-gradient-to-t from-slate-950 via-slate-950/20 to-transparent"></div>
                                <div class="absolute top-3 left-3 bg-black/75 backdrop-blur px-2.5 py-1 rounded-md text-xs font-bold text-white border border-white/10 mono">
                                    <i class="fa-solid fa-clock ${isWw2 ? 'text-amber-400' : 'text-emerald-400'} mr-1"></i> ${clip.timestamp}
                                </div>
                                <div class="absolute top-3 right-3 score-pill text-white px-2.5 py-0.5 rounded-full text-xs font-bold shadow-md mono">
                                    #${idx + 1} • ${clip.score.toFixed(3)}
                                </div>
                                <div class="absolute bottom-3 left-3 right-3 flex items-center justify-between text-xs text-slate-300">
                                    <span class="mono">Frame #${clip.clip_index}</span>
                                    <span class="${isWw2 ? 'text-amber-400' : 'text-emerald-400'} group-hover:underline flex items-center gap-1 font-semibold">
                                        <span>Play & Inspect</span>
                                        <i class="fa-solid fa-play text-[10px]"></i>
                                    </span>
                                </div>
                            </div>
                            <div class="p-4 bg-slate-950/60 border-t border-slate-800/80 flex flex-col justify-between flex-1">
                                <p class="text-xs text-slate-300 line-clamp-2 mb-3 leading-relaxed">${clip.caption}</p>
                                <div class="flex items-center justify-between pt-2 border-t border-slate-900 text-xs">
                                    <span class="text-slate-500 mono">${clip.start_time_s}s - ${clip.end_time_s}s</span>
                                    <button onclick="openModal('${frameSrc}', '${clip.timestamp}', '${clip.score}', '${clip.source_link}', '${clip.frame_filename}', '${clip.start_time_s}', '${encodeURIComponent(clip.caption)}')"
                                        class="${isWw2 ? 'text-amber-400 hover:text-amber-300' : 'text-emerald-400 hover:text-emerald-300'} font-semibold flex items-center gap-1 transition">
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

        async function openModal(imgSrc, timestamp, score, sourceLink, frameFile, startTimeSec, rawCaption) {
            document.getElementById('modalImg').src = imgSrc;
            document.getElementById('modalTimestamp').innerText = timestamp;
            document.getElementById('modalScore').innerText = `Cosine: ${score}`;
            document.getElementById('modalSourceLink').href = sourceLink;
            document.getElementById('modalCaption').innerText = rawCaption ? decodeURIComponent(rawCaption) : "";

            const nativePlayer = document.getElementById('nativePlayer');
            const modalIframe = document.getElementById('modalIframe');
            const sourceText = document.getElementById('modalSourceText');

            const startSec = Math.max(0, Math.floor(parseFloat(startTimeSec) || 0));

            if (currentNamespace === 'loc-ww2-color') {
                sourceText.innerText = "View Catalog on Library of Congress (LOC.gov)";
                modalIframe.classList.add('hidden');
                modalIframe.src = "";
                nativePlayer.classList.remove('hidden');

                // Seek native player
                nativePlayer.currentTime = startSec;
                nativePlayer.play().catch(e => console.log("Autoplay waiting for user gesture"));
            } else {
                sourceText.innerText = "Watch on YouTube";
                nativePlayer.pause();
                nativePlayer.classList.add('hidden');
                modalIframe.classList.remove('hidden');
                modalIframe.src = `https://www.youtube.com/embed/9Rul9N1LREQ?start=${startSec}&autoplay=1`;
            }

            toggleModalMedia('video');
            document.getElementById('modal').classList.remove('hidden');

            document.getElementById('aiDescription').innerText = "Querying GPT-4o-mini archival historian analysis...";
            const query = document.getElementById('searchInput').value;
            try {
                const res = await fetch(`/api/ai-describe?frame_file=${encodeURIComponent(frameFile)}&namespace=${currentNamespace}&query=${encodeURIComponent(query)}`);
                const data = await res.json();
                document.getElementById('aiDescription').innerText = data.description || "Scene analysis complete.";
            } catch (e) {
                document.getElementById('aiDescription').innerText = "Could not fetch AI analysis.";
            }
        }

        function closeModal() {
            document.getElementById('modal').classList.add('hidden');
            const nativePlayer = document.getElementById('nativePlayer');
            nativePlayer.pause();
            document.getElementById('modalIframe').src = "";
        }

        // Run default search on load
        executeSearch();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
