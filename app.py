"""
FastAPI Server & Video Search UI for Rick and Morty Video Retrieval.
Connects Pinecone Serverless Index (rick-morty-gemma2-video) with OpenAI 768-dim Embeddings
and visualizes matching 4-second video frames from local disk.
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
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
FRAMES_DIR = Path(__file__).parent / "data" / "temp_4s_frames"

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY or OPENAI_API_KEY in environment or .env file.")

# Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Video Semantic Search - Rick & Morty Battle Scenes")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local frames directory
if FRAMES_DIR.exists():
    app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")


@app.get("/api/stats")
async def get_stats():
    """Returns Pinecone index statistics and frame storage status."""
    try:
        stats = index.describe_index_stats()
        frames_count = len(list(FRAMES_DIR.glob("*.jpg"))) if FRAMES_DIR.exists() else 0
        return {
            "index_name": INDEX_NAME,
            "dimension": stats.dimension,
            "total_vectors": stats.total_vector_count,
            "metric": stats.metric,
            "local_frames_available": frames_count,
            "video_title": "Rick and Morty | Season 9 Battle Scenes | adult swim",
            "video_duration": "30m 4s (1,804.2 seconds)",
            "granularity": "1 frame every 4 seconds",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search")
async def search_video(
    q: str = Query(..., description="Text query to search within video footage"),
    top_k: int = Query(8, ge=1, le=24),
):
    """
    1. Embeds query into 768-dimensional space using OpenAI text-embedding-3-small (dimensions=768)
    2. Queries Pinecone index
    3. Resolves local frame images and timestamps
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
            namespace="default",
            include_metadata=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query error: {e}")

    query_time_ms = round((time.perf_counter() - t1) * 1000.0, 1)
    total_time_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    # 3. Format matches
    matches = []
    for m in pinecone_res.matches:
        meta = m.metadata or {}
        clip_index = int(meta.get("clip_index", 0))
        frame_filename = meta.get("local_frame_file") or f"clip_{clip_index:04d}_mid.jpg"
        frame_path = FRAMES_DIR / frame_filename

        matches.append({
            "id": m.id,
            "score": round(float(m.score), 4),
            "clip_index": clip_index,
            "start_time_s": meta.get("start_time_s"),
            "end_time_s": meta.get("end_time_s"),
            "timestamp": meta.get("timestamp", "00:00:00"),
            "youtube_url": meta.get("youtube_url", "https://youtu.be/9Rul9N1LREQ"),
            "frame_url": f"/frames/{frame_filename}" if frame_path.exists() else None,
            "frame_filename": frame_filename,
        })

    return {
        "query": q,
        "timing": {
            "embed_time_ms": embed_time_ms,
            "query_time_ms": query_time_ms,
            "total_time_ms": total_time_ms,
        },
        "total_matches": len(matches),
        "results": matches,
    }


@app.get("/api/ai-describe")
async def ai_describe(frame_file: str, query: str = ""):
    """Uses OpenAI GPT-4o-mini to inspect the local frame and describe what is happening."""
    frame_path = FRAMES_DIR / frame_file
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found on local disk.")

    with open(frame_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")

    prompt = f"Describe what is happening in this scene from Rick and Morty. "
    if query:
        prompt += f"Explain how this visual frame relates to the search query: '{query}'."

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        },
                    ],
                }
            ],
            max_tokens=200,
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
    <title>Rick & Morty Video Semantic Search | Pinecone + Gemma 2</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
        body { font-family: 'Space Grotesk', sans-serif; background-color: #0b0f19; color: #f1f5f9; }
        .portal-glow { box-shadow: 0 0 35px rgba(34, 197, 94, 0.25); }
        .portal-border { border: 1px solid rgba(34, 197, 94, 0.4); }
        .card-hover:hover { transform: translateY(-4px); box-shadow: 0 12px 30px rgba(16, 185, 129, 0.15); }
        .score-pill { background: linear-gradient(135deg, #059669 0%, #10b981 100%); }
    </style>
</head>
<body class="min-h-screen flex flex-col justify-between">
    <!-- Navbar -->
    <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/50 flex items-center justify-center text-emerald-400 text-xl font-bold">
                    <i class="fa-solid fa-atom"></i>
                </div>
                <div>
                    <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                        Rick & Morty Video Search
                        <span class="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">Pinecone Serverless</span>
                    </h1>
                    <p class="text-xs text-slate-400">4-Second Clips • 768-Dim Dense Vectors • Local Frame Storage</p>
                </div>
            </div>
            <div class="flex items-center space-x-4 text-xs" id="statsBar">
                <div class="bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700 flex items-center gap-2">
                    <i class="fa-solid fa-database text-emerald-400"></i>
                    <span id="statVectors">Loading...</span>
                </div>
                <div class="bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700 flex items-center gap-2">
                    <i class="fa-solid fa-images text-blue-400"></i>
                    <span id="statFrames">Loading...</span>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-6 py-8 flex-1 w-full">
        <!-- Search Hero Section -->
        <div class="text-center max-w-3xl mx-auto mb-10">
            <h2 class="text-4xl font-extrabold tracking-tight mb-3 text-white">
                Find Any Moment in the <span class="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">Battle Scenes</span>
            </h2>
            <p class="text-slate-400 text-sm mb-6">
                Type natural language descriptions to jump straight to matching 4-second scenes with high visual similarity.
            </p>

            <!-- Search Form -->
            <form id="searchForm" class="relative max-w-2xl mx-auto mb-4">
                <div class="relative flex items-center">
                    <i class="fa-solid fa-magnifying-glass absolute left-4 text-emerald-400 text-lg"></i>
                    <input type="text" id="searchInput" 
                        class="w-full pl-12 pr-32 py-4 bg-slate-900 border border-slate-700 rounded-2xl text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 portal-glow transition text-base"
                        placeholder="e.g. Rick firing laser gun at alien robots..." 
                        value="Rick firing laser cannon at alien fleet" required>
                    <button type="submit" id="searchBtn"
                        class="absolute right-2 px-6 py-2.5 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 font-bold rounded-xl transition flex items-center gap-2 text-sm shadow-lg shadow-emerald-500/20">
                        <span>Search</span>
                        <i class="fa-solid fa-bolt"></i>
                    </button>
                </div>
            </form>

            <!-- Suggestion Pills -->
            <div class="flex flex-wrap items-center justify-center gap-2 text-xs">
                <span class="text-slate-500">Try searching:</span>
                <button onclick="setQuery('Rick laser cannon battle against space ships')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-emerald-500 text-slate-300 hover:text-emerald-400 transition">🚀 Space Dogfight</button>
                <button onclick="setQuery('Green portal opening escape scene')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-emerald-500 text-slate-300 hover:text-emerald-400 transition">🌀 Green Portal Gun</button>
                <button onclick="setQuery('Morty screaming in fear while running')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-emerald-500 text-slate-300 hover:text-emerald-400 transition">😱 Morty Panicking</button>
                <button onclick="setQuery('Giant explosion destroying alien base')" class="px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 hover:border-emerald-500 text-slate-300 hover:text-emerald-400 transition">💥 Massive Explosion</button>
            </div>
        </div>

        <!-- Query Performance Timing Bar -->
        <div id="timingBar" class="hidden max-w-2xl mx-auto mb-8 bg-slate-900/80 border border-slate-800 rounded-xl px-4 py-2.5 text-xs flex items-center justify-between text-slate-400">
            <div class="flex items-center gap-4">
                <span><i class="fa-solid fa-stopwatch text-emerald-400 mr-1"></i> Total: <b id="timeTotal" class="text-white">-</b></span>
                <span>• Embed: <b id="timeEmbed" class="text-slate-300">-</b></span>
                <span>• Pinecone: <b id="timeQuery" class="text-slate-300">-</b></span>
            </div>
            <div class="text-emerald-400 font-semibold" id="matchesCount">0 matches</div>
        </div>

        <!-- Loading Spinner -->
        <div id="loading" class="hidden text-center py-16">
            <div class="inline-block animate-spin text-4xl text-emerald-400 mb-3"><i class="fa-solid fa-circle-notch"></i></div>
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

    <!-- Modal for Full Image & AI Analysis -->
    <div id="modal" class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 shadow-2xl relative">
            <button onclick="closeModal()" class="absolute top-4 right-4 text-slate-400 hover:text-white text-xl">
                <i class="fa-solid fa-xmark"></i>
            </button>
            <div class="mb-4">
                <span id="modalTimestamp" class="text-xs px-2.5 py-1 rounded-md bg-emerald-500/20 text-emerald-400 font-bold border border-emerald-500/30">00:00:00</span>
                <span id="modalScore" class="text-xs text-slate-400 ml-2">Score: 0.000</span>
            </div>
            <img id="modalImg" src="" alt="Scene Frame" class="w-full rounded-xl border border-slate-800 mb-4 object-cover max-h-96">
            
            <!-- AI Explanation Box -->
            <div id="aiBox" class="bg-slate-950/70 border border-slate-800 rounded-xl p-4 mb-4">
                <div class="flex items-center gap-2 text-xs font-bold text-emerald-400 mb-2">
                    <i class="fa-solid fa-brain"></i>
                    <span>GPT-4o-Mini Scene Breakdown</span>
                </div>
                <p id="aiDescription" class="text-xs text-slate-300 leading-relaxed">Analyzing frame context...</p>
            </div>

            <div class="flex items-center justify-between">
                <a id="modalYoutubeLink" href="#" target="_blank" 
                    class="px-5 py-2.5 bg-red-600 hover:bg-red-500 text-white font-bold rounded-xl text-sm flex items-center gap-2 transition">
                    <i class="fa-brands fa-youtube"></i>
                    <span>Watch at Exact Second</span>
                </a>
                <button onclick="closeModal()" class="px-5 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium rounded-xl text-sm transition">
                    Close
                </button>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer class="border-t border-slate-800 py-6 text-center text-xs text-slate-500">
        Rick and Morty Season 9 Battle Scenes • Powered by Fireworks Gemma 2 + Pinecone Serverless + OpenAI Embeddings
    </footer>

    <script>
        // Load initial index stats
        async function loadStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('statVectors').innerText = `${data.total_vectors} Vectors Indexed`;
                document.getElementById('statFrames').innerText = `${data.local_frames_available} Local Frames`;
            } catch(e) {
                console.error("Failed to load stats", e);
            }
        }
        loadStats();

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
                const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&top_k=9`);
                const data = await res.json();
                loading.classList.add('hidden');

                if (data.results && data.results.length > 0) {
                    timingBar.classList.remove('hidden');
                    document.getElementById('timeTotal').innerText = `${data.timing.total_time_ms} ms`;
                    document.getElementById('timeEmbed').innerText = `${data.timing.embed_time_ms} ms`;
                    document.getElementById('timeQuery').innerText = `${data.timing.query_time_ms} ms`;
                    document.getElementById('matchesCount').innerText = `${data.total_matches} scenes found`;

                    data.results.forEach((clip, idx) => {
                        const card = document.createElement('div');
                        card.className = "bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden card-hover transition duration-200 flex flex-col justify-between";
                        
                        const frameSrc = clip.frame_url || 'https://via.placeholder.com/336x336?text=No+Frame';
                        const scorePct = Math.round(clip.score * 100);

                        card.innerHTML = `
                            <div class="relative cursor-pointer group" onclick="openModal('${frameSrc}', '${clip.timestamp}', '${clip.score}', '${clip.youtube_url}', '${clip.frame_filename}')">
                                <img src="${frameSrc}" alt="Clip ${clip.clip_index}" class="w-full h-48 object-cover group-hover:scale-105 transition duration-300">
                                <div class="absolute inset-0 bg-gradient-to-t from-slate-950 via-transparent to-transparent opacity-80"></div>
                                <div class="absolute top-3 left-3 bg-black/70 backdrop-blur px-2.5 py-1 rounded-md text-xs font-bold text-white border border-white/10">
                                    <i class="fa-solid fa-clock text-emerald-400 mr-1"></i> ${clip.timestamp}
                                </div>
                                <div class="absolute top-3 right-3 score-pill text-white px-2 py-0.5 rounded-full text-xs font-bold shadow-md">
                                    #${idx + 1} • ${clip.score.toFixed(3)}
                                </div>
                                <div class="absolute bottom-3 left-3 right-3 flex items-center justify-between text-xs text-slate-300">
                                    <span>Clip #${clip.clip_index}</span>
                                    <span class="text-emerald-400 group-hover:underline flex items-center gap-1">
                                        <span>Inspect & AI Explain</span>
                                        <i class="fa-solid fa-arrow-right"></i>
                                    </span>
                                </div>
                            </div>
                            <div class="p-4 border-t border-slate-800/80 bg-slate-950/40 flex items-center justify-between">
                                <span class="text-xs text-slate-400">${clip.start_time_s}s - ${clip.end_time_s}s</span>
                                <a href="${clip.youtube_url}" target="_blank" 
                                    class="text-xs font-bold text-red-400 hover:text-red-300 flex items-center gap-1.5 transition">
                                    <i class="fa-brands fa-youtube text-red-500"></i>
                                    <span>Watch on YouTube</span>
                                </a>
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

        // Modal Logic
        async function openModal(imgSrc, timestamp, score, ytUrl, frameFile) {
            document.getElementById('modalImg').src = imgSrc;
            document.getElementById('modalTimestamp').innerText = timestamp;
            document.getElementById('modalScore').innerText = `Cosine Similarity: ${score}`;
            document.getElementById('modalYoutubeLink').href = ytUrl;
            document.getElementById('aiDescription').innerText = "Querying GPT-4o-mini to analyze scene details...";
            document.getElementById('modal').classList.remove('hidden');

            const query = document.getElementById('searchInput').value;
            try {
                const res = await fetch(`/api/ai-describe?frame_file=${encodeURIComponent(frameFile)}&query=${encodeURIComponent(query)}`);
                const data = await res.json();
                document.getElementById('aiDescription').innerText = data.description || "Scene analysis complete.";
            } catch (e) {
                document.getElementById('aiDescription').innerText = "Could not fetch AI description.";
            }
        }

        function closeModal() {
            document.getElementById('modal').classList.add('hidden');
        }

        // Run default search on load
        executeSearch();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
