"""
Library of Congress (LOC) Video Explorer and Downloader.
Searches the 16,000+ public-domain digitized film and video archives on loc.gov
and extracts direct MP4 download links for real-world visual embedding benchmarks.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx


def search_loc_videos(
    query: str = "",
    collection: str = "national-screening-room",
    count: int = 10,
) -> List[Dict[str, Any]]:
    """Searches Library of Congress moving image catalog via the public JSON API."""
    if collection:
        base_url = f"https://www.loc.gov/collections/{collection}/"
    else:
        base_url = "https://www.loc.gov/film-and-video/"

    params = {
        "fo": "json",
        "c": count,
    }
    if query:
        params["q"] = query

    headers = {"User-Agent": "LOC-Video-Explorer/1.0 (Educational/Research)"}
    print(f"[*] Querying Library of Congress API: {base_url} (q='{query}') ...")

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        resp = client.get(base_url, params=params, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()

        results = data.get("results", [])
        total_found = data.get("pagination", {}).get("total", len(results))
        print(f"[+] Total items found in collection: {total_found:,}")

        video_items = []
        for item in results:
            title = item.get("title", "Untitled")
            date = item.get("date", "Unknown date")
            item_id = item.get("id", "")
            summary = item.get("description", [""])[0] if item.get("description") else ""

            # Fetch detailed item metadata to resolve direct MP4 download links
            mp4_url = None
            if item_id:
                try:
                    detail_url = f"{item_id.rstrip('/')}/?fo=json"
                    d_resp = client.get(detail_url, headers=headers)
                    if d_resp.status_code == 200:
                        d_data = d_resp.json()
                        resources = d_data.get("resources", [])
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
                except Exception:
                    pass

            video_items.append({
                "title": title,
                "date": date,
                "item_url": item_id,
                "summary": summary[:120] + "..." if len(summary) > 120 else summary,
                "mp4_url": mp4_url,
            })

    return video_items


def download_loc_video(mp4_url: str, output_path: str):
    """Downloads an MP4 video directly from Library of Congress CDN with progress."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Downloading LOC video: {mp4_url}")
    print(f"    Target Destination:   {output_path}")

    headers = {"User-Agent": "LOC-Video-Explorer/1.0"}
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        with client.stream("GET", mp4_url, headers=headers) as response:
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code} while downloading")
            total_bytes = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(out, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=1024 * 512):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0:
                        pct = (downloaded / total_bytes) * 100
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_bytes / (1024 * 1024)
                        sys.stdout.write(f"\r    Progress: {mb:.1f}MB / {total_mb:.1f}MB ({pct:.1f}%)")
                        sys.stdout.flush()

    print(f"\n[+] Successfully downloaded to {output_path} ({out.stat().st_size / (1024*1024):.1f} MB)\n")


def main():
    parser = argparse.ArgumentParser(description="Explore and download Library of Congress videos")
    parser.add_argument("--query", type=str, default="", help="Search query keyword (e.g. flight, train, parade)")
    parser.add_argument("--collection", type=str, default="national-screening-room", help="Collection slug or empty")
    parser.add_argument("--count", type=int, default=5, help="Number of items to preview")
    parser.add_argument("--download-index", type=int, default=None, help="Download item at 1-based index")
    parser.add_argument("--output", type=str, default="data/videos/loc_video.mp4")

    args = parser.parse_args()

    videos = search_loc_videos(query=args.query, collection=args.collection, count=args.count)

    print("\n" + "=" * 95)
    print("LIBRARY OF CONGRESS (LOC) DIGITIZED MOVING IMAGE CATALOG")
    print("=" * 95)
    for idx, v in enumerate(videos, 1):
        has_mp4 = "✓ MP4 Available" if v["mp4_url"] else "No direct MP4"
        print(f"[{idx}] {v['title']} ({v['date']}) - [{has_mp4}]")
        print(f"    Item Page: {v['item_url']}")
        if v["mp4_url"]:
            print(f"    Direct MP4: {v['mp4_url']}")
        if v["summary"]:
            print(f"    Summary:   {v['summary']}")
        print()

    if args.download_index is not None:
        selected = videos[args.download_index - 1]
        if selected.get("mp4_url"):
            download_loc_video(selected["mp4_url"], args.output)
        else:
            print(f"[!] Item {args.download_index} does not have a direct MP4 link.")


if __name__ == "__main__":
    main()
