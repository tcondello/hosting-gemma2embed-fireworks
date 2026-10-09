---
title: LOC WWII Color Video Search
emoji: 🎥
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
tags:
  - video-search
  - embeddinggemma-2
  - gemma-2
  - multimodal
  - library-of-congress
license: apache-2.0
short_description: WWII Color Video Search with EmbeddingGemma 2
---

# Library of Congress WWII Color Video Search (EmbeddingGemma 2)

Natural-language visual search engine across authentic 16mm Kodachrome color film digitized by the Library of Congress ([LCCN 2020600759](https://www.loc.gov/item/2020600759/)).

Powered by **Google DeepMind EmbeddingGemma 2** (744M multimodal parameter architecture) and 100% in-browser, in-memory cosine similarity search (zero external vector database, zero API keys, $0.00 infrastructure cost).

## Highlights
- **Authentic Archival Footage**: 39 minutes, 54 seconds of George Stevens & SPECOU Kodachrome color film (1943–1945). Selected for the National Film Registry.
- **Sub-Millisecond Vector Search**: 598 unit-normalized 768-dimensional vectors queried in < 1 ms via JavaScript `Float32Array` dot products.
- **Direct LOC 1080p Streaming**: Native HTML5 player with instant timestamp seeking directly from Library of Congress CDN.
- **Dataset**: Full embeddings Parquet and frame archives published at [`astr010/loc-ww2-color-film-gemma2`](https://huggingface.co/datasets/astr010/loc-ww2-color-film-gemma2).
