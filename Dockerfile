FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user required by Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Install CPU PyTorch first to keep image lightweight and avoid CUDA bloat
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache EmbeddingGemma 2 model in the image layer for instant cold start
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('google/embeddinggemma-2', device='cpu')"

# Copy application files, frames, and metadata
COPY --chown=user:user . .

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Launch FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
