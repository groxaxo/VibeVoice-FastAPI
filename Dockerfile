FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_BUILD_ISOLATION=0 \
    PIP_ONLY_BINARY=:all:

# Install system dependencies (NO build-essential - we don't compile!)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml README.md ./
COPY requirements-api.txt ./

# Install PyTorch with CUDA support (CUDA 12.1 compatible)
RUN pip install --only-binary=:all: torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn from pre-built wheel directly from GitHub releases
# Python 3.10 (cp310), CUDA 12.1 (cu121), PyTorch 2.5.1
# Downloading directly ensures NO compilation ever happens
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu121torch2.5.1-cp310-cp310-linux_x86_64.whl || \
    echo "WARNING: flash-attn wheel install failed, continuing without it"

# Install VibeVoice package (with --only-binary to prevent any compilation)
COPY vibevoice/ ./vibevoice/
COPY demo/ ./demo/
RUN pip install --only-binary=:all: -e . || pip install -e .

# Install API dependencies
RUN pip install -r requirements-api.txt

# Copy API code
COPY api/ ./api/
COPY start.sh ./

# Create directories for voices and models
RUN mkdir -p /app/voices /app/models

# Expose API port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the API
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT:-8001}"]

