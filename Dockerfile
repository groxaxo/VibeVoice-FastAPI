FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ONLY_BINARY=:all:

# Triton needs Python headers and a C++ compiler at runtime when torch.compile
# builds kernels. ffmpeg/libsndfile provide audio format support; curl powers the
# container health check.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    gcc \
    g++ \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

WORKDIR /app
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# Copy dependency metadata first so application-only edits reuse the package layers.
COPY pyproject.toml README.md ./
COPY requirements-api.txt ./

# torchvision is intentionally omitted: no server or model code imports it, and it
# adds a large unused binary dependency to the image.
RUN pip install torch==2.8.* torchaudio --index-url https://download.pytorch.org/whl/cu128

# Optional quantized checkpoint/runtime backends supported by the API.
RUN pip install torchao==0.13.0 bitsandbytes autoawq

# Pre-built wheel only: the runtime image does not include nvcc for a source build.
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

COPY vibevoice/ ./vibevoice/
COPY demo/ ./demo/
RUN pip install --only-binary=:all: -e .
RUN pip install --only-binary=:all: -r requirements-api.txt

COPY api/ ./api/
COPY start.sh ./
RUN mkdir -p /app/voices /app/models

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["sh", "-c", "/app/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT:-8001} --workers 1"]
