# VibeVoice API - Docker Deployment Guide

## Quick Start

### 1. Build and Run with Docker Compose (Recommended)

```bash
# Copy environment file
cp docker-env.example .env

# Edit .env to configure paths
nano .env

# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### 2. Build and Run with Docker

```bash
# Build image
docker build -t vibevoice-api:latest .

# Run container
docker run -d \
  --name vibevoice-api \
  --gpus all \
  -p 8000:8000 \
  -v /path/to/voices:/app/voices:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface:rw \
  -e VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B \
  -e VIBEVOICE_DEVICE=cuda \
  -e VOICES_DIR=/app/voices \
  vibevoice-api:latest

# View logs
docker logs -f vibevoice-api

# Stop
docker stop vibevoice-api
```

## Configuration

### Environment Variables

Set these in your `.env` file for docker-compose or pass via `-e` flag:

```bash
# Model
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B
VIBEVOICE_DEVICE=cuda
VIBEVOICE_INFERENCE_STEPS=10

# Voices (inside container)
VOICES_DIR=/app/voices

# API
API_PORT=8000
API_CORS_ORIGINS=*

# Generation
DEFAULT_CFG_SCALE=1.3
DEFAULT_RESPONSE_FORMAT=mp3
```

### Volume Mounts

Mount these directories from your host:

1. **Voices** (required):
   ```bash
   -v /path/to/your/voices:/app/voices:ro
   ```

2. **HuggingFace Cache** (recommended, for faster model loading):
   ```bash
   -v ~/.cache/huggingface:/root/.cache/huggingface:rw
   ```

3. **Models** (optional, for local models):
   ```bash
   -v /path/to/models:/app/models:rw
   ```

## GPU Support

### NVIDIA GPU (Required)

Install NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Test GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Using Local Models

If you have models downloaded locally:

```bash
# Option 1: Mount HuggingFace cache
docker run -v ~/.cache/huggingface:/root/.cache/huggingface:rw ...

# Option 2: Mount specific model directory
docker run \
  -v /path/to/VibeVoice-1.5B:/app/models/VibeVoice-1.5B:ro \
  -e VIBEVOICE_MODEL_PATH=/app/models/VibeVoice-1.5B \
  ...
```

## Health Checks

The container includes health checks:

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' vibevoice-api

# Manual health check
curl http://localhost:8000/health
```

## Troubleshooting

### Container won't start

Check logs:
```bash
docker logs vibevoice-api
```

### GPU not detected

Verify GPU access:
```bash
docker exec vibevoice-api nvidia-smi
```

### Out of memory

Reduce batch size or use smaller model:
```bash
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-0.5B
```

### Slow model loading

Mount HuggingFace cache:
```bash
-v ~/.cache/huggingface:/root/.cache/huggingface:rw
```

## Production Deployment

### With Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name tts.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
        client_max_body_size 10M;
    }
}
```

### With Docker Swarm

```bash
docker stack deploy -c docker-compose.yml vibevoice
```

### With Kubernetes

See `k8s/` directory for Kubernetes manifests (coming soon).

## Resource Requirements

### Minimum
- GPU: 8GB VRAM (for 1.5B model)
- RAM: 16GB
- Storage: 10GB

### Recommended
- GPU: 16GB+ VRAM (for Large model)
- RAM: 32GB
- Storage: 50GB (with model cache)

## Updates

Pull latest image:
```bash
docker-compose pull
docker-compose up -d
```

Rebuild after code changes:
```bash
docker-compose build --no-cache
docker-compose up -d
```

