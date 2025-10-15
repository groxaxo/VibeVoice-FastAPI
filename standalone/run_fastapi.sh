#!/bin/bash
# Quick start script for FastAPI server

echo "🚀 Starting VibeVoice FastAPI Server..."

# Set default GPU to 0 if not specified
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Set models directory
export VIBEVOICE_MODELS_DIR=${VIBEVOICE_MODELS_DIR:-./models}

echo "GPU Device: $CUDA_VISIBLE_DEVICES"
echo "Models Directory: $VIBEVOICE_MODELS_DIR"
echo ""

# Run FastAPI server
python3 fastapi_server.py "$@"
