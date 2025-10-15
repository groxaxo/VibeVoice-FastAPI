#!/bin/bash
# Quick start script for Gradio interface

echo "🎙️ Starting VibeVoice Gradio Interface..."

# Set default GPU to 0 if not specified
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Set models directory
export VIBEVOICE_MODELS_DIR=${VIBEVOICE_MODELS_DIR:-./models}

echo "GPU Device: $CUDA_VISIBLE_DEVICES"
echo "Models Directory: $VIBEVOICE_MODELS_DIR"
echo ""

# Run Gradio app
python3 gradio_app.py "$@"
