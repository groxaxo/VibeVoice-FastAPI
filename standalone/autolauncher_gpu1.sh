#!/bin/bash
# Autolauncher for VibeVoice FastAPI Server on GPU 1
# This script automatically launches the FastAPI server on GPU 1

echo "🚀 VibeVoice AutoLauncher (GPU 1)"
echo "=================================="

# Set GPU to 1
export CUDA_VISIBLE_DEVICES=1

# Set models directory
export VIBEVOICE_MODELS_DIR=${VIBEVOICE_MODELS_DIR:-./models}

# Activate conda environment if it exists
if command -v conda &> /dev/null; then
    if conda env list | grep -q "vibevoice-standalone"; then
        echo "📦 Activating conda environment: vibevoice-standalone"
        eval "$(conda shell.bash hook)"
        conda activate vibevoice-standalone
    else
        echo "⚠️  Conda environment 'vibevoice-standalone' not found"
        echo "💡 To create it: conda env create -f environment.yml"
        echo "📍 Proceeding with current Python environment..."
    fi
fi

echo ""
echo "🎯 GPU Device: $CUDA_VISIBLE_DEVICES"
echo "📁 Models Directory: $VIBEVOICE_MODELS_DIR"
echo ""

# Check if flash-attn is installed
echo "🔍 Checking flash-attn installation..."
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "✅ flash-attn is installed"
else
    echo "⚠️  flash-attn not found"
    echo "💡 Install with: pip install flash-attn>=2.0.0"
    echo "📍 Proceeding anyway (will use fallback attention)..."
fi

echo ""
echo "🚀 Starting VibeVoice FastAPI Server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Change to the standalone directory
cd "$(dirname "$0")"

# Run FastAPI server with any additional arguments
python3 fastapi_server.py "$@"
