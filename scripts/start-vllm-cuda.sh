#!/bin/bash
# CUDA Training Setup for Autocontext
# Starts vLLM server with Qwen3.5-4B model for local training/distillation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-qwen3.5-4b}"

case "$MODEL" in
    qwen3.5-4b|4b)
        PORT=8130
        MODEL_NAME="qwen3.5-4b"
        QUANTIZATION="q4_0"
        GPU_LAYERS=40
        echo "Starting vLLM with Qwen3.5-4B on port $PORT"
        ;;
    qwen3.5-35b|35b|35b-a3b)
        PORT=8129
        MODEL_NAME="qwen3.5-35b-a3b"
        QUANTIZATION="q4_k_m"
        GPU_LAYERS=-1
        echo "Starting vLLM with Qwen3.5-35B-A3B on port $PORT (requires multi-GPU or large VRAM)"
        ;;
    *)
        echo "Usage: $0 [qwen3.5-4b|qwen3.5-35b]"
        exit 1
        ;;
esac

# Check if server is already running
if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "vLLM server already running on port $PORT"
    exit 0
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. CUDA not available."
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check vLLM availability
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# Start vLLM server
echo "Starting vLLM server..."
export CUDA_VISIBLE_DEVICES=0

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --quantization "$QUANTIZATION" \
    --enforce-eager \
    > /tmp/vllm-$MODEL.log 2>&1 &

VLLM_PID=$!

echo "vLLM started with PID: $VLLM_PID"
echo "Log file: /tmp/vllm-$MODEL.log"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "vLLM server ready on port $PORT"
        exit 0
    fi
    sleep 1
done

echo "Error: vLLM server failed to start. Check log: /tmp/vllm-$MODEL.log"
exit 1
