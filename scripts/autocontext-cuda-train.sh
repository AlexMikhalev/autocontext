#!/bin/bash
# Autocontext CUDA Training Script
# Runs autocontext with vLLM on NVIDIA GPU for local training/distillation

set -e

cd "$(dirname "$0")/.."

MODEL="${1:-qwen3.5-4b}"
PORT="${2:-8130}"
RUN_ID="${3:-cuda-training}"

echo "=== Autocontext CUDA Training ==="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Run ID: $RUN_ID"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: No NVIDIA GPU detected"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv

# Start vLLM if not running
if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "Starting vLLM server..."
    bash scripts/start-vllm-cuda.sh "$MODEL"
    sleep 5
fi

# Verify vLLM is running
if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "Error: vLLM server not available on port $PORT"
    exit 1
fi

echo "vLLM server ready"

# Activate venv
if [ -d "autocontext/.venv" ]; then
    source autocontext/.venv/bin/activate
else
    echo "Error: venv not found. Run: cd autocontext && uv venv && uv sync"
    exit 1
fi

# Run autocontext with CUDA
export AUTOCONTEXT_AGENT_PROVIDER=vllm
export AUTOCONTEXT_AGENT_BASE_URL="http://localhost:$PORT/v1"
export AUTOCONTEXT_AGENT_DEFAULT_MODEL="$MODEL"
export AUTOCONTEXT_MODEL_COMPETITOR="$MODEL"

echo ""
echo "Starting training run..."
uv run autoctx run \
    --scenario grid_ctf \
    --gens 5 \
    --run-id "$RUN_ID"

echo ""
echo "=== Training Complete ==="
echo "Check status: uv run autoctx status $RUN_ID"
echo "View logs: tail -f /tmp/vllm-$MODEL.log"
