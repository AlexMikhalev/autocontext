#!/bin/bash
# Autocontext Local Training with llama-server (Compiled from Source)

set -e

cd "$(dirname "$0")/.."

MODEL="${1:-Qwen_Qwen3.5-4B-IQ4_XS.gguf}"
PORT="${2:-8130}"
RUN_ID="${3:-llama-training}"

echo "=== Autocontext Local llama-server Training ==="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Run ID: $RUN_ID"

# Start llama-server if not running
if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "Starting llama-server..."
    bash scripts/start-llama-server.sh "$PORT" "$MODEL"
    sleep 5
fi

# Verify server
if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "Error: llama-server not available on port $PORT"
    exit 1
fi

echo "llama-server ready"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | head -1
fi

# Activate venv
if [ -d "autocontext/.venv" ]; then
    source autocontext/.venv/bin/activate
else
    echo "Error: venv not found. Run: cd autocontext && uv venv && uv sync"
    exit 1
fi

# Run autocontext with llama provider (uses thinking=false by default)
export AUTOCONTEXT_AGENT_PROVIDER=llama
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

# Show playbook
if [ -f "autocontext/knowledge/grid_ctf/playbook.md" ]; then
    echo ""
    echo "=== Playbook ==="
    head -30 autocontext/knowledge/grid_ctf/playbook.md
fi
