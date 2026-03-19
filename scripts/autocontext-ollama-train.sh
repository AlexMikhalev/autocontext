#!/bin/bash
# Autocontext Local Training with Ollama
# Works reliably with qwen2.5-coder model

set -e

cd "$(dirname "$0")/.."

MODEL="${1:-qwen2.5-coder}"
RUN_ID="${2:-ollama-training}"

echo "=== Autocontext Ollama Training ==="
echo "Model: $MODEL"
echo "Run ID: $RUN_ID"

# Check Ollama is running
if ! curl -s http://localhost:11434/health > /dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 3
fi

# Verify Ollama
if ! curl -s http://localhost:11434/health > /dev/null 2>&1; then
    echo "Error: Ollama not available"
    exit 1
fi

echo "Ollama server ready"

# Check GPU (optional)
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

# Run autocontext with Ollama
export AUTOCONTEXT_AGENT_PROVIDER=ollama
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
