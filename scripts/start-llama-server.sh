#!/bin/bash
# Start Local llama-server with Qwen 3.5 Model (Compiled from Source)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="/home/alex/models/qwen3.5"

PORT="${1:-8130}"
MODEL="${2:-Qwen_Qwen3.5-4B-IQ4_XS.gguf}"

echo "=== Starting Local llama-server ==="
echo "Model: $MODEL"
echo "Port: $PORT"

# Check if server is already running
if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "Server already running on port $PORT"
    exit 0
fi

# Check GPU availability
GPU_LAYERS=0
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
    GPU_LAYERS=99
    echo "GPU available - using CUDA acceleration"
else
    echo "No GPU - using CPU"
fi

# Check if compiled llama-server exists
if [ -f "$SCRIPT_DIR/llama-server" ]; then
    LLAMA_SERVER="$SCRIPT_DIR/llama-server"
else
    echo "Error: llama-server not found in $SCRIPT_DIR"
    echo "Build with: cd $SCRIPT_DIR/llama.cpp && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=ON && cmake --build . --target llama-server -j\$(nproc)"
    exit 1
fi

# Ensure library path
export LD_LIBRARY_PATH="$SCRIPT_DIR:$LD_LIBRARY_PATH"

# Start server
echo "Starting llama-server..."
$LLAMA_SERVER \
    -m "$MODEL_DIR/$MODEL" \
    -c 64000 \
    --port "$PORT" \
    --host "0.0.0.0" \
    -ngl "$GPU_LAYERS" \
    --reasoning off \
    -tb 6 \
    -t 6 \
    > /tmp/llama-server-$PORT.log 2>&1 &

echo "Server started with PID: $!"
echo "Log: /tmp/llama-server-$PORT.log"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server ready on port $PORT"
        curl -s "http://localhost:$PORT/health"
        echo ""
        exit 0
    fi
    sleep 1
done

echo "Error: Server failed to start"
echo "Check log: /tmp/llama-server-$PORT.log"
tail -20 /tmp/llama-server-$PORT.log
exit 1
