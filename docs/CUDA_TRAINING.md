# CUDA Training Setup for Autocontext

This guide covers running autocontext with NVIDIA GPU acceleration using vLLM for local training and distillation.

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (`nvidia-smi` should work)
- vLLM installed
- 8GB+ VRAM recommended for 4B models, 24GB+ for 35B models

## Installation

### 1. Install vLLM

```bash
# With pip
pip install vllm

# Or with uv
uv pip install vllm
```

### 2. Pull Models (Optional - for vLLM)

```bash
# For Qwen 3.5 models via Ollama (if using Ollama backend)
ollama pull qwen3.5-4b
ollama pull qwen3.5-35b-a3b
```

## Usage

### Option 1: Use Existing Ollama Server

If you already have an Ollama server running (like for pi-agent), use it directly:

```bash
cd ~/autocontext-fork/autocontext
source .venv/bin/activate

export AUTOCONTEXT_AGENT_PROVIDER=ollama
export AUTOCONTEXT_AGENT_DEFAULT_MODEL=qwen3.5-4b
export AUTOCONTEXT_MODEL_COMPETITOR=qwen3.5-4b

uv run autoctx run --scenario grid_ctf --gens 5 --run-id cuda-training
```

### Option 2: Use vLLM Server

Start vLLM server with CUDA acceleration:

```bash
# Start vLLM with Qwen3.5-4B
bash scripts/start-vllm-cuda.sh qwen3.5-4b

# Or with Qwen3.5-35B-A3B (requires more VRAM)
bash scripts/start-vllm-cuda.sh qwen3.5-35b
```

Run training with vLLM:

```bash
# Quick training script
bash scripts/autocontext-cuda-train.sh qwen3.5-4b 8130 my-run

# Or manually
cd ~/autocontext-fork/autocontext
source .venv/bin/activate

export AUTOCONTEXT_AGENT_PROVIDER=vllm
export AUTOCONTEXT_AGENT_BASE_URL=http://localhost:8130/v1
export AUTOCONTEXT_AGENT_DEFAULT_MODEL=qwen3.5-4b

uv run autoctx run --scenario grid_ctf --gens 5 --run-id cuda-training
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AUTOCONTEXT_AGENT_PROVIDER` | Provider type: `ollama`, `vllm`, `cuda` | `anthropic` |
| `AUTOCONTEXT_AGENT_BASE_URL` | API endpoint URL | `http://localhost:8130/v1` |
| `AUTOCONTEXT_AGENT_DEFAULT_MODEL` | Model name | `qwen3.5-4b` |
| `AUTOCONTEXT_MODEL_COMPETITOR` | Competitor model | Same as `AUTOCONTEXT_AGENT_DEFAULT_MODEL` |

## GPU Selection

For multi-GPU systems, set CUDA_VISIBLE_DEVICES:

```bash
# Use GPU 0 only
export CUDA_VISIBLE_DEVICES=0

# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1
```

## Tensor Parallelism

For multi-GPU setups with vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model qwen3.5-35b-a3b \
    --port 8129 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9
```

## Health Check

```bash
# Check if server is running
curl http://localhost:8130/health

# List available models
curl http://localhost:8130/v1/models
```

## Troubleshooting

### Out of Memory

- Reduce `--gpu-memory-utilization` to 0.7 or 0.8
- Use quantization: `--quantization q4_0` or `--quantization fp8`
- Use a smaller model (4B instead of 35B)

### Server Won't Start

Check the log file:
```bash
tail -f /tmp/vllm-qwen3.5-4b.log
```

### Model Not Found

Ensure the model is available:
```bash
ollama list
# or
ls ~/.ollama/models/
```

## Related Scripts

- `scripts/start-vllm-cuda.sh` - Start vLLM server
- `scripts/autocontext-cuda-train.sh` - Full training pipeline
- `~/llama-server/hermes-qwen.sh` - Hermes launcher with local models
