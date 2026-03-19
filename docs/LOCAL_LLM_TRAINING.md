# Local LLM Training Setup

## Current Status

### Working: Ollama (Recommended)

Ollama works reliably for local training with autocontext.

**Available Models:**
```bash
ollama list
# qwen2.5-coder:latest  - 4.7GB - Works well with JSON output
# qwen3:4b              - 2.5GB - Fails JSON output
# qwen3:8b              - 5.2GB - Needs /no_think prefix
```

**Quick Start:**
```bash
cd ~/autocontext-fork/autocontext
source .venv/bin/activate

export AUTOCONTEXT_AGENT_PROVIDER=ollama
export AUTOCONTEXT_AGENT_DEFAULT_MODEL=qwen2.5-coder
export AUTOCONTEXT_MODEL_COMPETITOR=qwen2.5-coder

# Run training
uv run autoctx run --scenario grid_ctf --gens 5 --run-id training-run
```

---

### llama-server (Compiled from Source - Working)

We compiled llama-server from source with the latest llama.cpp.

**Location:** `~/llama-server/llama-server`

**Build Notes:**
- Compiled with MTMD support and server enabled
- Using CPU backend (CUDA toolkit 11.2 incompatible with GCC 13)
- Model: Qwen3.5-4B with `--reasoning off` flag

**Important:** Qwen3.5 models use thinking mode by default. You MUST use `--reasoning off` when starting the server.

**Start Server:**
```bash
cd ~/llama-server
./llama-server \
    -m /home/alex/models/qwen3.5/Qwen_Qwen3.5-4B-IQ4_XS.gguf \
    -c 64000 \
    --port 8130 \
    --host "0.0.0.0" \
    -ngl 99 \
    --reasoning off \
    -tb 6 -t 6
```

**Or use the script:**
```bash
bash ~/autocontext-fork/scripts/start-llama-server.sh
```

**Run Training:**
```bash
cd ~/autocontext-fork/autocontext
source .venv/bin/activate

export AUTOCONTEXT_AGENT_PROVIDER=llama
export AUTOCONTEXT_AGENT_DEFAULT_MODEL=Qwen_Qwen3.5-4B-IQ4_XS.gguf

uv run autoctx run --scenario grid_ctf --gens 5 --run-id llama-training
```

**Or use the script:**
```bash
bash ~/autocontext-fork/scripts/autocontext-llama-train.sh
```

---

### llama-server Binary (Not Working)

The compiled llama-server binaries (`~/llama-server/llama-server` and `llama-server-cuda`) require a newer version of `libmtmd.so.0` than what's available in the system.

**Error:**
```
./llama-server: symbol lookup error: undefined symbol: llama_sampler_init_adoptive_p
```

**To Fix (requires rebuilding):**
1. Clone llama.cpp and compile with matching versions
2. Or install compatible llama-cpp-python package
3. Or use prebuilt binaries from llama.cpp releases

**Libraries found but incompatible:**
- `/home/alex/.local/lib/python3.10/site-packages/llama_cpp/lib/` - v0.3.x
- llama-server binary - expects v0.3.6+

---

## Alternative: Using llama.cpp with Ollama Backend

Ollama uses llama.cpp internally. For GPU acceleration:

```bash
# Check GPU support
nvidia-smi

# Ollama with CUDA (if available)
curl http://localhost:11434/api/tags
```

---

## Recommended Training Setup

### Option 1: Ollama + qwen2.5-coder (Works)

```bash
cd ~/autocontext-fork/autocontext
source .venv/bin/activate

export AUTOCONTEXT_AGENT_PROVIDER=ollama
export AUTOCONTEXT_AGENT_DEFAULT_MODEL=qwen2.5-coder
uv run autoctx run --scenario grid_ctf --gens 5 --run-id <name>
```

### Option 2: Ollama + qwen3:8b (Needs /no_think)

```bash
export AUTOCONTEXT_AGENT_DEFAULT_MODEL=qwen3:8b
# Note: Model requires /no_think prefix - not yet implemented in autocontext
```

### Option 3: Build llama-server (Future)

To use direct llama-server:

1. Build llama.cpp from source matching the binary version
2. Or find prebuilt binaries
3. Update provider to use OpenAI-compatible endpoint

---

## Pulling Models

```bash
# Qwen 2.5 Coder (works)
ollama pull qwen2.5-coder

# Qwen 3 (4B - doesn't follow JSON well)
ollama pull qwen3:4b

# Qwen 3 (8B - needs /no_think)
ollama pull qwen3:8b
```

---

## Troubleshooting

### Model not responding
```bash
# Check Ollama status
curl http://localhost:11434/health

# Check logs
systemctl status ollama
```

### Training fails with JSON error
- Model not following JSON format
- Switch to `qwen2.5-coder` which handles JSON better

### Out of memory
- Reduce context size
- Use smaller model (4B instead of 35B)
- Check GPU memory: `nvidia-smi`
