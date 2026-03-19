"""CUDAProvider — local model inference via vLLM on NVIDIA GPU.

Uses vLLM for high-performance inference with CUDA acceleration.
Supports tensor parallelism for multi-GPU setups.
"""
from __future__ import annotations

import logging
from typing import Any

from autocontext.providers.base import CompletionResult, LLMProvider, ProviderError

LOGGER = logging.getLogger(__name__)


class CUDAProvider(LLMProvider):
    """Provider using vLLM with CUDA for local strategy generation.

    Loads models served by vLLM server or connects to existing vLLM endpoint.
    Supports NVIDIA GPUs with tensor parallelism.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        api_key: str = "not-needed-for-local",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError(
                "openai package is required for CUDAProvider. Install with: uv add openai"
            ) from exc

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._default_model = model
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        LOGGER.info(
            f"CUDAProvider initialized: model={model}, tp={tensor_parallel_size}, gpu_mem={gpu_memory_utilization}"
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        """Generate a completion using the vLLM server."""
        model_id = model or self._default_model
        try:
            response = self._client.chat.completions.create(
                model=model_id,
                temperature=temperature if temperature > 0 else 0.7,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            raise ProviderError(f"vLLM CUDA API error: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        text = choice.message.content or "" if choice else ""

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
            }

        return CompletionResult(
            text=text,
            model=model_id,
            usage=usage,
        )

    def default_model(self) -> str:
        return self._default_model

    @property
    def name(self) -> str:
        return "cuda-vllm"
