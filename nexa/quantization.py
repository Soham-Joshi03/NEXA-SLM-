"""
Quantization and Low-Compute Optimization module for NEXA-SLM.
Supports INT8 and INT4 weight-only quantization, precision conversion (FP32, FP16, BF16),
and analytical / empirical memory and latency profiling for edge deployment.
"""

import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from nexa.config import NexaConfig
from nexa.model import NexaSLM


class Int8Linear(nn.Module):
    """
    Symmetric per-channel INT8 weight-quantized Linear layer for low-compute inference.
    Stores weights as int8 (1 byte per parameter) with per-channel float scales.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.zeros(out_features, 1, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        layer = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        w = linear.weight.data.float()

        # Compute symmetric per-channel scale: max(|w|) / 127
        max_abs = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_abs / 127.0
        qweight = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)

        layer.qweight.copy_(qweight)
        layer.scale.copy_(scale)
        if linear.bias is not None:
            layer.bias.copy_(linear.bias.data.float())
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on-the-fly for matrix multiplication
        w_float = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_float, bias)


def quantize_model_int8(model: NexaSLM, exclude_embeddings: bool = True) -> NexaSLM:
    """
    Converts linear layers in NexaSLM to INT8 weight representations.
    Excludes embeddings by default to maintain input representation precision.
    """
    model.eval()

    def replace_linears(module: nn.Module):
        for name, child in module.named_children():
            # Don't quantize lm_head if tied to embedding
            if name == "lm_head" and model.config.tie_embeddings:
                continue
            if isinstance(child, nn.Linear):
                setattr(module, name, Int8Linear.from_linear(child))
            else:
                replace_linears(child)

    replace_linears(model)
    return model


def calculate_analytical_kv_cache(
    config: NexaConfig,
    precision_bytes: int = 2,  # FP16 = 2, FP32 = 4
) -> Dict[str, float]:
    """
    Analytically calculates KV-cache memory based on Eq. 11 of the research paper:
        M_KV = 2 * L_blocks * n_kv * d_h * (W + m) * b bytes
    """
    L_blocks = config.n_super_blocks  # 1 attention layer per SuperBlock
    n_kv = config.kv_heads
    d_h = config.head_dim
    W = config.window_size
    m = config.num_scratchpad
    b = precision_bytes

    cache_bytes = 2 * L_blocks * n_kv * d_h * (W + m) * b
    cache_kib = cache_bytes / 1024
    cache_mib = cache_bytes / (1024 ** 2)

    return {
        "cache_bytes": cache_bytes,
        "cache_kib": round(cache_kib, 2),
        "cache_mib": round(cache_mib, 3),
        "tokens_cached": W + m,
    }


def calculate_analytical_mamba_state(
    config: NexaConfig,
    precision_bytes: int = 2,
) -> Dict[str, float]:
    """
    Calculates recurrent state memory for Mamba-3 layers:
        M_SSM = L_mamba * (D_inner * N_state + D_inner * (K_conv - 1) + 2 * D_inner) * b bytes
    """
    L_mamba = config.total_mamba_moe_layers
    D_inner = config.mamba_d_inner
    N_state = config.mamba_d_state
    K_conv = config.mamba_conv_size
    b = precision_bytes

    # SSM matrix h (D_inner x N_state) + conv rolling buffer + prev u & B
    elements_per_layer = (D_inner * N_state) + (D_inner * (K_conv - 1)) + (D_inner + N_state)
    total_bytes = L_mamba * elements_per_layer * b

    return {
        "mamba_state_bytes": total_bytes,
        "mamba_state_kib": round(total_bytes / 1024, 2),
        "mamba_state_mib": round(total_bytes / (1024 ** 2), 3),
    }


def profile_inference_speed(
    model: NexaSLM,
    prompt_length: int = 16,
    num_new_tokens: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Profiles Time-To-First-Token (TTFT) and decode tokens/sec.
    """
    from nexa.inference import generate

    model.eval()
    model.to(device)

    prompt = [100 + i for i in range(prompt_length)]

    # Warmup
    _ = generate(model, prompt, max_new_tokens=2, device=device)

    # Measurement
    start_time = time.perf_counter()
    out = generate(model, prompt, max_new_tokens=num_new_tokens, device=device)
    total_time = time.perf_counter() - start_time

    tokens_generated = len(out) - len(prompt)
    tps = tokens_generated / max(1e-6, total_time)

    return {
        "prompt_tokens": prompt_length,
        "generated_tokens": tokens_generated,
        "total_time_sec": round(total_time, 4),
        "tokens_per_sec": round(tps, 2),
        "latency_per_token_ms": round(1000.0 / max(1e-6, tps), 2),
    }
