"""
Sliding Window Grouped Query Attention (SW-GQA) with Learnable Scratchpad Access.
Supports:
1. True O(L * (W + m)) chunked local attention.
2. Dense masked reference attention for verification.
3. Constant-memory O(W + m) KV-cached incremental decoding for autoregressive inference.
4. Rotary Position Embeddings (RoPE).
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding (RoPE) to Query and Key tensors.
    Args:
        xq: (B, H_q, L, D_h)
        xk: (B, H_kv, L, D_h)
        freqs_cos: (L, D_h // 2)
        freqs_sin: (L, D_h // 2)
    Returns:
        xq_out, xk_out: Rotated tensors of same shape
    """
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    L = xq.shape[2]
    cos = freqs_cos[:L].view(1, 1, L, -1)
    sin = freqs_sin[:L].view(1, 1, L, -1)

    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SWGQA(nn.Module):
    """
    Sliding Window Grouped Query Attention with Scratchpad Prefix Attention.
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.q_heads = config.q_heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        self.num_scratchpad = config.num_scratchpad
        self.attn_mode = getattr(config, "attn_mode", "chunked")
        self.num_rep = self.q_heads // self.kv_heads

        self.q_proj = nn.Linear(self.d_model, self.q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.q_heads * self.head_dim, self.d_model, bias=False)

        # Precompute RoPE rotary frequencies
        dim = self.head_dim
        end = config.max_seq_len + self.num_scratchpad + 512
        rope_base = getattr(config, "rope_base", 10000.0)
        freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, dtype=torch.float32)
        freqs = torch.outer(t, freqs).float()
        self.register_buffer("freqs_cos", torch.cos(freqs), persistent=False)
        self.register_buffer("freqs_sin", torch.sin(freqs), persistent=False)

    def _dense_swgqa_forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, L: int
    ) -> torch.Tensor:
        """Reference dense masked SW-GQA attention for testing and short sequences."""
        B = q.shape[0]

        # Expand KV heads for GQA
        k_rep = k.repeat_interleave(self.num_rep, dim=1)
        v_rep = v.repeat_interleave(self.num_rep, dim=1)

        # Causal mask: j <= i
        mask_tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
        # Sliding window mask: i - j < window_size
        mask_sw = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=q.device), diagonal=-self.window_size + 1
        )
        # Scratchpad mask: any token can attend to the first m scratchpad tokens
        mask_scratchpad = torch.zeros(L, L, dtype=torch.bool, device=q.device)
        mask_scratchpad[:, : self.num_scratchpad] = True

        valid_attend = mask_tril & (mask_sw | mask_scratchpad)
        attn_bias = torch.zeros(L, L, device=q.device, dtype=q.dtype).masked_fill(
            ~valid_attend, float("-inf")
        )

        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_bias.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        # Clean any NaNs in padded positions
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v_rep)
        return out

    def _chunked_swgqa_forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, L: int
    ) -> torch.Tensor:
        """
        True O(L * (W + m)) chunked local sliding window attention with persistent scratchpad.
        Avoids allocating or computing an L x L matrix.
        """
        B, H, _, D = q.shape
        m = self.num_scratchpad
        W = self.window_size

        k_rep = k.repeat_interleave(self.num_rep, dim=1)
        v_rep = v.repeat_interleave(self.num_rep, dim=1)

        # If sequence is within scratchpad + window, dense attention is already small and exact
        if L <= m + W:
            return self._dense_swgqa_forward(q, k, v, L)

        outs = []
        # 1. Scratchpad tokens [0 : m] attend causally to scratchpad
        if m > 0:
            q_sp = q[:, :, :m]
            k_sp = k_rep[:, :, :m]
            v_sp = v_rep[:, :, :m]
            mask_sp = torch.tril(torch.ones(m, m, dtype=torch.bool, device=q.device))
            bias_sp = torch.zeros(m, m, device=q.device, dtype=q.dtype).masked_fill(
                ~mask_sp, float("-inf")
            )
            scores_sp = torch.matmul(q_sp, k_sp.transpose(-2, -1)) / math.sqrt(self.head_dim) + bias_sp
            attn_sp = F.softmax(scores_sp, dim=-1)
            attn_sp = torch.nan_to_num(attn_sp, nan=0.0)
            outs.append(torch.matmul(attn_sp, v_sp))

        # 2. Sequence tokens [m : L] processed in chunks of size W
        k_sp_all = k_rep[:, :, :m] if m > 0 else None
        v_sp_all = v_rep[:, :, :m] if m > 0 else None

        for i_start in range(m, L, W):
            i_end = min(i_start + W, L)
            q_blk = q[:, :, i_start:i_end]
            blk_len = i_end - i_start

            # Window of keys: [w_start, i_end)
            w_start = max(m, i_start - W)
            k_win = k_rep[:, :, w_start:i_end]
            v_win = v_rep[:, :, w_start:i_end]

            if m > 0:
                k_ctx = torch.cat([k_sp_all, k_win], dim=2)
                v_ctx = torch.cat([v_sp_all, v_win], dim=2)
            else:
                k_ctx = k_win
                v_ctx = v_win

            ctx_len = k_ctx.shape[2]

            # Construct causal / window validity mask for this block
            q_pos = torch.arange(i_start, i_end, device=q.device).unsqueeze(1)  # (blk_len, 1)
            k_pos_parts = []
            if m > 0:
                k_pos_parts.append(torch.arange(0, m, device=q.device))
            k_pos_parts.append(torch.arange(w_start, i_end, device=q.device))
            k_pos = torch.cat(k_pos_parts).unsqueeze(0)  # (1, ctx_len)

            valid_blk = (k_pos <= q_pos) & ((k_pos < m) | (q_pos - k_pos < W))
            bias_blk = torch.zeros(blk_len, ctx_len, device=q.device, dtype=q.dtype).masked_fill(
                ~valid_blk, float("-inf")
            )

            scores_blk = (
                torch.matmul(q_blk, k_ctx.transpose(-2, -1)) / math.sqrt(self.head_dim) + bias_blk
            )
            attn_blk = F.softmax(scores_blk, dim=-1)
            attn_blk = torch.nan_to_num(attn_blk, nan=0.0)
            outs.append(torch.matmul(attn_blk, v_ctx))

        return torch.cat(outs, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor, shape (B, L, D_model)
        Returns:
            out: Projected attention tensor, shape (B, L, D_model)
        """
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, self.freqs_cos, self.freqs_sin)

        if self.attn_mode == "chunked":
            out = self._chunked_swgqa_forward(q, k, v, L)
        else:
            out = self._dense_swgqa_forward(q, k, v, L)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)

    def step(
        self,
        x: torch.Tensor,
        cache: Dict,
        block_idx: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Single-token autoregressive decoding step with bounded KV cache:
        Cache size is strictly capped at (num_scratchpad + window_size).
        """
        B, L, D = x.shape
        pos_key = f"pos_{block_idx}_{layer_idx}"
        k_key = f"swgqa_{block_idx}_{layer_idx}_k"
        v_key = f"swgqa_{block_idx}_{layer_idx}_v"

        pos = cache.get(pos_key, 0)

        # Project Q, K, V for current token
        q = self.q_proj(x).view(B, L, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE at exact sequence position
        q_rot, k_rot = apply_rotary_emb(
            q, k, self.freqs_cos[pos : pos + 1], self.freqs_sin[pos : pos + 1]
        )

        k_cache = cache.get(k_key, [])
        v_cache = cache.get(v_key, [])

        k_cache.append(k_rot)
        v_cache.append(v)

        # Evict oldest sliding-window token when cache capacity exceeded
        max_cache_tokens = self.num_scratchpad + self.window_size
        if len(k_cache) > max_cache_tokens:
            # Index num_scratchpad is the oldest non-scratchpad token in the window
            k_cache.pop(self.num_scratchpad)
            v_cache.pop(self.num_scratchpad)

        cache[k_key] = k_cache
        cache[v_key] = v_cache
        cache[pos_key] = pos + 1

        keys = torch.cat(k_cache, dim=2)
        vals = torch.cat(v_cache, dim=2)

        # GQA Repetition
        keys = keys.repeat_interleave(self.num_rep, dim=1)
        vals = vals.repeat_interleave(self.num_rep, dim=1)

        scores = torch.matmul(q_rot, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, vals).transpose(1, 2).contiguous().view(B, 1, -1)

        return self.o_proj(out), cache
