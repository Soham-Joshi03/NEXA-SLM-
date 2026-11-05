"""
High-performance inference and generation pipeline for NEXA-SLM.
Supports:
1. Fast batched scratchpad & prompt prefill.
2. O(W + m) constant-size KV caching and recurrent SSM state caching.
3. Temperature, Top-K, and Top-P (nucleus) sampling.
4. End-of-sequence (EOS) early stopping.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from nexa.config import NexaConfig
from nexa.model import NexaSLM, SuperBlock


def inference_step(
    model: NexaSLM,
    hidden: torch.Tensor,
    cache: Dict,
) -> Tuple[torch.Tensor, Dict]:
    """
    Passes a single token (B, 1, D) through all SuperBlocks using cached step() calls.
    """
    for block_idx, block in enumerate(model.blocks):
        for layer_idx, layer in enumerate(block.layers):
            if layer_idx == 2:
                # SW-GQA + Shared MLP
                x_norm = layer["norm1"](hidden)
                attn_out, cache = layer["attn"].step(x_norm, cache, block_idx, layer_idx)
                hidden = hidden + attn_out

                x_norm2 = layer["norm2"](hidden)
                mlp_out = layer["shared_mlp"](x_norm2)
                hidden = hidden + mlp_out
            else:
                # Mamba-3 + MoE
                x_norm = layer["norm1"](hidden)
                mamba_out, cache = layer["mamba"].step(x_norm, cache, block_idx, layer_idx)
                hidden = hidden + mamba_out

                x_norm2 = layer["norm2"](hidden)
                moe_out, _, _, _ = layer["moe"](x_norm2)
                hidden = hidden + moe_out

    return model.norm(hidden), cache


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> int:
    """
    Samples next token ID from 1D or 2D unnormalized logits.
    """
    if logits.ndim > 1:
        logits = logits[0, -1, :]  # (vocab_size,)

    if temperature < 1e-5:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature

    # Top-K filtering
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = float("-inf")

    probs = F.softmax(logits, dim=-1)

    # Top-P (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: NexaSLM,
    prompt_tokens: List[int],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    eos_token_id: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> List[int]:
    """
    Generates text autoregressively using cached incremental decoding.
    Handles virtual scratchpad prefill, prompt prefill, and dynamic sampling.
    """
    model.eval()
    if device is not None:
        model = model.to(device)
    else:
        device = next(model.parameters()).device

    cache: Dict = {}
    generated = list(prompt_tokens)

    # 1. Prefill Scratchpad tokens
    m = model.config.num_scratchpad
    if m > 0 and model.scratchpad_manager.scratchpad is not None:
        scratchpad_emb = model.scratchpad_manager.scratchpad.to(device)
        for i in range(m):
            sp_tok = scratchpad_emb[i].view(1, 1, -1)
            _, cache = inference_step(model, sp_tok, cache)

    # 2. Prefill Prompt tokens
    if len(prompt_tokens) == 0:
        raise ValueError("prompt_tokens cannot be empty.")

    last_hidden = None
    for idx, token_id in enumerate(prompt_tokens):
        tok_t = torch.tensor([[token_id]], device=device, dtype=torch.long)
        tok_emb = model.embedding(tok_t)
        last_hidden, cache = inference_step(model, tok_emb, cache)

    # 3. Predict first new token from the prompt's last hidden state
    logits = model.lm_head(last_hidden)
    next_tok = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    generated.append(next_tok)

    if next_tok == eos_token_id:
        return generated

    # 4. Autoregressive generation loop
    for _ in range(max_new_tokens - 1):
        tok_t = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
        tok_emb = model.embedding(tok_t)
        hidden, cache = inference_step(model, tok_emb, cache)
        logits = model.lm_head(hidden)

        next_tok = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        generated.append(next_tok)

        if next_tok == eos_token_id:
            break

    return generated
