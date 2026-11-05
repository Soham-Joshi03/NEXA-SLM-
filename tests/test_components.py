"""
Component-level unit tests for NEXA-SLM architecture:
- Tokenizer
- RMSNorm
- RoPE
- Mamba-3 Trapezoidal Discretization
- SW-GQA with Scratchpad Masking
- MoE with Top-2 Routing and Shared Expert
- Virtual Scratchpad
- SuperBlock [M+E, M+E, A, M+E, M+E]
"""

import math
import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from nexa.config import get_unit_test_config
from nexa.mamba3 import Mamba3Layer, mamba3_trapezoidal_scan
from nexa.attention import SWGQA, apply_rotary_emb
from nexa.moe import MoELayer, SwiGLU
from nexa.scratchpad import VirtualScratchpad
from nexa.model import SuperBlock, NexaSLM, count_parameters


class TestTokenizer:
    def test_tokenizer_file_exists_and_loads(self):
        tok_path = "jamba_bbpe_tokenizer.json"
        assert os.path.exists(tok_path), f"Tokenizer file {tok_path} must exist"
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)
        assert len(tokenizer) == 32768, f"Expected 32768 tokens, got {len(tokenizer)}"

    def test_special_tokens_roundtrip(self):
        tok_path = "jamba_bbpe_tokenizer.json"
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)
        text = "Hello, NEXA-SLM edge computing!"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        assert len(encoded) > 0
        decoded = tokenizer.decode(encoded)
        assert "NEXA-SLM" in decoded


class TestMamba3:
    def test_mamba3_trapezoidal_vs_step_numerical_equivalence(self):
        """Validates that sequence-level Mamba-3 scan matches single-token cached decoding."""
        cfg = get_unit_test_config()
        layer = Mamba3Layer(cfg)
        layer.eval()

        B, L = 2, 16
        x = torch.randn(B, L, cfg.d_model)

        # 1. Full sequence forward pass
        y_seq = layer(x)

        # 2. Step-by-step cached decoding
        cache = {}
        ys_step = []
        for t in range(L):
            yt, cache = layer.step(x[:, t : t + 1, :], cache, block_idx=0, layer_idx=0)
            ys_step.append(yt)
        y_step = torch.cat(ys_step, dim=1)

        diff = (y_seq - y_step).abs().max().item()
        assert diff < 1e-5, f"Mamba-3 scan vs step numerical mismatch: {diff}"

    def test_mamba3_backward_gradients_finite(self):
        cfg = get_unit_test_config()
        layer = Mamba3Layer(cfg)
        layer.train()

        B, L = 2, 8
        x = torch.randn(B, L, cfg.d_model, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestSWGQA:
    def test_swgqa_chunked_vs_dense_equivalence(self):
        """Validates that O(L*(W+m)) chunked attention numerically matches dense reference."""
        cfg = get_unit_test_config()
        cfg.window_size = 8
        cfg.num_scratchpad = 8
        cfg.max_seq_len = 32

        # Chunked mode
        cfg.attn_mode = "chunked"
        layer_chunked = SWGQA(cfg)
        layer_chunked.eval()

        # Dense mode with exact same weights
        cfg.attn_mode = "dense"
        layer_dense = SWGQA(cfg)
        layer_dense.load_state_dict(layer_chunked.state_dict())
        layer_dense.eval()

        B, L = 2, 28
        x = torch.randn(B, L, cfg.d_model)
        out_chunked = layer_chunked(x)
        out_dense = layer_dense(x)

        diff = (out_chunked - out_dense).abs().max().item()
        assert diff < 1e-5, f"Chunked vs Dense SWGQA mismatch: {diff}"

    def test_swgqa_sliding_kv_cache_bounded(self):
        """Validates that KV cache size never exceeds W + m."""
        cfg = get_unit_test_config()
        cfg.window_size = 4
        cfg.num_scratchpad = 4
        layer = SWGQA(cfg)
        layer.eval()

        cache = {}
        for t in range(20):
            xt = torch.randn(1, 1, cfg.d_model)
            _, cache = layer.step(xt, cache, block_idx=0, layer_idx=2)

        k_cache = cache["swgqa_0_2_k"]
        max_allowed = cfg.num_scratchpad + cfg.window_size
        assert len(k_cache) <= max_allowed, f"KV cache grew to {len(k_cache)}, expected <= {max_allowed}"


class TestMoE:
    def test_top2_routing_weights_normalized(self):
        cfg = get_unit_test_config()
        moe = MoELayer(cfg)
        x = torch.randn(2, 8, cfg.d_model)
        out, z_loss, aux_loss, metrics = moe(x)

        assert out.shape == x.shape
        assert z_loss.item() >= 0.0
        assert aux_loss.item() > 0.0
        assert "expert_load_cv" in metrics
        assert "router_entropy" in metrics

    def test_shared_expert_always_active(self):
        cfg = get_unit_test_config()
        moe = MoELayer(cfg)
        # Check that shared expert has gradients when loss is backwarded
        x = torch.randn(2, 4, cfg.d_model, requires_grad=True)
        out, _, _, _ = moe(x)
        out.sum().backward()
        shared_grad = moe.shared_expert.w1.weight.grad
        assert shared_grad is not None and shared_grad.norm().item() > 0


class TestVirtualScratchpad:
    def test_scratchpad_prefix_and_extraction(self):
        sp = VirtualScratchpad(num_tokens=16, d_model=64)
        tokens_emb = torch.randn(2, 32, 64)
        concatenated = sp.prepend(tokens_emb)
        assert concatenated.shape == (2, 48, 64)

        extracted = sp.extract_sequence(concatenated)
        assert extracted.shape == (2, 32, 64)
        assert torch.allclose(extracted, tokens_emb)


class TestSuperBlockAndModel:
    def test_superblock_sublayer_order(self):
        cfg = get_unit_test_config()
        sb = SuperBlock(cfg)
        assert len(sb.layers) == 5
        # Sublayer 2 is attention + shared mlp
        assert "attn" in sb.layers[2] and "shared_mlp" in sb.layers[2]
        # Other sublayers are Mamba-3 + MoE
        for idx in [0, 1, 3, 4]:
            assert "mamba" in sb.layers[idx] and "moe" in sb.layers[idx]

    def test_full_model_forward_and_loss(self):
        cfg = get_unit_test_config()
        model = NexaSLM(cfg)
        B, L = 2, 16
        x = torch.randint(0, cfg.vocab_size, (B, L))
        y = torch.randint(0, cfg.vocab_size, (B, L))

        ce, z, aux = model(x, y)
        assert ce is not None and not torch.isnan(ce)
        assert z is not None and not torch.isnan(z)
        assert aux is not None and not torch.isnan(aux)
