"""
Core model definitions for NEXA-SLM:
- SuperBlock: 5 sub-layers [M+E, M+E, A, M+E, M+E]
- NexaSLM (and alias PerfectSLM for backward compatibility)
- Parameter counting utility with active-vs-total analysis
"""

import math
from typing import Dict, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

from nexa.config import NexaConfig
from nexa.mamba3 import Mamba3Layer
from nexa.attention import SWGQA
from nexa.moe import MoELayer, SwiGLU
from nexa.scratchpad import VirtualScratchpad


class SuperBlock(nn.Module):
    """
    NEXA-SLM SuperBlock composed of 5 sub-layers in fixed sequence:
        [M+E, M+E, A, M+E, M+E]
    where:
        M+E: RMSNorm -> Mamba-3 -> residual + RMSNorm -> MoE (top-2 + shared) -> residual
        A:   RMSNorm -> SW-GQA -> residual + RMSNorm -> Shared MLP (SwiGLU) -> residual
    """

    def __init__(self, config: NexaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()

        for i in range(5):
            if i == 2:  # Sub-layer 2 is Attention + Shared MLP
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "norm1": RMSNorm(config.d_model),
                            "attn": SWGQA(config),
                            "norm2": RMSNorm(config.d_model),
                            "shared_mlp": SwiGLU(config.d_model, config.moe_ffn_dim),
                        }
                    )
                )
            else:  # Sub-layers 0, 1, 3, 4 are Mamba-3 + MoE
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "norm1": RMSNorm(config.d_model),
                            "mamba": Mamba3Layer(config),
                            "norm2": RMSNorm(config.d_model),
                            "moe": MoELayer(config),
                        }
                    )
                )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Forward pass through one SuperBlock.
        Args:
            x: Input tensor, shape (B, L_total, D_model)
        Returns:
            x: Processed tensor, shape (B, L_total, D_model)
            tot_z: Accumulated router z-loss
            tot_aux: Accumulated MoE auxiliary loss
            metrics: List of layer-wise expert metrics
        """
        tot_z = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        tot_aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        metrics = []

        for i, layer in enumerate(self.layers):
            if i == 2:
                # SW-GQA + Shared MLP
                x = x + layer["attn"](layer["norm1"](x))
                x = x + layer["shared_mlp"](layer["norm2"](x))
            else:
                # Mamba-3 + MoE
                x = x + layer["mamba"](layer["norm1"](x))
                moe_out, z, aux, m = layer["moe"](layer["norm2"](x))
                x = x + moe_out
                tot_z = tot_z + z
                tot_aux = tot_aux + aux
                metrics.append(m)

        return x, tot_z, tot_aux, metrics


class NexaSLM(nn.Module):
    """
    NEXA-SLM: Scalable Hybrid Mamba-Attention Mixture-of-Experts Small Language Model.
    """

    def __init__(self, config: NexaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        # Token embedding and learnable virtual scratchpad
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.scratchpad_manager = VirtualScratchpad(
            config.num_scratchpad, config.d_model, std=config.initializer_range
        )
        # Compatibility attribute
        self.scratchpad = self.scratchpad_manager.scratchpad

        # Repeating SuperBlocks
        self.blocks = nn.ModuleList([SuperBlock(config) for _ in range(config.n_super_blocks)])

        # Final RMSNorm and LM projection head
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def gradient_checkpointing_enable(self):
        """Enables activation checkpointing across SuperBlocks to reduce training VRAM."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Union[
        Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor],
        Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass for causal language modeling.

        Args:
            x: Input token IDs, shape (B, L)
            targets: Optional ground-truth next-token targets, shape (B, L)
            return_logits: Whether to return logits alongside loss

        Returns:
            (ce_loss, total_z_loss, total_aux_loss) or with logits
        """
        B, L = x.shape

        # 1. Embed input tokens and prepend virtual scratchpad
        tok_emb = self.embedding(x)
        hidden = self.scratchpad_manager.prepend(tok_emb)  # (B, m + L, D)

        # 2. Forward through SuperBlocks
        total_z_loss = torch.tensor(0.0, device=x.device)
        total_aux_loss = torch.tensor(0.0, device=x.device)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Custom forward for torch checkpoint
                def create_custom_forward(b):
                    def custom_forward(h):
                        return b(h)
                    return custom_forward
                hidden, z, aux, _ = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), hidden, use_reentrant=False
                )
            else:
                hidden, z, aux, _ = block(hidden)

            total_z_loss = total_z_loss + z
            total_aux_loss = total_aux_loss + aux

        # 3. Extract sequence tokens (strip scratchpad prefix) and apply final RMSNorm
        hidden_seq = self.scratchpad_manager.extract_sequence(hidden)  # (B, L, D)
        hidden_seq = self.norm(hidden_seq)

        # 4. Project to vocabulary logits
        logits = self.lm_head(hidden_seq)  # (B, L, vocab_size)

        # 5. Compute causal Cross-Entropy Loss if targets are provided
        ce_loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100
            )

        if return_logits:
            return ce_loss, total_z_loss, total_aux_loss, logits
        return ce_loss, total_z_loss, total_aux_loss


# Backward compatibility alias
PerfectSLM = NexaSLM


def count_parameters(model: nn.Module, config: Optional[NexaConfig] = None) -> Dict[str, Any]:
    """
    Calculates detailed parameter counts and memory requirements.

    Returns:
        Dictionary containing:
        - total_trainable: Total trainable weights
        - total_non_trainable: Non-trainable weights
        - embedding_params: Parameters in embedding layer
        - scratchpad_params: Parameters in virtual scratchpad
        - expert_params: Parameters in all routed and shared experts
        - active_params_per_token: Parameters activated per token pass
        - per_superblock: Parameters in one SuperBlock
        - checkpoint_sizes_mb: Dict of model sizes in MB across dtypes
    """
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # Embedding and scratchpad params
    embed_params = sum(p.numel() for p in model.embedding.parameters())
    scratch_params = (
        model.scratchpad_manager.scratchpad.numel()
        if model.scratchpad_manager.scratchpad is not None
        else 0
    )

    # Experts parameter count
    expert_params = 0
    for name, param in model.named_parameters():
        if "moe.experts" in name or "moe.shared_expert" in name or "shared_mlp" in name:
            expert_params += param.numel()

    # Active parameters calculation:
    # Router selects top-2 experts out of 8, plus 1 shared expert is always active
    # In each SuperBlock:
    # 4 MoE layers: each has 8 routed experts + 1 shared expert.
    # In active pass: 2 routed experts + 1 shared expert are computed.
    # Inactive routed experts: (8 - 2) = 6 routed experts per MoE layer.
    inactive_expert_params = 0
    for name, module in model.named_modules():
        if isinstance(module, MoELayer):
            # Each inactive routed expert has parameters of module.experts[0]
            one_expert_params = sum(p.numel() for p in module.experts[0].parameters())
            num_inactive = module.num_experts - module.num_active
            inactive_expert_params += one_expert_params * num_inactive

    active_params = total_trainable - inactive_expert_params

    # Single superblock params
    first_sb_params = sum(p.numel() for p in model.blocks[0].parameters()) if len(model.blocks) > 0 else 0

    checkpoint_sizes_mb = {
        "FP32": round(total_trainable * 4 / (1024 ** 2), 2),
        "FP16": round(total_trainable * 2 / (1024 ** 2), 2),
        "INT8": round(total_trainable * 1 / (1024 ** 2), 2),
        "INT4": round(total_trainable * 0.5 / (1024 ** 2), 2),
    }

    return {
        "total_trainable": total_trainable,
        "total_non_trainable": total_non_trainable,
        "embedding_params": embed_params,
        "scratchpad_params": scratch_params,
        "expert_params": expert_params,
        "active_params_per_token": active_params,
        "per_superblock": first_sb_params,
        "checkpoint_sizes_mb": checkpoint_sizes_mb,
    }
