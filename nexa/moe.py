"""
Sparse Mixture-of-Experts (MoE) Layer with Top-2 Routing and Shared Expert.
Implements the architecture specified in NEXA-SLM research paper Section VI-C (Eqs. 7-10):

    g(x) = softmax(W_r * x)
    MoE(x) = E_s(x) + Σ_{i ∈ Top-2} g_i(x) * E_i(x)
    L_aux = N * Σ_{i=1}^N f_i * P_i
    L_z = (1 / B) * Σ (log Σ exp(z_i))^2
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network:
    FFN(x) = W3( SiLU(W1(x)) * W2(x) )
    """

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MoELayer(nn.Module):
    """
    MoE Layer with N routed experts, Top-K selection (default 2),
    one dedicated shared expert, load-balancing auxiliary loss,
    and router z-loss.
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.num_active = config.num_active_experts
        self.moe_ffn_dim = config.moe_ffn_dim
        self.normalize_topk_prob = getattr(config, "normalize_topk_prob", True)

        self.router = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(self.d_model, self.moe_ffn_dim) for _ in range(self.num_experts)]
        )
        self.shared_expert = SwiGLU(self.d_model, self.moe_ffn_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Forward pass for MoE layer.
        Args:
            x: Input tensor, shape (B, L, D_model)
        Returns:
            out: Combined output tensor, shape (B, L, D_model)
            z_loss: Router z-loss scalar tensor
            aux_loss: Load balancing auxiliary loss scalar tensor
            metrics: Dictionary of expert utilization statistics
        """
        B, L, D = x.shape
        flat_x = x.view(-1, D)
        total_tokens = flat_x.size(0)

        # 1. Shared expert is always active for every token
        shared_out = self.shared_expert(x)

        # 2. Router logits and probabilities
        logits = self.router(x)  # (B, L, num_experts)
        flat_logits = logits.view(-1, self.num_experts)
        routing_weights = F.softmax(flat_logits, dim=-1)  # (total_tokens, num_experts)

        # 3. Top-K routing
        top_weights, top_indices = torch.topk(routing_weights, self.num_active, dim=-1)
        if self.normalize_topk_prob:
            top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # 4. Expert computation with token dispatch
        flat_out = torch.zeros_like(flat_x)
        expert_mask_oh = F.one_hot(top_indices, num_classes=self.num_experts).float()
        # Shape: (total_tokens, num_active, num_experts)

        for i in range(self.num_experts):
            # Check which tokens routed to expert i
            expert_mask = (top_indices == i)
            if expert_mask.any():
                token_idx, active_pos = torch.where(expert_mask)
                expert_inputs = flat_x[token_idx]
                expert_outputs = self.experts[i](expert_inputs)
                weights_i = top_weights[token_idx, active_pos].unsqueeze(-1)
                flat_out[token_idx] += expert_outputs * weights_i

        out = flat_out.view(B, L, D) + shared_out

        # 5. Router z-loss (ST-MoE): (1/B) * Σ (log Σ exp(z_i))^2
        logsumexp_z = torch.logsumexp(flat_logits, dim=-1)
        z_loss = torch.mean(logsumexp_z ** 2)

        # 6. Load-balancing auxiliary loss: L_aux = N * Σ (f_i * P_i)
        # P_i: average routing probability for expert i
        P_i = routing_weights.mean(dim=0)  # (num_experts,)
        # f_i: fraction of token dispatch to expert i (normalized so Σ f_i = 1)
        dispatch_count = expert_mask_oh.sum(dim=(0, 1))  # tokens sent to each expert
        f_i = dispatch_count / (total_tokens * self.num_active + 1e-9)
        aux_loss = self.num_experts * torch.sum(f_i * P_i)

        # 7. Expert monitoring metrics
        with torch.no_grad():
            f_cpu = f_i.detach()
            load_std = f_cpu.std().item()
            load_mean = f_cpu.mean().item() + 1e-9
            cv = load_std / load_mean  # Coefficient of variation (0 = perfectly balanced)
            # Shannon entropy of routing distribution
            p_clean = P_i.clamp(min=1e-9)
            entropy = -(p_clean * torch.log(p_clean)).sum().item()

            metrics = {
                "expert_load_cv": cv,
                "router_entropy": entropy,
                "max_expert_share": f_cpu.max().item(),
                "min_expert_share": f_cpu.min().item(),
            }

        return out, z_loss, aux_loss, metrics
