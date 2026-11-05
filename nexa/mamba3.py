"""
Mamba-3 Sequence Modeling Layer with Trapezoidal Selective Scan.
Implements the continuous-time to discrete-time discretization via trapezoidal quadrature
as formulated in Lahoti et al. (2026) and NEXA-SLM research paper Section VI-C (Eqs. 3-4):

    h_t = α_t * h_{t-1} + β_t * (B_{t-1} ⊗ x_{t-1}) + γ_t * (B_t ⊗ x_t)
    y_t = C_t^T * h_t + D * x_t

where:
    α_t = exp(Δ_t * A)
    β_t = (1 - λ_t) * Δ_t * exp(Δ_t * A)
    γ_t = λ_t * Δ_t
    λ_t ∈ (0, 1) is an input-dependent trapezoidal mixing parameter.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def mamba3_trapezoidal_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    lam: torch.Tensor,
    A: torch.Tensor,
    B_proj: torch.Tensor,
    C_proj: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pure-PyTorch numerically exact trapezoidal selective scan for Mamba-3.

    Args:
        u: Input tensor after causal 1D conv and SiLU, shape (B, D_inner, L)
        delta: Discretization step size, shape (B, D_inner, L)
        lam: Input-dependent trapezoidal mixing factor in (0, 1), shape (B, D_inner, L)
        A: Transition diagonal matrix (< 0), shape (D_inner, N_state)
        B_proj: Input projection B_t, shape (B, N_state, L)
        C_proj: Output projection C_t, shape (B, N_state, L)
        D: Optional skip connection, shape (D_inner,)
        z: Optional gating tensor, shape (B, D_inner, L)

    Returns:
        y: Output tensor, shape (B, D_inner, L)
    """
    B_batch, D_dim, L = u.shape
    N = A.shape[1]

    # Expand tensors for discretization: (B, D, 1, L) and (1, D, N, 1)
    delta_exp = delta.unsqueeze(2)  # (B, D, 1, L)
    lam_exp = lam.unsqueeze(2)      # (B, D, 1, L)
    A_exp = A.unsqueeze(0).unsqueeze(-1)  # (1, D, N, 1)

    # Discretization coefficients (Eqs. 3-4)
    alpha = torch.exp(delta_exp * A_exp)            # (B, D, N, L)
    beta = (1.0 - lam_exp) * delta_exp * alpha      # (B, D, N, L)
    gamma = lam_exp * delta_exp                     # (B, D, 1, L)

    # Input tensor outer products: v_t = u_t ⊗ B_t -> (B, D, N, L)
    v = u.unsqueeze(2) * B_proj.unsqueeze(1)
    # v_{t-1} with zero initial boundary condition
    v_prev = torch.cat([torch.zeros(B_batch, D_dim, N, 1, device=u.device, dtype=u.dtype), v[:, :, :, :-1]], dim=-1)
    # Trapezoidal input driving term: β_t * v_{t-1} + γ_t * v_t
    tilde_v = beta * v_prev + gamma * v  # (B, D, N, L)

    # Sequential scan over sequence length L
    h = torch.zeros(B_batch, D_dim, N, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = alpha[:, :, :, t] * h + tilde_v[:, :, :, t]
        y_t = (h * C_proj[:, :, t].unsqueeze(1)).sum(dim=-1)  # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=-1)  # (B, D, L)

    if D is not None:
        y = y + u * D.unsqueeze(-1)
    if z is not None:
        y = y * F.silu(z)

    return y


def mamba1_standard_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_proj: torch.Tensor,
    C_proj: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standard Mamba-1 Euler/ZOH scan (used for ablation baseline)."""
    B_batch, D_dim, L = u.shape
    N = A.shape[1]

    delta_exp = delta.unsqueeze(2)
    A_exp = A.unsqueeze(0).unsqueeze(-1)
    alpha = torch.exp(delta_exp * A_exp)
    dB_u = delta_exp * (u.unsqueeze(2) * B_proj.unsqueeze(1))

    h = torch.zeros(B_batch, D_dim, N, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = alpha[:, :, :, t] * h + dB_u[:, :, :, t]
        y_t = (h * C_proj[:, :, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)

    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + u * D.unsqueeze(-1)
    if z is not None:
        y = y * F.silu(z)
    return y


class Mamba3Layer(nn.Module):
    """
    Mamba-3 layer with trapezoidal selective scan, 1D causal convolution,
    and state caching for single-token autoregressive decoding.
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_inner = config.mamba_d_inner
        self.d_state = config.mamba_d_state
        self.dt_rank = config.mamba_dt_rank
        self.conv_size = config.mamba_conv_size
        self.use_trapezoidal = getattr(config, "use_mamba3_trapezoidal", True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.conv_size,
            groups=self.d_inner,
            padding=self.conv_size - 1,
            bias=True,
        )

        # Projections: dt, lambda, B, C
        # For Mamba-3: dt_rank (for delta) + dt_rank (for lambda) + 2 * d_state (for B, C)
        proj_dim = self.dt_rank * 2 + self.d_state * 2 if self.use_trapezoidal else self.dt_rank + self.d_state * 2
        self.x_proj = nn.Linear(self.d_inner, proj_dim, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        if self.use_trapezoidal:
            self.lam_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full sequence forward pass.
        Args:
            x: shape (B, L, D_model)
        Returns:
            out: shape (B, L, D_model)
        """
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # 1D Causal Convolution: truncate right padding to preserve causality
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L]
        x_conv = F.silu(x_conv)  # (B, D_inner, L)

        # State space parameter projection from x_conv
        x_dbl = self.x_proj(x_conv.transpose(1, 2))  # (B, L, proj_dim)

        A = -torch.exp(self.A_log.float())

        if self.use_trapezoidal:
            dt_raw, lam_raw, B_proj, C_proj = torch.split(
                x_dbl, [self.dt_rank, self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = F.softplus(self.dt_proj(dt_raw))           # (B, L, D_inner)
            lam = torch.sigmoid(self.lam_proj(lam_raw))      # (B, L, D_inner) in (0, 1)

            y = mamba3_trapezoidal_scan(
                u=x_conv,
                delta=dt.transpose(1, 2).to(x.dtype),
                lam=lam.transpose(1, 2).to(x.dtype),
                A=A.to(x.dtype),
                B_proj=B_proj.transpose(1, 2),
                C_proj=C_proj.transpose(1, 2),
                D=self.D.to(x.dtype),
                z=None,
            )
        else:
            dt_raw, B_proj, C_proj = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = F.softplus(self.dt_proj(dt_raw))
            y = mamba1_standard_scan(
                u=x_conv,
                delta=dt.transpose(1, 2).to(x.dtype),
                A=A.to(x.dtype),
                B_proj=B_proj.transpose(1, 2),
                C_proj=C_proj.transpose(1, 2),
                D=self.D.to(x.dtype),
                z=None,
            )

        y = y.transpose(1, 2)  # (B, L, D_inner)
        return self.out_proj(y * F.silu(z))

    def step(
        self,
        x: torch.Tensor,
        cache: Dict,
        block_idx: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Single-token autoregressive decoding step with cached conv and SSM state.
        Guaranteed to match the full-sequence scan numerically.

        Args:
            x: Current token tensor, shape (B, 1, D_model)
            cache: State dictionary
            block_idx: Index of SuperBlock
            layer_idx: Index of sub-layer within SuperBlock

        Returns:
            out: (B, 1, D_model)
            cache: Updated state dictionary
        """
        B, L, D = x.shape
        conv_key = f"mamba_{block_idx}_{layer_idx}_conv"
        ssm_key = f"mamba_{block_idx}_{layer_idx}_ssm"
        prev_u_key = f"mamba_{block_idx}_{layer_idx}_prev_u"
        prev_B_key = f"mamba_{block_idx}_{layer_idx}_prev_B"

        conv_state = cache.get(conv_key)
        ssm_state = cache.get(ssm_key)
        prev_u = cache.get(prev_u_key)
        prev_B = cache.get(prev_B_key)

        if conv_state is None:
            conv_state = torch.zeros(
                B, self.d_inner, self.conv_size - 1, device=x.device, dtype=x.dtype
            )
            ssm_state = torch.zeros(
                B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype
            )
            prev_u = torch.zeros(B, self.d_inner, device=x.device, dtype=x.dtype)
            prev_B = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)

        # In-projection
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # Update 1D Conv state and apply conv
        conv_state = torch.cat([conv_state, x_in.transpose(1, 2)], dim=-1)
        x_conv = F.conv1d(
            conv_state,
            self.conv1d.weight,
            bias=self.conv1d.bias,
            groups=self.conv1d.groups,
            padding=0,
        ).squeeze(-1)  # (B, D_inner)
        x_conv = F.silu(x_conv)

        # State space parameter projections
        x_dbl = self.x_proj(x_conv.unsqueeze(1))  # (B, 1, proj_dim)
        A = -torch.exp(self.A_log.to(x.dtype))

        if self.use_trapezoidal:
            dt_raw, lam_raw, B_proj, C_proj = torch.split(
                x_dbl, [self.dt_rank, self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = F.softplus(self.dt_proj(dt_raw)).squeeze(1)     # (B, D_inner)
            lam = torch.sigmoid(self.lam_proj(lam_raw)).squeeze(1) # (B, D_inner)
            B_curr = B_proj.squeeze(1)                             # (B, N_state)
            C_curr = C_proj.squeeze(1)                             # (B, N_state)

            # Discretization coefficients for step
            alpha_t = torch.exp(dt.unsqueeze(-1) * A)              # (B, D_inner, N_state)
            beta_t = (1.0 - lam).unsqueeze(-1) * dt.unsqueeze(-1) * alpha_t
            gamma_t = lam.unsqueeze(-1) * dt.unsqueeze(-1)

            # Trapezoidal driving term
            v_prev = prev_u.unsqueeze(-1) * prev_B.unsqueeze(1)    # (B, D_inner, N_state)
            v_curr = x_conv.unsqueeze(-1) * B_curr.unsqueeze(1)    # (B, D_inner, N_state)
            tilde_v = beta_t * v_prev + gamma_t * v_curr

            ssm_state = alpha_t * ssm_state + tilde_v
            y = (ssm_state * C_curr.unsqueeze(1)).sum(dim=-1) + self.D.to(x.dtype) * x_conv

            cache[prev_u_key] = x_conv.detach()
            cache[prev_B_key] = B_curr.detach()
        else:
            dt_raw, B_proj, C_proj = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = F.softplus(self.dt_proj(dt_raw)).squeeze(1)
            B_curr = B_proj.squeeze(1)
            C_curr = C_proj.squeeze(1)

            alpha_t = torch.exp(dt.unsqueeze(-1) * A)
            dB_u = dt.unsqueeze(-1) * (x_conv.unsqueeze(-1) * B_curr.unsqueeze(1))
            ssm_state = alpha_t * ssm_state + dB_u
            y = (ssm_state * C_curr.unsqueeze(1)).sum(dim=-1) + self.D.to(x.dtype) * x_conv

        y = y * F.silu(z.squeeze(1))
        out = self.out_proj(y.unsqueeze(1))  # (B, 1, D_model)

        # Maintain rolling conv window
        conv_state = conv_state[:, :, -(self.conv_size - 1):]
        cache[conv_key] = conv_state.detach()
        cache[ssm_key] = ssm_state.detach()

        return out, cache
