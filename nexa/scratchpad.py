"""
Learnable Virtual Scratchpad module for NEXA-SLM.
Manages persistent global memory vectors P ∈ R^(m x d) prepended to the token stream.
"""

from typing import Tuple
import torch
import torch.nn as nn


class VirtualScratchpad(nn.Module):
    """
    Maintains m learnable prefix tokens that provide persistent global context
    accessible to local attention without inflating the Key-Value cache.
    """

    def __init__(self, num_tokens: int, d_model: int, std: float = 0.02):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        if num_tokens > 0:
            self.scratchpad = nn.Parameter(torch.randn(num_tokens, d_model) * std)
        else:
            self.register_parameter("scratchpad", None)

    def prepend(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Prepends scratchpad tokens to the input token embeddings:
            Z_0 = [P ; T]
        Args:
            token_embeddings: (B, L, D)
        Returns:
            concatenated: (B, m + L, D)
        """
        if self.scratchpad is None or self.num_tokens == 0:
            return token_embeddings
        B = token_embeddings.size(0)
        prefix = self.scratchpad.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([prefix, token_embeddings], dim=1)

    def extract_sequence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Extracts the sequence representations (excluding scratchpad positions)
        for computing language model logits and cross-entropy loss.
        Args:
            hidden_states: (B, m + L, D)
        Returns:
            sequence_states: (B, L, D)
        """
        if self.scratchpad is None or self.num_tokens == 0:
            return hidden_states
        return hidden_states[:, self.num_tokens :, :]

    def get_prefix_tokens(self) -> torch.Tensor:
        """Returns the raw scratchpad parameters for cached prefill in inference."""
        return self.scratchpad
