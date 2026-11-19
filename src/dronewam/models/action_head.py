"""Minimal action decoding from predicted future latent tokens."""

from __future__ import annotations

import torch


class MeanPoolActionHead(torch.nn.Sequential):
    """Spatial mean pooling followed by the legacy checkpoint-compatible MLP."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, future_tokens: torch.Tensor) -> torch.Tensor:
        if future_tokens.ndim != 4:
            raise ValueError(f"future_tokens must be [B,T,P,D], got {tuple(future_tokens.shape)}")
        return self.decode_pooled(future_tokens.mean(dim=2))

    def decode_pooled(self, pooled_tokens: torch.Tensor) -> torch.Tensor:
        """Decode already pooled tokens while preserving legacy parameter keys."""
        if pooled_tokens.ndim != 3:
            raise ValueError(f"pooled_tokens must be [B,T,D], got {tuple(pooled_tokens.shape)}")
        return super().forward(pooled_tokens)
