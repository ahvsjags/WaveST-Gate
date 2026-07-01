"""ST expression encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class ExpressionEncoder(nn.Module):
    """MLP encoder for log-normalized spot expression."""

    def __init__(self, num_genes: int, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.net(expression)
