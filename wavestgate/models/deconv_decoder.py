"""Deconvolution decoder heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeconvolutionDecoder(nn.Module):
    """Decode fused spot features into proportions and reconstructed expression."""

    def __init__(self, latent_dim: int, num_cell_types: int, num_genes: int, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.proportion_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_cell_types),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, num_genes),
        )

    def forward(self, fused_feature: torch.Tensor, reference_prototypes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.proportion_head(fused_feature)
        proportions = logits.softmax(dim=-1)
        base_expression = proportions @ reference_prototypes
        residual = 0.1 * torch.tanh(self.residual_head(fused_feature))
        reconstructed = F.relu(base_expression + residual)
        return proportions, reconstructed


class NicheHead(nn.Module):
    def __init__(self, latent_dim: int, niche_classes: int, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, niche_classes),
        )

    def forward(self, fused_feature: torch.Tensor) -> torch.Tensor:
        return self.net(fused_feature)
