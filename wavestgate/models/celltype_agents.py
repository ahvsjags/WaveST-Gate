"""scRNA-initialized cell-type prototype agents."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CellTypePrototypeAgents(nn.Module):
    """Map reference prototypes to agent tokens and attend spot features to them."""

    def __init__(self, num_genes: int, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.prototype_mlp = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.output = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.scale = latent_dim**-0.5

    def forward(self, spot_feature: torch.Tensor, reference_prototypes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        agents = self.prototype_mlp(reference_prototypes)
        q = self.query(spot_feature)
        k = self.key(agents)
        v = self.value(agents)
        logits = (q @ k.transpose(0, 1)) * self.scale
        attention = logits.softmax(dim=-1)
        agent_feature = attention @ v
        agent_feature = self.output(agent_feature) + agent_feature
        return agent_feature, attention, agents
