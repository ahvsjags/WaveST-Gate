"""Cross-modal reliability gating."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalReliabilityGate(nn.Module):
    """Learn soft reliability weights for image, expression, and reference agents."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.fuse = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        image_feature: torch.Tensor,
        expression_feature: torch.Tensor,
        agent_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.scorer(torch.cat([image_feature, expression_feature, agent_feature], dim=-1))
        weights = logits.softmax(dim=-1)
        stacked = torch.stack([image_feature, expression_feature, agent_feature], dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return self.fuse(fused), weights


class ModalityUncertaintyHead(nn.Module):
    """Estimate spot-level and modality-level uncertainty for reliability fusion."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim * 4 + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.spot_head = nn.Linear(hidden_dim, 1)
        self.modality_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        image_feature: torch.Tensor,
        expression_feature: torch.Tensor,
        agent_feature: torch.Tensor,
        fused_feature: torch.Tensor,
        gate_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat(
            [image_feature, expression_feature, agent_feature, fused_feature, gate_weights],
            dim=-1,
        )
        hidden = self.encoder(features)
        spot_uncertainty = F.softplus(self.spot_head(hidden)) + 1e-6
        modality_uncertainty = torch.sigmoid(self.modality_head(hidden))
        return spot_uncertainty, modality_uncertainty


def calibrate_gate_weights(
    gate_weights: torch.Tensor,
    modality_uncertainty: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Down-weight modalities whose estimated uncertainty is high."""

    confidence = (1.0 - modality_uncertainty).clamp_min(eps)
    logits = gate_weights.clamp_min(eps).log() + confidence.log()
    return logits.softmax(dim=-1)
