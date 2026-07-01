"""Typed containers used by the WaveST-Gate model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class WaveSTGateConfig:
    """Configuration for WaveST-Gate."""

    num_genes: int
    num_cell_types: int
    latent_dim: int = 64
    hidden_dim: int = 128
    image_channels: int = 3
    patch_size: int = 256
    niche_classes: int = 0
    dropout: float = 0.0
    loss_expr_weight: float = 1.0
    loss_prop_weight: float = 0.0
    loss_sparse_weight: float = 0.0
    loss_spatial_weight: float = 0.0
    loss_contrast_weight: float = 0.0
    loss_uncertainty_weight: float = 0.0
    loss_boundary_weight: float = 0.0
    loss_niche_weight: float = 0.0
    loss_image_gate_weight: float = 0.0
    boundary_temperature: float = 0.5
    target_image_gate: float = 0.0
    target_boundary_image_gate: float = 0.0
    target_raw_image_gate: float = 0.0
    raw_image_gate_loss_scale: float = 0.5
    image_encoder_type: str = "wavelet"
    use_celltype_agents: bool = True
    use_cross_modal_gate: bool = True
    use_uncertainty_calibration: bool = True
    use_local_refinement: bool = True
    use_expression_branch: bool = True
    modality_dropout_prob: float = 0.0

    @classmethod
    def from_dict(cls, values: dict) -> "WaveSTGateConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in values.items() if k in allowed})


@dataclass
class WaveSTGateBatch:
    """Mini-batch expected by `WaveSTGate.forward`."""

    image_patches: torch.Tensor
    st_expression: torch.Tensor
    reference_prototypes: torch.Tensor
    coords: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    edge_weight: Optional[torch.Tensor] = None
    proportion_gt: Optional[torch.Tensor] = None
    niche_gt: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "WaveSTGateBatch":
        def move(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return x.to(device) if x is not None else None

        return WaveSTGateBatch(
            image_patches=self.image_patches.to(device),
            st_expression=self.st_expression.to(device),
            reference_prototypes=self.reference_prototypes.to(device),
            coords=move(self.coords),
            edge_index=move(self.edge_index),
            edge_weight=move(self.edge_weight),
            proportion_gt=move(self.proportion_gt),
            niche_gt=move(self.niche_gt),
        )


@dataclass
class WaveSTGateOutput:
    """Model outputs used by training, evaluation, and interpretation."""

    proportions: torch.Tensor
    reconstructed_expression: torch.Tensor
    gate_weights: torch.Tensor
    agent_attention: torch.Tensor
    image_feature: torch.Tensor
    expression_feature: torch.Tensor
    agent_feature: torch.Tensor
    fused_feature: torch.Tensor
    niche_logits: Optional[torch.Tensor] = None
    raw_gate_weights: Optional[torch.Tensor] = None
    spot_uncertainty: Optional[torch.Tensor] = None
    modality_uncertainty: Optional[torch.Tensor] = None
    modality_reliability: Optional[torch.Tensor] = None
    morphology_feature: Optional[torch.Tensor] = None
