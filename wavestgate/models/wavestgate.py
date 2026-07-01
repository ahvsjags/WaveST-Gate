"""WaveST-Gate model assembly."""

from __future__ import annotations

import torch
import torch.nn as nn

from wavestgate.models.celltype_agents import CellTypePrototypeAgents
from wavestgate.models.cross_modal_gate import CrossModalReliabilityGate, ModalityUncertaintyHead, calibrate_gate_weights
from wavestgate.models.deconv_decoder import DeconvolutionDecoder, NicheHead
from wavestgate.models.expression_encoder import ExpressionEncoder
from wavestgate.models.local_refinement import LocalCrossModalRefinement
from wavestgate.models.types import WaveSTGateBatch, WaveSTGateConfig, WaveSTGateOutput
from wavestgate.models.wavelet_encoder import WaveletMorphologyEncoder


class SimpleCNNMorphologyEncoder(nn.Module):
    """Small CNN image encoder used as an ablation replacement for wavelets."""

    def __init__(self, in_channels: int, latent_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor, return_map: bool = False):
        fmap = self.net(x)
        token = fmap.mean(dim=(2, 3))
        return (token, fmap) if return_map else token


class WaveSTGate(nn.Module):
    """MVP implementation of morphology-aware ST deconvolution."""

    def __init__(self, config: WaveSTGateConfig) -> None:
        super().__init__()
        self.config = config
        if config.image_encoder_type == "wavelet":
            self.image_encoder = WaveletMorphologyEncoder(
                in_channels=config.image_channels,
                latent_dim=config.latent_dim,
                dropout=config.dropout,
            )
        elif config.image_encoder_type == "cnn":
            self.image_encoder = SimpleCNNMorphologyEncoder(
                in_channels=config.image_channels,
                latent_dim=config.latent_dim,
                dropout=config.dropout,
            )
        elif config.image_encoder_type == "none":
            self.image_encoder = None
        else:
            raise ValueError("image_encoder_type must be one of: wavelet, cnn, none")
        self.expression_encoder = ExpressionEncoder(
            num_genes=config.num_genes,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.local_refinement = LocalCrossModalRefinement(config.latent_dim)
        self.agents = CellTypePrototypeAgents(
            num_genes=config.num_genes,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.gate = CrossModalReliabilityGate(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.uncertainty_head = ModalityUncertaintyHead(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.decoder = DeconvolutionDecoder(
            latent_dim=config.latent_dim,
            num_cell_types=config.num_cell_types,
            num_genes=config.num_genes,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.niche_head = (
            NicheHead(config.latent_dim, config.niche_classes, config.hidden_dim, config.dropout)
            if config.niche_classes and config.niche_classes > 0
            else None
        )

    def forward(self, batch: WaveSTGateBatch) -> WaveSTGateOutput:
        self._validate_batch(batch)
        if self.image_encoder is None:
            bsz, _, height, width = batch.image_patches.shape
            morphology_feature = batch.image_patches.new_zeros(bsz, self.config.latent_dim)
            image_map = batch.image_patches.new_zeros(bsz, self.config.latent_dim, height, width)
        else:
            morphology_feature, image_map = self.image_encoder(batch.image_patches, return_map=True)
        if self.config.use_expression_branch:
            expression_feature = self.expression_encoder(batch.st_expression)
        else:
            expression_feature = batch.st_expression.new_zeros(batch.st_expression.size(0), self.config.latent_dim)

        if self.config.use_local_refinement:
            expression_map = expression_feature[:, :, None, None].expand(-1, -1, image_map.size(-2), image_map.size(-1))
            refined_map = self.local_refinement(image_map, expression_map)
            image_feature = refined_map.mean(dim=(2, 3)) + morphology_feature
        else:
            image_feature = morphology_feature

        agent_query = image_feature + expression_feature
        if self.config.use_celltype_agents:
            agent_feature, agent_attention, _ = self.agents(agent_query, batch.reference_prototypes)
        else:
            agent_feature = torch.zeros_like(expression_feature)
            agent_attention = expression_feature.new_full(
                (expression_feature.size(0), self.config.num_cell_types),
                1.0 / self.config.num_cell_types,
            )
        image_gate_feature = image_feature
        expression_gate_feature = expression_feature
        agent_gate_feature = agent_feature
        if self.training and self.config.modality_dropout_prob > 0:
            keep = torch.rand(
                image_feature.size(0),
                3,
                device=image_feature.device,
                dtype=image_feature.dtype,
            ) > self.config.modality_dropout_prob
            empty = keep.sum(dim=1) == 0
            if empty.any():
                keep[empty, torch.randint(0, 3, (int(empty.sum().item()),), device=image_feature.device)] = True
            image_gate_feature = image_gate_feature * keep[:, 0:1]
            expression_gate_feature = expression_gate_feature * keep[:, 1:2]
            agent_gate_feature = agent_gate_feature * keep[:, 2:3]
        if self.config.use_cross_modal_gate:
            base_fused_feature, raw_gate_weights = self.gate(image_gate_feature, expression_gate_feature, agent_gate_feature)
        else:
            raw_gate_weights = expression_feature.new_full((expression_feature.size(0), 3), 1.0 / 3.0)
            stacked_base = torch.stack([image_gate_feature, expression_gate_feature, agent_gate_feature], dim=1)
            base_fused_feature = self.gate.fuse((raw_gate_weights.unsqueeze(-1) * stacked_base).sum(dim=1))
        spot_uncertainty, modality_uncertainty = self.uncertainty_head(
            image_gate_feature,
            expression_gate_feature,
            agent_gate_feature,
            base_fused_feature,
            raw_gate_weights,
        )
        gate_weights = (
            calibrate_gate_weights(raw_gate_weights, modality_uncertainty)
            if self.config.use_uncertainty_calibration
            else raw_gate_weights
        )
        stacked_features = torch.stack([image_gate_feature, expression_gate_feature, agent_gate_feature], dim=1)
        fused_feature = self.gate.fuse((gate_weights.unsqueeze(-1) * stacked_features).sum(dim=1))
        proportions, reconstructed_expression = self.decoder(fused_feature, batch.reference_prototypes)
        niche_logits = self.niche_head(fused_feature) if self.niche_head is not None else None

        return WaveSTGateOutput(
            proportions=proportions,
            reconstructed_expression=reconstructed_expression,
            gate_weights=gate_weights,
            agent_attention=agent_attention,
            image_feature=image_feature,
            expression_feature=expression_feature,
            agent_feature=agent_feature,
            fused_feature=fused_feature,
            niche_logits=niche_logits,
            raw_gate_weights=raw_gate_weights,
            spot_uncertainty=spot_uncertainty,
            modality_uncertainty=modality_uncertainty,
            modality_reliability=1.0 - modality_uncertainty,
            morphology_feature=morphology_feature,
        )

    def _validate_batch(self, batch: WaveSTGateBatch) -> None:
        if batch.image_patches.ndim != 4:
            raise ValueError("image_patches must have shape [B, C, H, W]")
        if batch.st_expression.ndim != 2:
            raise ValueError("st_expression must have shape [B, G]")
        if batch.reference_prototypes.ndim != 2:
            raise ValueError("reference_prototypes must have shape [C, G]")
        if batch.st_expression.size(1) != self.config.num_genes:
            raise ValueError("st_expression gene dimension does not match config.num_genes")
        if batch.reference_prototypes.shape != (self.config.num_cell_types, self.config.num_genes):
            raise ValueError("reference_prototypes must match [num_cell_types, num_genes]")
        if batch.image_patches.size(1) != self.config.image_channels:
            raise ValueError("image channel dimension does not match config.image_channels")
