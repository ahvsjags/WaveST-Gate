"""Synthetic data generation for MVP tests and demos."""

from __future__ import annotations

import torch

from wavestgate.models.types import WaveSTGateBatch, WaveSTGateConfig


def make_synthetic_batch(
    config: WaveSTGateConfig,
    batch_size: int = 8,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> WaveSTGateBatch:
    """Create a small correlated H&E/ST/scRNA batch for smoke testing."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    cell_types = config.num_cell_types
    genes = config.num_genes
    patch_size = config.patch_size

    reference = torch.rand(cell_types, genes, generator=generator) * 2.0
    marker_width = max(genes // cell_types, 1)
    for idx in range(cell_types):
        start = idx * marker_width
        end = min(start + marker_width, genes)
        reference[idx, start:end] += 2.0

    raw_props = torch.rand(batch_size, cell_types, generator=generator) + 0.2
    proportions = raw_props / raw_props.sum(dim=1, keepdim=True)
    expression = proportions @ reference
    expression = (expression + 0.05 * torch.randn(batch_size, genes, generator=generator)).clamp_min(0.0)

    colors = torch.rand(cell_types, 3, generator=generator)
    base_colors = proportions @ colors
    yy = torch.linspace(0, 1, patch_size).view(1, 1, patch_size, 1)
    xx = torch.linspace(0, 1, patch_size).view(1, 1, 1, patch_size)
    gradient = 0.15 * torch.sin(2 * torch.pi * (xx + yy))
    patches = base_colors[:, :, None, None].expand(batch_size, 3, patch_size, patch_size).clone()
    patches = patches + gradient + 0.05 * torch.randn(batch_size, 3, patch_size, patch_size, generator=generator)
    patches = patches.clamp(0.0, 1.0)

    coords = torch.stack(
        [
            torch.arange(batch_size, dtype=torch.float32),
            torch.zeros(batch_size, dtype=torch.float32),
        ],
        dim=1,
    )
    if batch_size > 1:
        src = torch.arange(batch_size - 1)
        dst = src + 1
        edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
        edge_weight = torch.ones(edge_index.size(1))
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_weight = torch.empty(0)

    return WaveSTGateBatch(
        image_patches=patches.to(device),
        st_expression=expression.to(device),
        reference_prototypes=reference.to(device),
        coords=coords.to(device),
        edge_index=edge_index.to(device),
        edge_weight=edge_weight.to(device),
        proportion_gt=proportions.to(device),
    )
