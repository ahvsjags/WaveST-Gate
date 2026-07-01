"""Marker enrichment utilities."""

from __future__ import annotations

import torch


def marker_scores(expression: torch.Tensor, marker_indices: list[int]) -> torch.Tensor:
    """Average marker-gene expression per spot."""

    if not marker_indices:
        raise ValueError("marker_indices cannot be empty")
    return expression[:, marker_indices].mean(dim=1)
