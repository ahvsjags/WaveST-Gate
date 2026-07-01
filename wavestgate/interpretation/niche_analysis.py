"""Simple niche-analysis helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wavestgate.interpretation.visualize_gates import GATE_COLUMNS


def niche_labels_from_logits(niche_logits: torch.Tensor) -> torch.Tensor:
    """Convert niche logits to hard labels."""

    if niche_logits.ndim != 2:
        raise ValueError("niche_logits must have shape [B, K]")
    return niche_logits.argmax(dim=1)


def niche_probabilities_from_logits(niche_logits: torch.Tensor) -> torch.Tensor:
    """Convert niche logits to probabilities."""

    if niche_logits.ndim != 2:
        raise ValueError("niche_logits must have shape [B, K]")
    return F.softmax(niche_logits, dim=1)


def summarize_niche_composition(
    proportions: torch.Tensor,
    niche_logits: torch.Tensor,
    cell_types: list[str] | None = None,
) -> list[dict[str, float | int | str]]:
    """Summarize mean cell-type composition within each predicted niche."""

    if proportions.ndim != 2:
        raise ValueError("proportions must have shape [B, C]")
    labels = niche_labels_from_logits(niche_logits)
    if proportions.size(0) != labels.size(0):
        raise ValueError("proportions and niche_logits must contain the same number of spots")
    if cell_types is None:
        cell_types = [f"cell_type_{idx}" for idx in range(proportions.size(1))]
    if len(cell_types) != proportions.size(1):
        raise ValueError("cell_types length must match the proportion dimension")

    rows: list[dict[str, float | int | str]] = []
    for niche_idx in range(niche_logits.size(1)):
        mask = labels == niche_idx
        if mask.any():
            mean_props = proportions[mask].mean(dim=0)
            dominant_idx = int(mean_props.argmax().item())
            row: dict[str, float | int | str] = {
                "niche": int(niche_idx),
                "num_spots": int(mask.sum().item()),
                "dominant_cell_type": cell_types[dominant_idx],
                "dominant_cell_type_fraction": float(mean_props[dominant_idx].item()),
            }
            for name, value in zip(cell_types, mean_props):
                row[f"mean_{name}"] = float(value.item())
        else:
            row = {
                "niche": int(niche_idx),
                "num_spots": 0,
                "dominant_cell_type": "",
                "dominant_cell_type_fraction": 0.0,
            }
            for name in cell_types:
                row[f"mean_{name}"] = 0.0
        rows.append(row)
    return rows


def summarize_gate_reliability_by_niche(
    gate_weights: torch.Tensor,
    niche_logits: torch.Tensor,
    gate_names: list[str] | None = None,
) -> list[dict[str, float | int]]:
    """Summarize mean modality reliability inside each predicted niche."""

    if gate_weights.ndim != 2 or gate_weights.size(1) != 3:
        raise ValueError("gate_weights must have shape [B, 3]")
    labels = niche_labels_from_logits(niche_logits)
    if gate_weights.size(0) != labels.size(0):
        raise ValueError("gate_weights and niche_logits must contain the same number of spots")
    names = gate_names or GATE_COLUMNS
    rows: list[dict[str, float | int]] = []
    for niche_idx in range(niche_logits.size(1)):
        mask = labels == niche_idx
        row: dict[str, float | int] = {"niche": int(niche_idx), "num_spots": int(mask.sum().item())}
        means = gate_weights[mask].mean(dim=0) if mask.any() else gate_weights.new_zeros(3)
        for name, value in zip(names, means):
            row[f"mean_{name}_reliability"] = float(value.item())
        rows.append(row)
    return rows
