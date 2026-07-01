"""Gate-weight interpretation helpers."""

from __future__ import annotations

import torch


GATE_COLUMNS = ["image", "expression", "reference"]


def gate_weight_table(gate_weights: torch.Tensor) -> list[dict[str, float]]:
    """Convert `[B, 3]` gate weights to a serializable table."""

    if gate_weights.ndim != 2 or gate_weights.size(1) != 3:
        raise ValueError("gate_weights must have shape [B, 3]")
    rows = []
    for row in gate_weights.detach().cpu():
        rows.append({name: float(value) for name, value in zip(GATE_COLUMNS, row)})
    return rows


def reliability_uncertainty_table(
    spot_uncertainty: torch.Tensor,
    modality_uncertainty: torch.Tensor,
) -> list[dict[str, float]]:
    """Convert uncertainty tensors to a serializable per-spot table."""

    if spot_uncertainty.ndim not in {1, 2}:
        raise ValueError("spot_uncertainty must have shape [B] or [B, 1]")
    if modality_uncertainty.ndim != 2 or modality_uncertainty.size(1) != 3:
        raise ValueError("modality_uncertainty must have shape [B, 3]")
    spot_values = spot_uncertainty.reshape(-1).detach().cpu()
    modality_values = modality_uncertainty.detach().cpu()
    if spot_values.size(0) != modality_values.size(0):
        raise ValueError("spot_uncertainty and modality_uncertainty must contain the same number of spots")
    rows = []
    for spot_value, row in zip(spot_values, modality_values):
        item = {"spot_uncertainty": float(spot_value)}
        for name, value in zip(GATE_COLUMNS, row):
            item[f"{name}_uncertainty"] = float(value)
        rows.append(item)
    return rows
