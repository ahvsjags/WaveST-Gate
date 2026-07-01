"""Cell-type agent interpretation helpers."""

from __future__ import annotations

import torch


def top_agent_indices(agent_attention: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Return top-k cell-type agent indices per spot."""

    if agent_attention.ndim != 2:
        raise ValueError("agent_attention must have shape [B, C]")
    k = min(k, agent_attention.size(1))
    return agent_attention.topk(k, dim=1).indices
