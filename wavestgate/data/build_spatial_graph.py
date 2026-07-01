"""Spatial graph construction."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(
    coords: torch.Tensor | np.ndarray,
    k: int = 6,
    include_self: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a directed kNN graph from `[N, 2]` coordinates.

    Returns `(edge_index, edge_weight)`, where weights are inverse-distance
    similarities normalized to `[0, 1]`.
    """

    coord_np = np.asarray(coords, dtype=np.float32)
    if coord_np.ndim != 2 or coord_np.shape[1] != 2:
        raise ValueError("coords must have shape [N, 2]")
    if len(coord_np) == 0:
        raise ValueError("coords cannot be empty")
    n_neighbors = min(k + (0 if include_self else 1), len(coord_np))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coord_np)
    distances, indices = nbrs.kneighbors(coord_np)

    src_edges = []
    dst_edges = []
    weights = []
    scale = float(np.median(distances[:, -1])) if distances.size else 1.0
    scale = max(scale, 1e-6)
    for src, (row_dist, row_idx) in enumerate(zip(distances, indices)):
        for dist, dst in zip(row_dist, row_idx):
            if not include_self and src == int(dst):
                continue
            src_edges.append(src)
            dst_edges.append(int(dst))
            weights.append(float(np.exp(-dist / scale)))

    edge_index = torch.as_tensor([src_edges, dst_edges], dtype=torch.long)
    edge_weight = torch.as_tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def build_radius_graph(
    coords: torch.Tensor | np.ndarray,
    radius: float,
    include_self: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a directed radius graph from `[N, 2]` coordinates."""

    coord_np = np.asarray(coords, dtype=np.float32)
    if coord_np.ndim != 2 or coord_np.shape[1] != 2:
        raise ValueError("coords must have shape [N, 2]")
    nbrs = NearestNeighbors(radius=radius).fit(coord_np)
    distances, indices = nbrs.radius_neighbors(coord_np)

    src_edges = []
    dst_edges = []
    weights = []
    radius = max(float(radius), 1e-6)
    for src, (row_dist, row_idx) in enumerate(zip(distances, indices)):
        for dist, dst in zip(row_dist, row_idx):
            if not include_self and src == int(dst):
                continue
            src_edges.append(src)
            dst_edges.append(int(dst))
            weights.append(float(np.exp(-dist / radius)))
    return torch.as_tensor([src_edges, dst_edges], dtype=torch.long), torch.as_tensor(weights, dtype=torch.float32)
