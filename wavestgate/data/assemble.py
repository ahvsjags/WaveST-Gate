"""Assemble preprocessed inputs into `WaveSTGateBatch`."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from wavestgate.data.build_spatial_graph import build_knn_graph
from wavestgate.data.preprocess_scrna import ReferencePrototypes, align_reference_to_genes
from wavestgate.data.preprocess_st import SpotExpressionTable
from wavestgate.models.types import WaveSTGateBatch


@dataclass
class BatchMetadata:
    spot_ids: list[str]
    gene_names: list[str]
    cell_types: list[str]


def _align_proportion_table(
    proportions: pd.DataFrame | torch.Tensor | None,
    spot_ids: list[str],
    cell_types: list[str],
) -> torch.Tensor | None:
    if proportions is None:
        return None
    if isinstance(proportions, torch.Tensor):
        return proportions.float()
    missing_spots = [spot_id for spot_id in spot_ids if spot_id not in proportions.index]
    if missing_spots:
        raise ValueError(f"Proportion table is missing spot ids: {missing_spots[:5]}")
    aligned = proportions.reindex(index=spot_ids, columns=cell_types, fill_value=0.0)
    return torch.as_tensor(aligned.values, dtype=torch.float32)


def _align_niche_labels(
    niche_labels: pd.Series | torch.Tensor | None,
    spot_ids: list[str],
) -> torch.Tensor | None:
    if niche_labels is None:
        return None
    if isinstance(niche_labels, torch.Tensor):
        return niche_labels.long()
    missing_spots = [spot_id for spot_id in spot_ids if spot_id not in niche_labels.index]
    if missing_spots:
        raise ValueError(f"Niche label table is missing spot ids: {missing_spots[:5]}")
    aligned = niche_labels.reindex(index=spot_ids)
    return torch.as_tensor(aligned.values, dtype=torch.long)


def assemble_wavestgate_batch(
    spots: SpotExpressionTable,
    reference: ReferencePrototypes,
    image_patches: torch.Tensor,
    proportions: pd.DataFrame | torch.Tensor | None = None,
    niche_labels: pd.Series | torch.Tensor | None = None,
    graph_k: int | None = 6,
) -> tuple[WaveSTGateBatch, BatchMetadata]:
    """Align genes/cell types and assemble a model-ready batch."""

    if image_patches.size(0) != spots.expression.size(0):
        raise ValueError("image_patches and spot expression must have the same number of spots")

    reference = align_reference_to_genes(reference, spots.gene_names)
    edge_index = None
    edge_weight = None
    if graph_k is not None and spots.coords is not None and spots.coords.size(0) > 1:
        edge_index, edge_weight = build_knn_graph(spots.coords, k=graph_k)

    batch = WaveSTGateBatch(
        image_patches=image_patches.float(),
        st_expression=spots.expression.float(),
        reference_prototypes=reference.prototypes.float(),
        coords=spots.coords.float() if spots.coords is not None else None,
        edge_index=edge_index,
        edge_weight=edge_weight,
        proportion_gt=_align_proportion_table(proportions, spots.spot_ids, reference.cell_types),
        niche_gt=_align_niche_labels(niche_labels, spots.spot_ids),
    )
    metadata = BatchMetadata(
        spot_ids=spots.spot_ids,
        gene_names=spots.gene_names,
        cell_types=reference.cell_types,
    )
    return batch, metadata
