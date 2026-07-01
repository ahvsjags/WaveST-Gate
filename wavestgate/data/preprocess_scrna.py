"""scRNA reference prototype preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from wavestgate.data.preprocess_st import _infer_id_column, numeric_gene_frame, read_delimited_table


@dataclass
class ReferencePrototypes:
    """Mean expression prototypes by cell type."""

    prototypes: torch.Tensor
    cell_types: list[str]
    gene_names: list[str]


def load_cell_labels(
    labels_path: str | Path,
    cell_id_col: str = "cell_id",
    cell_type_col: str = "cell_type",
) -> pd.Series:
    """Load a cell-id-indexed cell-type label series."""

    labels = read_delimited_table(labels_path)
    for col in [cell_id_col, cell_type_col]:
        if col not in labels.columns:
            raise ValueError(f"Label table is missing column: {col}")
    labels[cell_id_col] = labels[cell_id_col].astype(str)
    return labels.set_index(cell_id_col)[cell_type_col].astype(str)


def build_reference_prototypes(
    expression_path: str | Path,
    labels_path: str | Path | None = None,
    cell_id_col: str | None = "cell_id",
    cell_type_col: str = "cell_type",
    target_genes: Sequence[str] | None = None,
) -> ReferencePrototypes:
    """Aggregate single-cell expression into mean cell-type prototypes."""

    df = read_delimited_table(expression_path)
    id_col = _infer_id_column(df, cell_id_col)
    cell_ids = df[id_col].astype(str).tolist() if id_col is not None else [str(i) for i in range(len(df))]

    if labels_path is None:
        if cell_type_col not in df.columns:
            raise ValueError("cell_type_col must exist in expression table when labels_path is not provided")
        labels = df[cell_type_col].astype(str)
    else:
        label_series = load_cell_labels(labels_path, cell_id_col=cell_id_col or "cell_id", cell_type_col=cell_type_col)
        missing = [cell_id for cell_id in cell_ids if cell_id not in label_series.index]
        if missing:
            raise ValueError(f"Label table is missing cell ids: {missing[:5]}")
        labels = label_series.loc[cell_ids].reset_index(drop=True)

    exclude = [col for col in [id_col, cell_type_col] if col is not None]
    genes = numeric_gene_frame(df, exclude=exclude)
    if target_genes is not None:
        genes = genes.reindex(columns=list(target_genes), fill_value=0.0)

    grouped = genes.groupby(labels.values).mean().sort_index()
    return ReferencePrototypes(
        prototypes=torch.as_tensor(grouped.values, dtype=torch.float32),
        cell_types=[str(idx) for idx in grouped.index],
        gene_names=[str(col) for col in grouped.columns],
    )


def load_reference_prototypes_table(
    prototypes_path: str | Path,
    cell_type_col: str = "cell_type",
    target_genes: Sequence[str] | None = None,
) -> ReferencePrototypes:
    """Load a cell-type-by-gene prototype table."""

    df = read_delimited_table(prototypes_path)
    if cell_type_col not in df.columns:
        raise ValueError(f"Prototype table is missing column: {cell_type_col}")
    cell_types = df[cell_type_col].astype(str).tolist()
    genes = numeric_gene_frame(df, exclude=[cell_type_col])
    if target_genes is not None:
        genes = genes.reindex(columns=list(target_genes), fill_value=0.0)
    return ReferencePrototypes(
        prototypes=torch.as_tensor(genes.values, dtype=torch.float32),
        cell_types=cell_types,
        gene_names=[str(col) for col in genes.columns],
    )


def align_reference_to_genes(reference: ReferencePrototypes, target_genes: Sequence[str]) -> ReferencePrototypes:
    """Reorder/fill reference genes to match an ST gene list."""

    values = reference.prototypes.detach().cpu().numpy()
    df = pd.DataFrame(values, index=reference.cell_types, columns=reference.gene_names)
    aligned = df.reindex(columns=list(target_genes), fill_value=0.0)
    return ReferencePrototypes(
        prototypes=torch.as_tensor(aligned.values, dtype=torch.float32),
        cell_types=reference.cell_types,
        gene_names=[str(gene) for gene in target_genes],
    )
