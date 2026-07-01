"""Xenium-to-Visium ground-truth construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from wavestgate.data.preprocess_st import read_delimited_table


def count_cells_in_spots(
    cells: pd.DataFrame,
    spots: pd.DataFrame,
    spot_radius: float,
    cell_type_col: str = "cell_type",
    cell_x_col: str = "x",
    cell_y_col: str = "y",
    spot_id_col: str = "spot_id",
    spot_x_col: str = "x",
    spot_y_col: str = "y",
) -> pd.DataFrame:
    """Count Xenium cells by cell type inside each Visium spot radius."""

    required_cells = [cell_type_col, cell_x_col, cell_y_col]
    required_spots = [spot_id_col, spot_x_col, spot_y_col]
    for col in required_cells:
        if col not in cells.columns:
            raise ValueError(f"cells table is missing column: {col}")
    for col in required_spots:
        if col not in spots.columns:
            raise ValueError(f"spots table is missing column: {col}")

    cell_types = sorted(cells[cell_type_col].astype(str).unique().tolist())
    spot_ids = spots[spot_id_col].astype(str).tolist()
    counts = pd.DataFrame(0, index=spot_ids, columns=cell_types, dtype=np.int64)

    cell_xy = cells[[cell_x_col, cell_y_col]].astype(float).values
    spot_xy = spots[[spot_x_col, spot_y_col]].astype(float).values
    if len(cells) == 0 or len(spots) == 0:
        return counts

    nbrs = NearestNeighbors(radius=spot_radius).fit(cell_xy)
    _, neighbor_indices = nbrs.radius_neighbors(spot_xy)
    cell_labels = cells[cell_type_col].astype(str).to_numpy()
    for spot_id, indices in zip(spot_ids, neighbor_indices):
        if len(indices) == 0:
            continue
        labels, label_counts = np.unique(cell_labels[indices], return_counts=True)
        for label, value in zip(labels, label_counts):
            counts.loc[spot_id, label] = int(value)
    return counts


def proportions_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """Normalize cell-type counts to per-spot proportions."""

    totals = counts.sum(axis=1)
    proportions = counts.astype(float).div(totals.replace(0, np.nan), axis=0)
    return proportions.fillna(0.0)


def xenium_to_visium_proportions(
    cells_path: str | Path,
    spots_path: str | Path,
    spot_radius: float,
    output_counts_path: str | Path | None = None,
    output_proportions_path: str | Path | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cell/spot tables and construct counts plus proportions."""

    cells = read_delimited_table(cells_path)
    spots = read_delimited_table(spots_path)
    counts = count_cells_in_spots(cells, spots, spot_radius=spot_radius, **kwargs)
    proportions = proportions_from_counts(counts)
    if output_counts_path is not None:
        counts.to_csv(output_counts_path, index_label="spot_id")
    if output_proportions_path is not None:
        proportions.to_csv(output_proportions_path, index_label="spot_id")
    return counts, proportions
