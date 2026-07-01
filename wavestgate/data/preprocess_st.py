"""ST expression and coordinate preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch


@dataclass
class SpotExpressionTable:
    """Spot expression with optional pixel/spatial coordinates."""

    expression: torch.Tensor
    spot_ids: list[str]
    gene_names: list[str]
    coords: torch.Tensor | None = None


def read_delimited_table(path: str | Path) -> pd.DataFrame:
    """Read CSV/TSV/TXT using the suffix to select the delimiter."""

    path = Path(path)
    if path.suffix == ".tsv" or path.name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t")
    if path.suffix in {".csv", ".txt"} or path.name.endswith((".csv.gz", ".txt.gz")):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _infer_id_column(df: pd.DataFrame, preferred: str | None) -> str | None:
    if preferred and preferred in df.columns:
        return preferred
    first = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[first]):
        return str(first)
    return None


def numeric_gene_frame(df: pd.DataFrame, exclude: Sequence[str] = ()) -> pd.DataFrame:
    """Return numeric gene columns, excluding metadata columns."""

    excluded = set(exclude)
    gene_df = df.drop(columns=[col for col in excluded if col in df.columns])
    gene_df = gene_df.apply(pd.to_numeric, errors="coerce")
    gene_df = gene_df.dropna(axis=1, how="all")
    if gene_df.empty:
        raise ValueError("No numeric gene columns found")
    return gene_df.fillna(0.0)


def load_spot_expression(
    expression_path: str | Path,
    spot_id_col: str | None = "spot_id",
    gene_names: Sequence[str] | None = None,
) -> SpotExpressionTable:
    """Load a spot-by-gene expression table."""

    df = read_delimited_table(expression_path)
    id_col = _infer_id_column(df, spot_id_col)
    spot_ids = df[id_col].astype(str).tolist() if id_col is not None else [str(i) for i in range(len(df))]
    exclude = [id_col] if id_col is not None else []
    gene_df = numeric_gene_frame(df, exclude=exclude)
    if gene_names is not None:
        missing = [gene for gene in gene_names if gene not in gene_df.columns]
        if missing:
            raise ValueError(f"Expression table is missing requested genes: {missing[:5]}")
        gene_df = gene_df.loc[:, list(gene_names)]
    return SpotExpressionTable(
        expression=torch.as_tensor(gene_df.values, dtype=torch.float32),
        spot_ids=spot_ids,
        gene_names=[str(col) for col in gene_df.columns],
    )


def load_spot_coordinates(
    coords_path: str | Path,
    spot_id_col: str = "spot_id",
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Load spot coordinates indexed by spot id."""

    df = read_delimited_table(coords_path)
    for col in [spot_id_col, x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Coordinate table is missing column: {col}")
    coords = df[[spot_id_col, x_col, y_col]].copy()
    coords[spot_id_col] = coords[spot_id_col].astype(str)
    coords[x_col] = pd.to_numeric(coords[x_col], errors="raise")
    coords[y_col] = pd.to_numeric(coords[y_col], errors="raise")
    return coords.set_index(spot_id_col)


def attach_coordinates(
    spots: SpotExpressionTable,
    coords: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
) -> SpotExpressionTable:
    """Attach coordinates to an already loaded spot expression table."""

    missing = [spot_id for spot_id in spots.spot_ids if spot_id not in coords.index]
    if missing:
        raise ValueError(f"Coordinate table is missing spot ids: {missing[:5]}")
    coord_values = coords.loc[spots.spot_ids, [x_col, y_col]].values
    return SpotExpressionTable(
        expression=spots.expression,
        spot_ids=spots.spot_ids,
        gene_names=spots.gene_names,
        coords=torch.as_tensor(coord_values, dtype=torch.float32),
    )
