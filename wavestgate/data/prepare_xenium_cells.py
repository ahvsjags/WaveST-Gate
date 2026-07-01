"""Prepare typed Xenium cell tables for Visium aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from wavestgate.data.preprocess_st import read_delimited_table


def normalize_cell_type(value: object) -> str:
    """Normalize workbook cell-type labels to reference-prototype labels."""

    label = str(value).strip()
    label = label.replace("_", " ")
    label = " ".join(label.split())
    return label


def load_affine_matrix(path: str | Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.shape != (3, 3):
        raise ValueError("Affine matrix must have shape [3, 3]")
    return matrix.astype(float)


def apply_affine(
    x: np.ndarray,
    y: np.ndarray,
    matrix: np.ndarray,
    direction: str = "inverse",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a 3x3 affine transform to cell centroids."""

    if direction not in {"forward", "inverse", "none"}:
        raise ValueError("direction must be one of: forward, inverse, none")
    if direction == "none":
        return x.astype(float), y.astype(float)
    transform = matrix if direction == "forward" else np.linalg.inv(matrix)
    pts = np.stack([x.astype(float), y.astype(float), np.ones_like(x, dtype=float)], axis=1)
    out = pts @ transform.T
    return out[:, 0], out[:, 1]


def read_xenium_label_sheet(
    labels_path: str | Path,
    sheet_name: str,
    barcode_col: str = "Barcode",
    label_col: str = "Cluster",
) -> pd.DataFrame:
    labels = pd.read_excel(labels_path, sheet_name=sheet_name)
    for col in [barcode_col, label_col]:
        if col not in labels.columns:
            raise ValueError(f"Xenium label sheet is missing column: {col}")
    labels = labels[[barcode_col, label_col]].copy()
    labels[barcode_col] = labels[barcode_col].astype(str)
    labels["cell_type"] = labels[label_col].map(normalize_cell_type)
    return labels.rename(columns={barcode_col: "cell_id"})[["cell_id", "cell_type"]]


def prepare_typed_xenium_cells(
    cells_path: str | Path,
    labels_path: str | Path,
    sheet_name: str,
    output_path: str | Path,
    qc_path: str | Path | None = None,
    affine_path: str | Path | None = None,
    affine_direction: str = "inverse",
    cell_id_col: str = "cell_id",
    x_col: str = "x_centroid",
    y_col: str = "y_centroid",
    drop_labels: Iterable[str] = ("Unlabeled", "Undefined"),
) -> dict[str, object]:
    """Join Xenium cell coordinates with supervised cell-type labels."""

    cells = read_delimited_table(cells_path)
    for col in [cell_id_col, x_col, y_col]:
        if col not in cells.columns:
            raise ValueError(f"Xenium cells table is missing column: {col}")
    cells = cells.copy()
    cells[cell_id_col] = cells[cell_id_col].astype(str)
    labels = read_xenium_label_sheet(labels_path, sheet_name=sheet_name)
    merged = cells.merge(labels, left_on=cell_id_col, right_on="cell_id", how="left", suffixes=("", "_label"))
    missing_label = int(merged["cell_type"].isna().sum())
    merged["cell_type"] = merged["cell_type"].fillna("Unlabeled")

    drop = {normalize_cell_type(label) for label in drop_labels}
    if drop:
        merged = merged[~merged["cell_type"].isin(drop)].copy()

    if affine_path is not None:
        matrix = load_affine_matrix(affine_path)
        out_x, out_y = apply_affine(
            merged[x_col].to_numpy(dtype=float),
            merged[y_col].to_numpy(dtype=float),
            matrix,
            direction=affine_direction,
        )
    else:
        out_x = merged[x_col].to_numpy(dtype=float)
        out_y = merged[y_col].to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "cell_id": merged[cell_id_col].astype(str).to_numpy(),
            "x": out_x,
            "y": out_y,
            "x_xenium": merged[x_col].astype(float).to_numpy(),
            "y_xenium": merged[y_col].astype(float).to_numpy(),
            "cell_type": merged["cell_type"].astype(str).to_numpy(),
            "transcript_counts": pd.to_numeric(merged.get("transcript_counts", 0), errors="coerce").fillna(0).to_numpy(),
            "cell_area": pd.to_numeric(merged.get("cell_area", 0), errors="coerce").fillna(0).to_numpy(),
        }
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    counts = out["cell_type"].value_counts().rename_axis("cell_type").reset_index(name="num_cells")
    qc = {
        "cells_path": str(cells_path),
        "labels_path": str(labels_path),
        "sheet_name": sheet_name,
        "affine_path": str(affine_path) if affine_path is not None else None,
        "affine_direction": affine_direction,
        "output_path": str(output_path),
        "num_input_cells": int(len(cells)),
        "num_output_cells": int(len(out)),
        "missing_label_cells": missing_label,
        "num_cell_types": int(counts.shape[0]),
        "dropped_labels": sorted(drop),
        "x_min": float(out["x"].min()) if len(out) else None,
        "x_max": float(out["x"].max()) if len(out) else None,
        "y_min": float(out["y"].min()) if len(out) else None,
        "y_max": float(out["y"].max()) if len(out) else None,
        "cell_type_counts": counts.to_dict(orient="records"),
    }
    if qc_path is not None:
        qc_path = Path(qc_path)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Join Xenium cells with supervised cell-type labels.")
    parser.add_argument("--cells", required=True, help="Xenium outs/cells.csv.gz.")
    parser.add_argument("--labels", required=True, help="Cell_Barcode_Type_Matrices.xlsx.")
    parser.add_argument("--sheet", required=True, help="Workbook sheet with Xenium supervised labels.")
    parser.add_argument("--output", required=True, help="Output typed cell CSV.")
    parser.add_argument("--qc", default=None, help="Optional output QC JSON.")
    parser.add_argument("--affine", default=None, help="Optional 3x3 imagealignment CSV.")
    parser.add_argument("--affine-direction", default="inverse", choices=["forward", "inverse", "none"])
    parser.add_argument("--keep-unlabeled", action="store_true", help="Keep Unlabeled/Undefined cells.")
    args = parser.parse_args()
    qc = prepare_typed_xenium_cells(
        cells_path=args.cells,
        labels_path=args.labels,
        sheet_name=args.sheet,
        output_path=args.output,
        qc_path=args.qc,
        affine_path=args.affine,
        affine_direction=args.affine_direction,
        drop_labels=() if args.keep_unlabeled else ("Unlabeled", "Undefined"),
    )
    print(json.dumps(qc, indent=2), flush=True)


if __name__ == "__main__":
    main()
