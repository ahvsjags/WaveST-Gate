"""Reproducible Xenium-to-Visium benchmark construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wavestgate.data.ground_truth import count_cells_in_spots, proportions_from_counts
from wavestgate.data.preprocess_st import read_delimited_table
from wavestgate.training.train import load_config


DEFAULT_METRICS = [
    "spotwise_cosine",
    "mean_celltype_pearson",
    "rmse",
    "jsd",
    "uncertainty_error_pearson",
    "uncertainty_risk_gap",
]


def make_spatial_holdout_splits(
    spots: pd.DataFrame,
    spot_id_col: str = "spot_id",
    x_col: str = "x",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> pd.DataFrame:
    """Create deterministic spatial holdout splits along the x-axis."""

    for col in [spot_id_col, x_col]:
        if col not in spots.columns:
            raise ValueError(f"spots table is missing column: {col}")
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("val_fraction and test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    n_spots = len(spots)
    labels = np.array(["train"] * n_spots, dtype=object)
    if n_spots == 0:
        return pd.DataFrame(columns=["spot_id", "split"])

    n_test = int(round(n_spots * test_fraction))
    n_val = int(round(n_spots * val_fraction))
    if n_spots >= 3:
        n_test = max(n_test, 1)
        n_val = max(n_val, 1)
    if n_test + n_val >= n_spots:
        n_test = max(min(n_test, n_spots - 2), 0)
        n_val = max(min(n_val, n_spots - n_test - 1), 0)

    order = np.argsort(spots[x_col].astype(float).to_numpy())
    if n_test > 0:
        labels[order[:n_test]] = "test"
    if n_val > 0:
        labels[order[-n_val:]] = "val"

    return pd.DataFrame(
        {
            "spot_id": spots[spot_id_col].astype(str).to_numpy(),
            "split": labels,
        }
    )


def summarize_ground_truth_qc(counts: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-spot Xenium coverage and cell-type diversity."""

    totals = counts.sum(axis=1).astype(float)
    proportions = proportions_from_counts(counts)
    values = proportions.to_numpy(dtype=float)
    entropy = -(values * np.log(np.clip(values, 1e-12, None))).sum(axis=1)
    dominant = proportions.idxmax(axis=1) if len(proportions.columns) else pd.Series("", index=proportions.index)
    qc = pd.DataFrame(
        {
            "spot_id": counts.index.astype(str),
            "xenium_cell_count": totals.to_numpy(),
            "ground_truth_entropy": entropy,
            "dominant_cell_type": dominant.astype(str).to_numpy(),
            "has_xenium_ground_truth": (totals > 0).to_numpy(),
        }
    )
    return qc


def build_xenium_visium_benchmark(
    cells_path: str | Path,
    spots_path: str | Path,
    output_dir: str | Path,
    spot_radius: float,
    cell_type_col: str = "cell_type",
    cell_x_col: str = "x",
    cell_y_col: str = "y",
    spot_id_col: str = "spot_id",
    spot_x_col: str = "x",
    spot_y_col: str = "y",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, Any]:
    """Aggregate Xenium cells to Visium spots and write benchmark artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = read_delimited_table(cells_path)
    spots = read_delimited_table(spots_path)

    counts = count_cells_in_spots(
        cells=cells,
        spots=spots,
        spot_radius=spot_radius,
        cell_type_col=cell_type_col,
        cell_x_col=cell_x_col,
        cell_y_col=cell_y_col,
        spot_id_col=spot_id_col,
        spot_x_col=spot_x_col,
        spot_y_col=spot_y_col,
    )
    proportions = proportions_from_counts(counts)
    splits = make_spatial_holdout_splits(
        spots,
        spot_id_col=spot_id_col,
        x_col=spot_x_col,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    qc = summarize_ground_truth_qc(counts)

    counts_path = output_dir / "xenium_cell_counts.csv"
    proportions_path = output_dir / "xenium_cell_proportions.csv"
    splits_path = output_dir / "spot_splits.csv"
    qc_path = output_dir / "spot_ground_truth_qc.csv"
    manifest_path = output_dir / "xenium_visium_benchmark_manifest.json"

    counts.to_csv(counts_path, index_label="spot_id")
    proportions.to_csv(proportions_path, index_label="spot_id")
    splits.to_csv(splits_path, index=False)
    qc.to_csv(qc_path, index=False)

    manifest = {
        "protocol": "xenium_to_visium_spot_aggregation",
        "cells_path": str(cells_path),
        "spots_path": str(spots_path),
        "spot_radius": float(spot_radius),
        "columns": {
            "cell_type_col": cell_type_col,
            "cell_x_col": cell_x_col,
            "cell_y_col": cell_y_col,
            "spot_id_col": spot_id_col,
            "spot_x_col": spot_x_col,
            "spot_y_col": spot_y_col,
        },
        "split_protocol": {
            "name": "deterministic_spatial_holdout_x_axis",
            "val_fraction": float(val_fraction),
            "test_fraction": float(test_fraction),
        },
        "metrics": DEFAULT_METRICS,
        "artifacts": {
            "counts": str(counts_path),
            "proportions": str(proportions_path),
            "splits": str(splits_path),
            "qc": str(qc_path),
            "manifest": str(manifest_path),
        },
        "num_spots": int(len(spots)),
        "num_cells": int(len(cells)),
        "num_cell_types": int(len(counts.columns)),
        "num_spots_with_ground_truth": int((counts.sum(axis=1) > 0).sum()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_from_config(path: str | Path) -> dict[str, Any]:
    """Build a benchmark from a YAML/JSON config."""

    raw = load_config(path)
    data_cfg = raw.get("data", {})
    required = ["cells_path", "spots_path", "output_dir"]
    missing = [key for key in required if key not in data_cfg]
    if missing:
        raise ValueError(f"Benchmark config is missing required data keys: {missing}")
    xenium_cfg = raw.get("xenium", {})
    if "spot_radius" not in xenium_cfg:
        raise ValueError("Benchmark config requires xenium.spot_radius")
    columns = raw.get("columns", {})
    split = raw.get("split", {})
    return build_xenium_visium_benchmark(
        cells_path=data_cfg["cells_path"],
        spots_path=data_cfg["spots_path"],
        output_dir=data_cfg["output_dir"],
        spot_radius=float(xenium_cfg["spot_radius"]),
        cell_type_col=columns.get("cell_type_col", "cell_type"),
        cell_x_col=columns.get("cell_x_col", "x"),
        cell_y_col=columns.get("cell_y_col", "y"),
        spot_id_col=columns.get("spot_id_col", "spot_id"),
        spot_x_col=columns.get("spot_x_col", "x"),
        spot_y_col=columns.get("spot_y_col", "y"),
        val_fraction=float(split.get("val_fraction", 0.15)),
        test_fraction=float(split.get("test_fraction", 0.15)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Xenium-to-Visium deconvolution benchmark.")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML/JSON config.")
    args = parser.parse_args()
    manifest = build_from_config(args.config)
    for key in ["num_spots", "num_cells", "num_cell_types", "num_spots_with_ground_truth"]:
        print(f"{key}: {manifest[key]}")
    print(f"manifest: {manifest['artifacts']['manifest']}")


if __name__ == "__main__":
    main()
