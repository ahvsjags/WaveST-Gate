"""Build pseudo-Visium spots from Xenium cell features and typed cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.spatial import cKDTree

from wavestgate.data.assemble import assemble_wavestgate_batch
from wavestgate.data.extract_he_patches import extract_spot_patches
from wavestgate.data.prepare_dataset import save_prepared_dataset
from wavestgate.data.preprocess_scrna import load_reference_prototypes_table
from wavestgate.data.preprocess_st import SpotExpressionTable


def _decode(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _load_feature_matrix(path: str | Path, feature_type: str = "Gene Expression") -> tuple[sparse.csc_matrix, list[str], list[str]]:
    """Load a Xenium cell_feature_matrix.h5 as features x cells CSC matrix."""

    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(v) for v in group["shape"][:])
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape,
        )
        barcodes = _decode(group["barcodes"][:])
        names = _decode(group["features/name"][:])
        feature_types = _decode(group["features/feature_type"][:])
    keep = [idx for idx, current_type in enumerate(feature_types) if current_type == feature_type]
    if not keep:
        raise ValueError(f"No features found with feature_type={feature_type!r}")
    return matrix[keep, :], [names[idx] for idx in keep], barcodes


def _load_target_genes(path: str | Path | None, available_genes: Sequence[str]) -> list[str]:
    if path is None:
        return list(available_genes)
    frame = pd.read_csv(path, header=None)
    genes = frame.iloc[:, 0].astype(str).tolist()
    available = set(available_genes)
    return [gene for gene in genes if gene in available]


def _make_grid(cells: pd.DataFrame, stride: float, margin: float) -> pd.DataFrame:
    x_min, x_max = float(cells["x"].min()), float(cells["x"].max())
    y_min, y_max = float(cells["y"].min()), float(cells["y"].max())
    xs = np.arange(x_min - margin, x_max + margin + 1e-6, stride)
    ys = np.arange(y_min - margin, y_max + margin + 1e-6, stride)
    rows = []
    for y_idx, y in enumerate(ys):
        for x_idx, x in enumerate(xs):
            rows.append({"spot_id": f"pseudo_x{x_idx:03d}_y{y_idx:03d}", "x": float(x), "y": float(y)})
    return pd.DataFrame(rows)


def _deterministic_spatial_split(coords: pd.DataFrame, val_fraction: float = 0.15, test_fraction: float = 0.15) -> pd.Series:
    x_rank = coords["x"].rank(method="first", pct=True)
    split = pd.Series("train", index=coords.index, dtype=object)
    split[x_rank <= test_fraction] = "test"
    split[(x_rank > test_fraction) & (x_rank <= test_fraction + val_fraction)] = "val"
    return split


def _aggregate_spots(
    cells: pd.DataFrame,
    feature_matrix: sparse.csc_matrix,
    gene_names: list[str],
    matrix_barcodes: list[str],
    centers: pd.DataFrame,
    radius: float,
    min_cells: int,
    cell_types: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    barcode_to_col = {barcode: idx for idx, barcode in enumerate(matrix_barcodes)}
    cells = cells.copy()
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells["matrix_col"] = cells["cell_id"].map(barcode_to_col)
    cells = cells.dropna(subset=["matrix_col"]).copy()
    cells["matrix_col"] = cells["matrix_col"].astype(int)
    tree = cKDTree(cells[["x", "y"]].to_numpy(dtype=float))
    cell_type_index = {cell_type: idx for idx, cell_type in enumerate(cell_types)}

    expression_rows = []
    count_rows = []
    qc_rows = []
    coord_rows = []
    cell_type_values = cells["cell_type"].astype(str).to_numpy()
    matrix_cols = cells["matrix_col"].to_numpy(dtype=int)
    for _, center in centers.iterrows():
        local_indices = tree.query_ball_point([float(center["x"]), float(center["y"])], r=float(radius))
        if len(local_indices) < min_cells:
            continue
        local_cols = matrix_cols[local_indices]
        local_types = cell_type_values[local_indices]
        counts = np.zeros(len(cell_types), dtype=np.int64)
        for cell_type in local_types:
            idx = cell_type_index.get(str(cell_type))
            if idx is not None:
                counts[idx] += 1
        if counts.sum() < min_cells:
            continue
        expr = np.asarray(feature_matrix[:, local_cols].sum(axis=1)).reshape(-1)
        spot_id = str(center["spot_id"])
        expression_rows.append({"spot_id": spot_id, **{gene: float(value) for gene, value in zip(gene_names, expr)}})
        count_rows.append({"spot_id": spot_id, **{cell_type: int(value) for cell_type, value in zip(cell_types, counts)}})
        proportions = counts / max(float(counts.sum()), 1.0)
        entropy = float(-(proportions[proportions > 0] * np.log(proportions[proportions > 0])).sum())
        qc_rows.append(
            {
                "spot_id": spot_id,
                "xenium_cell_count": int(counts.sum()),
                "ground_truth_entropy": entropy,
                "dominant_cell_type": cell_types[int(np.argmax(counts))],
                "has_xenium_ground_truth": True,
            }
        )
        coord_rows.append({"spot_id": spot_id, "x": float(center["x"]), "y": float(center["y"])})
    if not expression_rows:
        raise ValueError("No pseudo-spots passed the min_cells filter")
    expression = pd.DataFrame(expression_rows)
    counts = pd.DataFrame(count_rows)
    proportions = counts.copy()
    values = proportions[cell_types].to_numpy(dtype=float)
    values = values / np.maximum(values.sum(axis=1, keepdims=True), 1.0)
    proportions[cell_types] = values
    qc = pd.DataFrame(qc_rows)
    coords = pd.DataFrame(coord_rows)
    return expression, coords, counts, proportions, qc


def prepare_xenium_pseudospots(
    typed_cells_path: str | Path,
    cell_feature_matrix_path: str | Path,
    he_image_path: str | Path,
    reference_prototypes_path: str | Path,
    output_dir: str | Path,
    prepared_output: str | Path | None = None,
    target_genes_path: str | Path | None = None,
    radius: float = 55.0,
    stride: float = 110.0,
    min_cells: int = 5,
    patch_size: int = 128,
    graph_k: int = 6,
    feature_type: str = "Gene Expression",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, object]:
    """Create a pseudo-spot benchmark and prepared WaveST-Gate batch."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_output = Path(prepared_output) if prepared_output is not None else output_dir / "prepared.pt"
    cells = pd.read_csv(typed_cells_path)
    required_cols = {"cell_id", "x", "y", "cell_type"}
    missing = required_cols - set(cells.columns)
    if missing:
        raise ValueError(f"Typed cell table is missing columns: {sorted(missing)}")
    feature_matrix, available_genes, barcodes = _load_feature_matrix(cell_feature_matrix_path, feature_type=feature_type)
    target_genes = _load_target_genes(target_genes_path, available_genes)
    gene_idx = [available_genes.index(gene) for gene in target_genes]
    feature_matrix = feature_matrix[gene_idx, :]

    reference = load_reference_prototypes_table(reference_prototypes_path, target_genes=target_genes)
    cell_types = list(reference.cell_types)
    centers = _make_grid(cells, stride=stride, margin=radius)
    expression_df, coords_df, counts_df, proportions_df, qc_df = _aggregate_spots(
        cells=cells,
        feature_matrix=feature_matrix,
        gene_names=target_genes,
        matrix_barcodes=barcodes,
        centers=centers,
        radius=radius,
        min_cells=min_cells,
        cell_types=cell_types,
    )

    expression_path = output_dir / "spot_expression.csv.gz"
    coords_path = output_dir / "spot_coords.csv"
    counts_path = output_dir / "xenium_cell_counts.csv"
    proportions_path = output_dir / "xenium_cell_proportions.csv"
    qc_path = output_dir / "spot_ground_truth_qc.csv"
    split_path = output_dir / "spot_splits.csv"
    expression_df.to_csv(expression_path, index=False)
    coords_df.to_csv(coords_path, index=False)
    counts_df.to_csv(counts_path, index=False)
    proportions_df.to_csv(proportions_path, index=False)
    qc_df.to_csv(qc_path, index=False)
    split = _deterministic_spatial_split(coords_df.set_index("spot_id"), val_fraction=val_fraction, test_fraction=test_fraction)
    split.reset_index().rename(columns={0: "split"}).to_csv(split_path, index=False)

    spot_table = SpotExpressionTable(
        expression=torch.as_tensor(expression_df[target_genes].to_numpy(dtype=np.float32), dtype=torch.float32),
        spot_ids=expression_df["spot_id"].astype(str).tolist(),
        gene_names=target_genes,
        coords=torch.as_tensor(coords_df[["x", "y"]].to_numpy(dtype=np.float32), dtype=torch.float32),
    )
    patches = extract_spot_patches(
        he_image_path,
        coords_df.set_index("spot_id"),
        patch_size=patch_size,
        spot_ids=spot_table.spot_ids,
    )
    proportions_indexed = proportions_df.set_index("spot_id")
    batch, metadata = assemble_wavestgate_batch(
        spot_table,
        reference,
        image_patches=patches,
        proportions=proportions_indexed,
        graph_k=graph_k,
    )
    extra = {
        "source": "xenium_pseudospots",
        "typed_cells_path": str(typed_cells_path),
        "cell_feature_matrix_path": str(cell_feature_matrix_path),
        "he_image_path": str(he_image_path),
        "radius": float(radius),
        "stride": float(stride),
        "min_cells": int(min_cells),
        "patch_size": int(patch_size),
        "feature_type": feature_type,
        "num_candidate_spots": int(len(centers)),
        "num_spots": int(len(metadata.spot_ids)),
        "num_genes": int(len(metadata.gene_names)),
        "num_cell_types": int(len(metadata.cell_types)),
    }
    save_prepared_dataset(prepared_output, batch, metadata, extra=extra)
    manifest = {
        **extra,
        "prepared_path": str(prepared_output),
        "spot_expression_path": str(expression_path),
        "spot_coords_path": str(coords_path),
        "xenium_cell_counts_path": str(counts_path),
        "xenium_cell_proportions_path": str(proportions_path),
        "spot_ground_truth_qc_path": str(qc_path),
        "spot_splits_path": str(split_path),
        "reference_prototypes_path": str(reference_prototypes_path),
        "target_genes_path": str(target_genes_path) if target_genes_path is not None else "",
    }
    manifest_path = output_dir / "xenium_pseudospot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Xenium pseudo-spots for matched-GT external evaluation.")
    parser.add_argument("--typed-cells", required=True)
    parser.add_argument("--cell-feature-matrix", required=True)
    parser.add_argument("--he-image", required=True)
    parser.add_argument("--reference-prototypes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepared-output", default=None)
    parser.add_argument("--target-genes", default=None)
    parser.add_argument("--radius", type=float, default=55.0)
    parser.add_argument("--stride", type=float, default=110.0)
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--graph-k", type=int, default=6)
    parser.add_argument("--feature-type", default="Gene Expression")
    args = parser.parse_args()
    manifest = prepare_xenium_pseudospots(
        typed_cells_path=args.typed_cells,
        cell_feature_matrix_path=args.cell_feature_matrix,
        he_image_path=args.he_image,
        reference_prototypes_path=args.reference_prototypes,
        output_dir=args.output_dir,
        prepared_output=args.prepared_output,
        target_genes_path=args.target_genes,
        radius=args.radius,
        stride=args.stride,
        min_cells=args.min_cells,
        patch_size=args.patch_size,
        graph_k=args.graph_k,
        feature_type=args.feature_type,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
