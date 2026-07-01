"""Prepare labelled scRNA references for WaveST-Gate and baselines."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import io as scipy_io

from wavestgate.data.prepare_10x_visium import _load_gene_list, _select_gene_indices, read_10x_filtered_matrix


def read_labels_table(path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    """Read cell labels from CSV/TSV or XLSX."""

    path = Path(path)
    if path.suffix == ".xlsx":
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    if path.suffix == ".tsv" or path.name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t")
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported labels format: {path.suffix}")


def _write_prototypes(
    path: Path,
    cell_types: Sequence[str],
    gene_names: Sequence[str],
    values: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_type", *gene_names])
        for cell_type, row in zip(cell_types, values, strict=True):
            writer.writerow([cell_type, *[float(value) for value in row]])


def _write_cell_expression(path: Path, cell_ids: Sequence[str], gene_names: Sequence[str], matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", *gene_names])
        for cell_id, row in zip(cell_ids, matrix, strict=True):
            values = row.toarray().reshape(-1) if hasattr(row, "toarray") else np.asarray(row).reshape(-1)
            writer.writerow([cell_id, *[float(value) for value in values]])


def _write_sparse_mtx(path: Path, matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        scipy_io.mmwrite(handle, matrix)


def _write_tsv(path: Path, values: Sequence[str], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([header])
        for value in values:
            writer.writerow([value])


def convert_scrna_reference(
    matrix_h5_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    labels_sheet: str | None = None,
    barcode_col: str = "Barcode",
    cell_type_col: str = "Annotation",
    genes_path: str | Path | None = None,
    max_genes: int | None = None,
    write_cell_expression: bool = False,
    write_mtx: bool = False,
) -> dict[str, object]:
    """Create baseline labels and WaveST-Gate cell-type prototypes from a 10x scRNA h5."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tenx = read_10x_filtered_matrix(matrix_h5_path)
    labels = read_labels_table(labels_path, sheet_name=labels_sheet)
    for column in [barcode_col, cell_type_col]:
        if column not in labels.columns:
            raise ValueError(f"Labels table is missing column: {column}")
    labels = labels[[barcode_col, cell_type_col]].dropna().copy()
    labels[barcode_col] = labels[barcode_col].astype(str)
    labels[cell_type_col] = labels[cell_type_col].astype(str)
    labels = labels.drop_duplicates(subset=[barcode_col], keep="first")

    barcode_to_idx = {barcode: idx for idx, barcode in enumerate(tenx.barcodes)}
    labels = labels[labels[barcode_col].isin(barcode_to_idx)].copy()
    if labels.empty:
        raise ValueError("No label barcodes overlap the 10x matrix")
    cell_ids = labels[barcode_col].astype(str).tolist()
    row_indices = [barcode_to_idx[barcode] for barcode in cell_ids]
    requested_genes = _load_gene_list(genes_path)
    gene_indices, selected_genes = _select_gene_indices(tenx.gene_names, requested_genes, max_genes=max_genes)

    labelled_matrix = tenx.matrix[row_indices, :][:, gene_indices].tocsr()
    cell_types = sorted(labels[cell_type_col].unique())
    prototype_rows = []
    for cell_type in cell_types:
        group_idx = np.flatnonzero(labels[cell_type_col].values == cell_type)
        mean_row = labelled_matrix[group_idx].mean(axis=0)
        prototype_rows.append(np.asarray(mean_row).reshape(-1))
    prototypes = np.vstack(prototype_rows)

    labels_out = output / "cell_labels.csv"
    labels.rename(columns={barcode_col: "cell_id", cell_type_col: "cell_type"}).to_csv(labels_out, index=False)
    prototypes_out = output / "reference_prototypes.csv.gz"
    _write_prototypes(prototypes_out, cell_types, selected_genes, prototypes)
    genes_out = output / "genes.csv"
    pd.DataFrame(
        {
            "gene_name": selected_genes,
            "gene_id": [tenx.gene_ids[idx] for idx in gene_indices],
            "feature_type": [tenx.feature_types[idx] for idx in gene_indices],
        }
    ).to_csv(genes_out, index=False)

    cell_type_counts = labels[cell_type_col].value_counts().sort_index()
    cell_type_counts_out = output / "cell_type_counts.csv"
    cell_type_counts.rename_axis("cell_type").reset_index(name="num_cells").to_csv(cell_type_counts_out, index=False)

    cell_expression_out = None
    if write_cell_expression:
        cell_expression_out = output / "cell_expression.csv.gz"
        _write_cell_expression(cell_expression_out, cell_ids, selected_genes, labelled_matrix)

    reference_mtx_out = None
    reference_cells_out = None
    reference_genes_out = None
    if write_mtx:
        reference_mtx_out = output / "reference_gene_by_cell.mtx.gz"
        reference_cells_out = output / "reference_cells.tsv"
        reference_genes_out = output / "reference_genes.tsv"
        _write_sparse_mtx(reference_mtx_out, labelled_matrix.T.tocoo())
        _write_tsv(reference_cells_out, cell_ids, "cell_id")
        _write_tsv(reference_genes_out, selected_genes, "gene_name")

    manifest = {
        "dataset_id": dataset_id,
        "raw_matrix_h5_path": str(matrix_h5_path),
        "labels_path": str(labels_path),
        "labels_sheet": labels_sheet,
        "cell_labels_path": str(labels_out),
        "cell_expression_path": str(cell_expression_out) if cell_expression_out is not None else None,
        "reference_prototypes_path": str(prototypes_out),
        "reference_mtx_path": str(reference_mtx_out) if reference_mtx_out is not None else None,
        "reference_cells_path": str(reference_cells_out) if reference_cells_out is not None else None,
        "reference_genes_path": str(reference_genes_out) if reference_genes_out is not None else None,
        "genes_path": str(genes_out),
        "cell_type_counts_path": str(cell_type_counts_out),
        "num_labelled_cells": int(len(labels)),
        "num_cell_types": int(len(cell_types)),
        "num_genes": int(len(selected_genes)),
        "baseline_notes": {
            "cell2location": "Use raw_matrix_h5_path plus cell_labels_path for cell-type labels.",
            "rctd": "Use raw_matrix_h5_path plus cell_labels_path as the reference object.",
            "card": "Use raw_matrix_h5_path plus cell_labels_path as the reference object.",
            "tangram": "Use raw_matrix_h5_path plus cell_labels_path and align genes to Visium datasets.",
        },
    }
    manifest_out = output / "baseline_scrna_reference.json"
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "dataset_id": dataset_id,
        "cell_labels_path": str(labels_out),
        "cell_expression_path": str(cell_expression_out) if cell_expression_out is not None else None,
        "reference_prototypes_path": str(prototypes_out),
        "reference_mtx_path": str(reference_mtx_out) if reference_mtx_out is not None else None,
        "reference_cells_path": str(reference_cells_out) if reference_cells_out is not None else None,
        "reference_genes_path": str(reference_genes_out) if reference_genes_out is not None else None,
        "genes_path": str(genes_out),
        "cell_type_counts_path": str(cell_type_counts_out),
        "baseline_manifest_path": str(manifest_out),
        "num_labelled_cells": int(len(labels)),
        "num_cell_types": int(len(cell_types)),
        "num_genes": int(len(selected_genes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare labelled scRNA reference prototypes from 10x h5 and labels.")
    parser.add_argument("--matrix-h5", required=True, help="10x single-cell feature matrix h5.")
    parser.add_argument("--labels", required=True, help="CSV/TSV/XLSX cell labels.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--dataset-id", required=True, help="Reference dataset id.")
    parser.add_argument("--labels-sheet", default=None, help="XLSX sheet name for labels.")
    parser.add_argument("--barcode-col", default="Barcode", help="Barcode/cell id column in labels.")
    parser.add_argument("--cell-type-col", default="Annotation", help="Cell type column in labels.")
    parser.add_argument("--genes", default=None, help="Optional newline-delimited gene list.")
    parser.add_argument("--max-genes", type=int, default=None, help="Optional gene cap for smoke tests.")
    parser.add_argument("--write-cell-expression", action="store_true", help="Write labelled cell-by-gene expression CSV.")
    parser.add_argument("--write-mtx", action="store_true", help="Write sparse gene-by-cell MatrixMarket reference.")
    args = parser.parse_args()
    result = convert_scrna_reference(
        matrix_h5_path=args.matrix_h5,
        labels_path=args.labels,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        labels_sheet=args.labels_sheet,
        barcode_col=args.barcode_col,
        cell_type_col=args.cell_type_col,
        genes_path=args.genes,
        max_genes=args.max_genes,
        write_cell_expression=args.write_cell_expression,
        write_mtx=args.write_mtx,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
