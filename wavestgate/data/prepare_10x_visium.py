"""Convert 10x Visium outputs into WaveST-Gate standard tables."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml


@dataclass
class TenxMatrix:
    """Sparse 10x matrix in spot-by-gene orientation."""

    matrix: "sparse.csr_matrix"
    barcodes: list[str]
    gene_ids: list[str]
    gene_names: list[str]
    feature_types: list[str]


def _require_10x_dependencies():
    try:
        import h5py
        from scipy import sparse
    except Exception as exc:  # pragma: no cover - exercised only when optional deps are absent.
        raise RuntimeError("prepare_10x_visium requires optional dependencies: h5py and scipy") from exc
    return h5py, sparse


def _decode_array(values: np.ndarray) -> list[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def read_10x_filtered_matrix(path: str | Path) -> TenxMatrix:
    """Read a 10x `filtered_feature_bc_matrix.h5` file."""

    h5py, sparse = _require_10x_dependencies()
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        data = group["data"][()]
        indices = group["indices"][()]
        indptr = group["indptr"][()]
        shape = tuple(int(v) for v in group["shape"][()])
        features = group["features"]
        gene_ids = _decode_array(features["id"][()])
        gene_names = _decode_array(features["name"][()])
        feature_types = _decode_array(features["feature_type"][()]) if "feature_type" in features else ["Gene Expression"] * len(gene_names)
        barcodes = _decode_array(group["barcodes"][()])

    gene_by_spot = sparse.csc_matrix((data, indices, indptr), shape=shape)
    return TenxMatrix(
        matrix=gene_by_spot.T.tocsr(),
        barcodes=barcodes,
        gene_ids=gene_ids,
        gene_names=gene_names,
        feature_types=feature_types,
    )


def _read_tar_csv(tar_path: Path, suffixes: Sequence[str]) -> pd.DataFrame:
    with tarfile.open(tar_path, "r:gz") as archive:
        candidates = [member for member in archive.getmembers() if any(member.name.endswith(suffix) for suffix in suffixes)]
        if not candidates:
            raise ValueError(f"Could not find any of {suffixes} in {tar_path}")
        member = candidates[0]
        fileobj = archive.extractfile(member)
        if fileobj is None:
            raise ValueError(f"Could not read {member.name} from {tar_path}")
        return pd.read_csv(fileobj)


def read_10x_spatial_positions(path: str | Path) -> pd.DataFrame:
    """Read Space Ranger spatial coordinates from a `spatial.tar.gz` bundle."""

    path = Path(path)
    try:
        df = _read_tar_csv(path, ["spatial/tissue_positions.csv", "tissue_positions.csv"])
    except ValueError:
        with tarfile.open(path, "r:gz") as archive:
            candidates = [
                member for member in archive.getmembers() if member.name.endswith(("spatial/tissue_positions_list.csv", "tissue_positions_list.csv"))
            ]
            if not candidates:
                raise
            fileobj = archive.extractfile(candidates[0])
            if fileobj is None:
                raise ValueError(f"Could not read {candidates[0].name} from {path}")
            df = pd.read_csv(fileobj, header=None)
            df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]

    rename = {
        "barcode": "spot_id",
        "pxl_col_in_fullres": "x",
        "pxl_row_in_fullres": "y",
    }
    df = df.rename(columns=rename)
    required = ["spot_id", "in_tissue", "array_row", "array_col", "x", "y"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Spatial positions are missing columns: {missing}")
    out = df[required].copy()
    out["spot_id"] = out["spot_id"].astype(str)
    for column in ["in_tissue", "array_row", "array_col"]:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(int)
    for column in ["x", "y"]:
        out[column] = pd.to_numeric(out[column], errors="raise")
    return out


def _load_gene_list(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    values = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                values.append(value)
    return values


def _select_gene_indices(gene_names: list[str], requested_genes: Sequence[str] | None, max_genes: int | None) -> tuple[list[int], list[str]]:
    if requested_genes is None:
        indices = list(range(len(gene_names)))
    else:
        lookup: dict[str, int] = {}
        for idx, gene in enumerate(gene_names):
            lookup.setdefault(gene, idx)
        missing = [gene for gene in requested_genes if gene not in lookup]
        if missing:
            raise ValueError(f"Requested genes are missing from the 10x matrix: {missing[:10]}")
        indices = [lookup[gene] for gene in requested_genes]
    if max_genes is not None:
        indices = indices[:max_genes]
    return indices, [gene_names[idx] for idx in indices]


def _write_expression_csv_gz(
    path: Path,
    matrix: "sparse.csr_matrix",
    spot_ids: list[str],
    gene_names: list[str],
    row_indices: Sequence[int],
    gene_indices: Sequence[int],
    chunk_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["spot_id", *gene_names])
        for start in range(0, len(row_indices), chunk_size):
            chunk_rows = list(row_indices[start : start + chunk_size])
            dense = matrix[chunk_rows, :][:, gene_indices].toarray()
            for spot_id, values in zip((spot_ids[idx] for idx in chunk_rows), dense, strict=True):
                writer.writerow([spot_id, *[float(value) for value in values]])


def convert_10x_visium(
    matrix_h5_path: str | Path,
    spatial_tar_path: str | Path,
    image_path: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    scrna_expression_path: str | Path | None = None,
    reference_prototypes_path: str | Path | None = None,
    genes_path: str | Path | None = None,
    max_genes: int | None = None,
    in_tissue_only: bool = True,
    patch_size: int = 256,
    graph_k: int = 6,
    chunk_size: int = 128,
) -> dict[str, object]:
    """Convert one Visium dataset into standard files and metadata."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tenx = read_10x_filtered_matrix(matrix_h5_path)
    spatial = read_10x_spatial_positions(spatial_tar_path)
    spatial = spatial.set_index("spot_id")

    selected_rows = []
    selected_spots = []
    missing_positions = []
    for idx, barcode in enumerate(tenx.barcodes):
        if barcode not in spatial.index:
            missing_positions.append(barcode)
            continue
        if in_tissue_only and int(spatial.loc[barcode, "in_tissue"]) != 1:
            continue
        selected_rows.append(idx)
        selected_spots.append(barcode)
    if missing_positions:
        raise ValueError(f"Spatial bundle is missing barcodes from matrix: {missing_positions[:5]}")
    if not selected_rows:
        raise ValueError("No spots selected after applying in_tissue filter")

    requested_genes = _load_gene_list(genes_path)
    gene_indices, selected_genes = _select_gene_indices(tenx.gene_names, requested_genes, max_genes=max_genes)

    expression_path = output / "spot_expression.csv.gz"
    coords_path = output / "spot_coords.csv"
    _write_expression_csv_gz(
        expression_path,
        tenx.matrix,
        tenx.barcodes,
        selected_genes,
        selected_rows,
        gene_indices,
        chunk_size=chunk_size,
    )

    coords = spatial.loc[selected_spots, ["x", "y", "in_tissue", "array_row", "array_col"]].reset_index()
    coords = coords.rename(columns={"index": "spot_id"})
    coords.to_csv(coords_path, index=False)

    genes_path_out = output / "genes.csv"
    pd.DataFrame(
        {
            "gene_name": selected_genes,
            "gene_id": [tenx.gene_ids[idx] for idx in gene_indices],
            "feature_type": [tenx.feature_types[idx] for idx in gene_indices],
        }
    ).to_csv(genes_path_out, index=False)

    baseline_manifest = {
        "dataset_id": dataset_id,
        "raw_matrix_h5_path": str(matrix_h5_path),
        "raw_spatial_tar_path": str(spatial_tar_path),
        "image_path": str(image_path),
        "spot_expression_path": str(expression_path),
        "spot_coords_path": str(coords_path),
        "genes_path": str(genes_path_out),
        "scrna_expression_path": str(scrna_expression_path) if scrna_expression_path is not None else None,
        "reference_prototypes_path": str(reference_prototypes_path) if reference_prototypes_path is not None else None,
        "num_spots": len(selected_spots),
        "num_genes": len(selected_genes),
        "in_tissue_only": in_tissue_only,
        "baseline_notes": {
            "cell2location": "Use raw_matrix_h5_path or spot_expression_path plus scrna_expression_path and cell-type labels.",
            "rctd": "Use spot_expression_path, spot_coords_path, and the same scRNA reference.",
            "card": "Use spot_expression_path, spot_coords_path, and the same scRNA reference.",
            "tangram": "Use spot_expression_path and scrna_expression_path aligned to genes_path.",
        },
    }
    baseline_manifest_path = output / "baseline_manifest.json"
    baseline_manifest_path.write_text(json.dumps(baseline_manifest, indent=2), encoding="utf-8")

    data_config = {
        "spot_expression_path": str(expression_path),
        "spot_coords_path": str(coords_path),
        "he_image_path": str(image_path),
        "output_path": str(output / "prepared.pt"),
    }
    if reference_prototypes_path is not None:
        data_config["reference_prototypes_path"] = str(reference_prototypes_path)
    else:
        data_config["scrna_expression_path"] = str(scrna_expression_path) if scrna_expression_path is not None else "CHANGE_ME_scrna_expression.csv"

    prepare_config = {
        "data": data_config,
        "patches": {"patch_size": patch_size},
        "graph": {"k": graph_k},
        "columns": {"spot_id_col": "spot_id", "spot_x_col": "x", "spot_y_col": "y"},
    }
    config_name = "prepare_dataset.yaml" if scrna_expression_path is not None or reference_prototypes_path is not None else "prepare_dataset.template.yaml"
    prepare_config_path = output / config_name
    prepare_config_path.write_text(yaml.safe_dump(prepare_config, sort_keys=False), encoding="utf-8")

    return {
        "dataset_id": dataset_id,
        "spot_expression_path": str(expression_path),
        "spot_coords_path": str(coords_path),
        "genes_path": str(genes_path_out),
        "baseline_manifest_path": str(baseline_manifest_path),
        "prepare_config_path": str(prepare_config_path),
        "num_spots": len(selected_spots),
        "num_genes": len(selected_genes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert 10x Visium files into WaveST-Gate standard tables.")
    parser.add_argument("--matrix-h5", required=True, help="10x filtered_feature_bc_matrix.h5 path.")
    parser.add_argument("--spatial-tar", required=True, help="10x spatial.tar.gz path.")
    parser.add_argument("--image", required=True, help="Full-resolution H&E/IF image path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for standardized files.")
    parser.add_argument("--dataset-id", required=True, help="Dataset identifier.")
    parser.add_argument("--scrna-expression-path", default=None, help="Optional shared scRNA expression table for prepare_dataset config.")
    parser.add_argument("--reference-prototypes-path", default=None, help="Optional shared cell-type prototype table for prepare_dataset config.")
    parser.add_argument("--genes", default=None, help="Optional newline-delimited gene list.")
    parser.add_argument("--max-genes", type=int, default=None, help="Optional cap for smoke tests or panel-limited runs.")
    parser.add_argument("--include-off-tissue", action="store_true", help="Keep all spots instead of only in-tissue spots.")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size for generated prepare_dataset config.")
    parser.add_argument("--graph-k", type=int, default=6, help="KNN graph size for generated prepare_dataset config.")
    parser.add_argument("--chunk-size", type=int, default=128, help="Number of spots to densify per write chunk.")
    args = parser.parse_args()
    result = convert_10x_visium(
        matrix_h5_path=args.matrix_h5,
        spatial_tar_path=args.spatial_tar,
        image_path=args.image,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        scrna_expression_path=args.scrna_expression_path,
        reference_prototypes_path=args.reference_prototypes_path,
        genes_path=args.genes,
        max_genes=args.max_genes,
        in_tissue_only=not args.include_off_tissue,
        patch_size=args.patch_size,
        graph_k=args.graph_k,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
