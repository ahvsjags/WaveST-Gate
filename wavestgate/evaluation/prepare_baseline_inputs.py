"""Create baseline-compatible input bundles for WaveST-Gate benchmarks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import io as scipy_io
from scipy import sparse

from wavestgate.data.preprocess_st import read_delimited_table


BASELINE_METHODS = ["cell2location", "RCTD", "CARD", "Tangram", "SPOTlight", "BayesPrism", "SpatialDWLS", "SpatialDWLS/Seurat"]


@dataclass
class DatasetBundle:
    dataset_id: str
    dataset_dir: Path
    role: str = "test"


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_dataset_bundles(path: str | Path | None, dataset_dirs: list[str] | None) -> list[DatasetBundle]:
    bundles: list[DatasetBundle] = []
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        for item in loaded.get("datasets", []):
            bundles.append(
                DatasetBundle(
                    dataset_id=str(item["dataset_id"]),
                    dataset_dir=Path(item["dataset_dir"]),
                    role=str(item.get("role", "test")),
                )
            )
    for dataset_dir in dataset_dirs or []:
        path_obj = Path(dataset_dir)
        manifest = _load_json(path_obj / "baseline_manifest.json")
        bundles.append(DatasetBundle(dataset_id=str(manifest["dataset_id"]), dataset_dir=path_obj, role="test"))
    if not bundles:
        raise ValueError("At least one dataset must be provided")
    return bundles


def _read_expression_header(path: str | Path) -> tuple[list[str], list[str]]:
    df = read_delimited_table(path)
    if "spot_id" not in df.columns:
        raise ValueError(f"Expression table is missing spot_id: {path}")
    return df["spot_id"].astype(str).tolist(), [str(col) for col in df.columns if col != "spot_id"]


def _write_sparse_mtx(path: Path, matrix) -> None:
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        scipy_io.mmwrite(handle, matrix)


def _write_tsv(path: Path, values: list[str], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({header: values}).to_csv(path, sep="\t", index=False)


def _method_filename(method: str) -> str:
    return method.replace("/", "_").replace(" ", "_")


def _create_spatial_baseline_inputs(dataset_id: str, dataset_manifest: dict[str, Any], output_dir: Path) -> dict[str, str]:
    expression = read_delimited_table(dataset_manifest["spot_expression_path"])
    if "spot_id" not in expression.columns:
        raise ValueError(f"Expression table is missing spot_id: {dataset_manifest['spot_expression_path']}")
    spot_ids = expression["spot_id"].astype(str).tolist()
    gene_names = [str(col) for col in expression.columns if col != "spot_id"]
    values = expression.drop(columns=["spot_id"]).apply(pd.to_numeric, errors="raise").values

    coords = read_delimited_table(dataset_manifest["spot_coords_path"])
    if "spot_id" not in coords.columns:
        raise ValueError(f"Coordinate table is missing spot_id: {dataset_manifest['spot_coords_path']}")
    coords["spot_id"] = coords["spot_id"].astype(str)
    coords = coords.set_index("spot_id").loc[spot_ids].reset_index()

    shared = output_dir / "baseline_inputs"
    shared.mkdir(parents=True, exist_ok=True)
    spatial_mtx = shared / "spatial_gene_by_spot.mtx.gz"
    spots_tsv = shared / "spatial_spots.tsv"
    genes_tsv = shared / "spatial_genes.tsv"
    coords_csv = shared / "spatial_coords.csv"
    _write_sparse_mtx(spatial_mtx, sparse.csr_matrix(values.T))
    _write_tsv(spots_tsv, spot_ids, "spot_id")
    _write_tsv(genes_tsv, gene_names, "gene_name")
    coords.to_csv(coords_csv, index=False)

    manifest = {
        "dataset_id": dataset_id,
        "spatial_mtx_path": str(spatial_mtx),
        "spatial_spots_path": str(spots_tsv),
        "spatial_genes_path": str(genes_tsv),
        "spatial_coords_path": str(coords_csv),
        "spot_expression_path": dataset_manifest["spot_expression_path"],
        "spot_coords_path": dataset_manifest["spot_coords_path"],
        "raw_matrix_h5_path": dataset_manifest.get("raw_matrix_h5_path"),
        "raw_spatial_tar_path": dataset_manifest.get("raw_spatial_tar_path"),
        "image_path": dataset_manifest.get("image_path"),
        "num_spots": len(spot_ids),
        "num_genes": len(gene_names),
    }
    manifest_path = shared / "spatial_baseline_inputs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {key: str(value) for key, value in manifest.items() if key.endswith("_path")}


def _make_splits(spot_ids: list[str], mode: str, seed: int, train_fraction: float, val_fraction: float) -> pd.DataFrame:
    if mode == "all_test":
        return pd.DataFrame({"spot_id": spot_ids, "split": ["test"] * len(spot_ids)})
    if mode != "random":
        raise ValueError(f"Unsupported split mode: {mode}")
    if train_fraction < 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction and val_fraction must be non-negative and sum to < 1")
    rng = np.random.default_rng(seed)
    spot_ids_arr = np.asarray(spot_ids, dtype=object)
    order = rng.permutation(len(spot_ids_arr))
    n_train = int(round(len(spot_ids_arr) * train_fraction))
    n_val = int(round(len(spot_ids_arr) * val_fraction))
    split = np.full(len(spot_ids_arr), "test", dtype=object)
    split[order[:n_train]] = "train"
    split[order[n_train : n_train + n_val]] = "val"
    return pd.DataFrame({"spot_id": spot_ids_arr.tolist(), "split": split.tolist()})


def _method_config(
    method: str,
    dataset_id: str,
    dataset_manifest: dict[str, Any],
    reference_manifest: dict[str, Any],
    split_path: Path,
    spatial_inputs: dict[str, str],
) -> dict[str, Any]:
    return {
        "method": method,
        "dataset_id": dataset_id,
        "spatial": {
            "spot_expression_path": dataset_manifest["spot_expression_path"],
            "spot_coords_path": dataset_manifest["spot_coords_path"],
            "raw_matrix_h5_path": dataset_manifest.get("raw_matrix_h5_path"),
            "raw_spatial_tar_path": dataset_manifest.get("raw_spatial_tar_path"),
            "image_path": dataset_manifest.get("image_path"),
            "genes_path": dataset_manifest["genes_path"],
            "split_path": str(split_path),
        },
        "reference": {
            "raw_matrix_h5_path": reference_manifest.get("raw_matrix_h5_path"),
            "cell_labels_path": reference_manifest["cell_labels_path"],
            "cell_expression_path": reference_manifest.get("cell_expression_path"),
            "reference_prototypes_path": reference_manifest.get("reference_prototypes_path"),
            "reference_mtx_path": reference_manifest.get("reference_mtx_path"),
            "reference_cells_path": reference_manifest.get("reference_cells_path"),
            "reference_genes_path": reference_manifest.get("reference_genes_path"),
            "genes_path": reference_manifest["genes_path"],
            "cell_type_counts_path": reference_manifest.get("cell_type_counts_path"),
        },
        "matrix_market_inputs": {
            "spatial_mtx_path": spatial_inputs.get("spatial_mtx_path"),
            "spatial_spots_path": spatial_inputs.get("spatial_spots_path"),
            "spatial_genes_path": spatial_inputs.get("spatial_genes_path"),
            "spatial_coords_path": spatial_inputs.get("spatial_coords_path"),
            "reference_mtx_path": reference_manifest.get("reference_mtx_path"),
            "reference_cells_path": reference_manifest.get("reference_cells_path"),
            "reference_genes_path": reference_manifest.get("reference_genes_path"),
        },
        "notes": _baseline_method_notes(method),
    }


def _baseline_method_notes(method: str) -> str:
    notes = {
        "cell2location": "Create AnnData from spatial raw_matrix_h5_path and scRNA raw_matrix_h5_path; join reference cell_labels_path before model training.",
        "RCTD": "Create SpatialRNA from spot_expression_path/spot_coords_path and Reference from scRNA raw_matrix_h5_path plus cell_labels_path.",
        "CARD": "Use spot_expression_path/spot_coords_path as spatial count and coordinates, and scRNA raw_matrix_h5_path plus cell_labels_path as reference.",
        "Tangram": "Create AnnData objects for spatial and scRNA, subset both to genes_path, then map cells to spots.",
        "SPOTlight": "Use spot_expression_path as spatial count matrix and scRNA cell_expression_path plus cell_labels_path as reference for seeded NMF deconvolution.",
        "BayesPrism": "Use spot_expression_path as sample-by-gene mixture and scRNA cell_expression_path plus cell_labels_path as the reference; GEP mode aggregates the same reference to cell-type profiles.",
        "SpatialDWLS": "Use spot_expression_path as spatial expression and scRNA cell_expression_path plus cell_labels_path to build a DWLS signature matrix for quadprog-based spatial deconvolution.",
        "SpatialDWLS/Seurat": "Use the Giotto/Seurat package stack through Giotto::runDWLSDeconv with the same spot_expression_path, scRNA cell_expression_path, cell_labels_path, genes, and split.",
    }
    return notes[method]


def create_baseline_bundle(
    datasets_manifest: str | Path | None,
    dataset_dirs: list[str] | None,
    reference_manifest_path: str | Path,
    output_dir: str | Path,
    split_mode: str = "all_test",
    seed: int = 17,
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
) -> dict[str, Any]:
    """Create split tables, method configs, and an alignment report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundles = _load_dataset_bundles(datasets_manifest, dataset_dirs)
    reference_manifest = _load_json(reference_manifest_path)
    required_reference_keys = ["raw_matrix_h5_path", "cell_labels_path", "genes_path"]
    missing_reference = [key for key in required_reference_keys if not reference_manifest.get(key)]
    if missing_reference:
        raise ValueError(f"Reference manifest is missing required keys: {missing_reference}")
    reference_genes = set(pd.read_csv(reference_manifest["genes_path"])["gene_name"].astype(str))

    registry_rows = []
    checks = []
    master_split_rows = []
    for bundle in bundles:
        dataset_manifest_path = bundle.dataset_dir / "baseline_manifest.json"
        dataset_manifest = _load_json(dataset_manifest_path)
        spot_ids, expression_genes = _read_expression_header(dataset_manifest["spot_expression_path"])
        coords = read_delimited_table(dataset_manifest["spot_coords_path"])
        if "spot_id" not in coords.columns:
            raise ValueError(f"Coordinate table is missing spot_id: {dataset_manifest['spot_coords_path']}")
        coords_spots = coords["spot_id"].astype(str).tolist()
        missing_coords = sorted(set(spot_ids) - set(coords_spots))
        if missing_coords:
            raise ValueError(f"{bundle.dataset_id} has expression spots missing coordinates: {missing_coords[:5]}")
        gene_overlap = len(set(expression_genes) & reference_genes)
        if gene_overlap != len(expression_genes):
            missing_genes = sorted(set(expression_genes) - reference_genes)
            raise ValueError(f"{bundle.dataset_id} has genes missing from reference: {missing_genes[:10]}")

        dataset_out = output / "datasets" / bundle.dataset_id
        configs_out = dataset_out / "method_configs"
        configs_out.mkdir(parents=True, exist_ok=True)
        spatial_inputs = _create_spatial_baseline_inputs(bundle.dataset_id, dataset_manifest, dataset_out)
        split = _make_splits(spot_ids, mode=split_mode, seed=seed, train_fraction=train_fraction, val_fraction=val_fraction)
        split_path = dataset_out / "spot_splits.csv"
        split.to_csv(split_path, index=False)
        for row in split.itertuples(index=False):
            master_split_rows.append({"dataset_id": bundle.dataset_id, "spot_id": row.spot_id, "split": row.split, "role": bundle.role})

        for method in BASELINE_METHODS:
            config = _method_config(method, bundle.dataset_id, dataset_manifest, reference_manifest, split_path, spatial_inputs)
            (configs_out / f"{_method_filename(method)}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        registry_rows.append(
            {
                "dataset_id": bundle.dataset_id,
                "role": bundle.role,
                "dataset_dir": str(bundle.dataset_dir),
                "spot_expression_path": dataset_manifest["spot_expression_path"],
                "spot_coords_path": dataset_manifest["spot_coords_path"],
                "prepared_path": str(bundle.dataset_dir / "prepared.pt"),
                "num_spots": len(spot_ids),
                "num_genes": len(expression_genes),
                "split_path": str(split_path),
                "spatial_mtx_path": spatial_inputs["spatial_mtx_path"],
            }
        )
        checks.append(
            {
                "dataset_id": bundle.dataset_id,
                "status": "ok",
                "num_spots": len(spot_ids),
                "num_coords": len(coords_spots),
                "num_genes": len(expression_genes),
                "reference_gene_overlap": gene_overlap,
                "methods": BASELINE_METHODS,
            }
        )

    registry_path = output / "benchmark_registry.csv"
    pd.DataFrame(registry_rows).to_csv(registry_path, index=False)
    split_path = output / "master_spot_splits.csv"
    pd.DataFrame(master_split_rows).to_csv(split_path, index=False)
    report = {
        "reference_manifest_path": str(reference_manifest_path),
        "reference": {
            "raw_matrix_h5_path": reference_manifest.get("raw_matrix_h5_path"),
            "cell_labels_path": reference_manifest["cell_labels_path"],
            "cell_expression_path": reference_manifest.get("cell_expression_path"),
            "reference_prototypes_path": reference_manifest.get("reference_prototypes_path"),
            "reference_mtx_path": reference_manifest.get("reference_mtx_path"),
            "reference_cells_path": reference_manifest.get("reference_cells_path"),
            "reference_genes_path": reference_manifest.get("reference_genes_path"),
            "num_labelled_cells": reference_manifest.get("num_labelled_cells"),
            "num_cell_types": reference_manifest.get("num_cell_types"),
            "num_genes": reference_manifest.get("num_genes"),
        },
        "split_mode": split_mode,
        "seed": seed,
        "datasets": checks,
        "registry_path": str(registry_path),
        "master_split_path": str(split_path),
    }
    report_path = output / "baseline_bundle_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create baseline-compatible benchmark bundles.")
    parser.add_argument("--datasets-manifest", default=None, help="YAML manifest with dataset_id, dataset_dir, and role entries.")
    parser.add_argument("--dataset-dir", action="append", default=None, help="Processed dataset directory containing baseline_manifest.json.")
    parser.add_argument("--reference-manifest", required=True, help="scRNA reference baseline_scrna_reference.json path.")
    parser.add_argument("--output-dir", required=True, help="Output bundle directory.")
    parser.add_argument("--split-mode", choices=["all_test", "random"], default="all_test", help="Spot split strategy.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for random split mode.")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Train fraction for random split mode.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction for random split mode.")
    args = parser.parse_args()
    report = create_baseline_bundle(
        datasets_manifest=args.datasets_manifest,
        dataset_dirs=args.dataset_dir,
        reference_manifest_path=args.reference_manifest,
        output_dir=args.output_dir,
        split_mode=args.split_mode,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
