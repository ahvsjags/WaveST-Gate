"""Worker script for running cell2location in an isolated Python environment.

This file intentionally avoids importing wavestgate so it can be executed by a
separate conda environment that contains cell2location/scvi-tools.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from cell2location.models import Cell2location


def _read_list(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _read_spatial_expression(path: str | Path, spot_id_col: str, genes: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if spot_id_col not in frame.columns:
        raise ValueError(f"{path} is missing required spot id column '{spot_id_col}'")
    missing = [gene for gene in genes if gene not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing benchmark genes: {missing[:10]}")
    expression = frame[[spot_id_col, *genes]].copy()
    expression[spot_id_col] = expression[spot_id_col].astype(str)
    expression = expression.drop_duplicates(spot_id_col).set_index(spot_id_col)
    expression = expression[genes].astype(np.float32)
    return expression


def _build_signatures(
    expression_path: str | Path,
    labels_path: str | Path,
    genes: list[str],
    cell_types: list[str],
    cell_id_col: str,
    cell_type_col: str,
    max_cells_per_type: int | None,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expr = pd.read_csv(expression_path)
    labels = pd.read_csv(labels_path)
    for col, frame, path in [
        (cell_id_col, expr, expression_path),
        (cell_id_col, labels, labels_path),
        (cell_type_col, labels, labels_path),
    ]:
        if col not in frame.columns:
            raise ValueError(f"{path} is missing required column '{col}'")
    missing = [gene for gene in genes if gene not in expr.columns]
    if missing:
        raise ValueError(f"{expression_path} is missing benchmark genes: {missing[:10]}")

    labels = labels[[cell_id_col, cell_type_col]].drop_duplicates(cell_id_col)
    merged = labels.merge(expr[[cell_id_col, *genes]], on=cell_id_col, how="inner")
    merged = merged[merged[cell_type_col].isin(cell_types)].copy()
    if merged.empty:
        raise ValueError("No labelled scRNA cells overlap the benchmark cell types")

    if max_cells_per_type is not None and max_cells_per_type > 0:
        merged = (
            merged.groupby(cell_type_col, group_keys=False)
            .apply(lambda df: df.sample(min(len(df), max_cells_per_type), random_state=seed))
            .reset_index(drop=True)
        )

    missing_cell_types = sorted(set(cell_types) - set(merged[cell_type_col].unique()))
    if missing_cell_types:
        raise ValueError(f"scRNA reference is missing benchmark cell types: {missing_cell_types}")

    counts = merged[cell_type_col].value_counts().reindex(cell_types).fillna(0).astype(int)
    signature_by_type = merged.groupby(cell_type_col)[genes].mean().reindex(cell_types)
    signatures = signature_by_type.transpose().astype(np.float32)
    signatures.index = pd.Index(genes, name="gene")
    signatures.columns = pd.Index(cell_types, name="cell_type")
    signatures = signatures.clip(lower=1e-6)
    summary = {
        "num_cells_used": int(merged.shape[0]),
        "num_genes": int(len(genes)),
        "cell_type_counts": {str(k): int(v) for k, v in counts.items()},
        "max_cells_per_type": max_cells_per_type,
        "signature_mode": "mean_raw_counts_by_cell_type",
    }
    return signatures, summary


def _extract_abundance(adata: ad.AnnData, spot_ids: list[str], cell_types: list[str]) -> pd.DataFrame:
    key = "means_cell_abundance_w_sf"
    if key not in adata.obsm:
        available = sorted(str(k) for k in adata.obsm.keys())
        raise KeyError(f"cell2location posterior key '{key}' not found. Available obsm keys: {available}")
    raw = adata.obsm[key]
    frame = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(np.asarray(raw), index=spot_ids)
    frame.index = frame.index.astype(str)
    frame = frame.reindex(index=spot_ids)

    aligned = pd.DataFrame(index=spot_ids)
    for cell_type in cell_types:
        matches = [
            col
            for col in frame.columns
            if str(col) == cell_type or str(col).endswith(f"_{cell_type}") or str(col).endswith(cell_type)
        ]
        aligned[cell_type] = frame[matches[0]].astype(float) if matches else 0.0
    return aligned.fillna(0.0).clip(lower=0.0)


def _to_proportions(abundance: pd.DataFrame) -> pd.DataFrame:
    values = abundance.to_numpy(dtype=np.float64, copy=True)
    row_sums = values.sum(axis=1, keepdims=True)
    empty = row_sums[:, 0] <= 1e-12
    if empty.any():
        values[empty, :] = 1.0 / values.shape[1]
        row_sums = values.sum(axis=1, keepdims=True)
    values = values / np.maximum(row_sums, 1e-12)
    return pd.DataFrame(values, index=abundance.index, columns=abundance.columns)


def _write_history(model: Cell2location, output_dir: Path) -> str | None:
    history = getattr(model, "history", None)
    if history is None:
        return None
    path = output_dir / "cell2location_training_history.csv"
    try:
        if isinstance(history, pd.DataFrame):
            history.to_csv(path, index=False)
        elif isinstance(history, dict):
            columns = {}
            max_len = 0
            for key, value in history.items():
                array = np.asarray(value).reshape(-1)
                columns[str(key)] = array
                max_len = max(max_len, len(array))
            padded = {key: np.pad(val.astype(float), (0, max_len - len(val)), constant_values=np.nan) for key, val in columns.items()}
            pd.DataFrame(padded).to_csv(path, index=False)
        else:
            pd.DataFrame({"history": np.asarray(history).reshape(-1)}).to_csv(path, index=False)
    except Exception:
        return None
    return str(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    genes = _read_list(args.genes)
    cell_types = _read_list(args.cell_types)

    spatial = _read_spatial_expression(args.spatial_expression, args.spot_id_col, genes)
    signatures, reference_summary = _build_signatures(
        expression_path=args.scrna_expression,
        labels_path=args.scrna_labels,
        genes=genes,
        cell_types=cell_types,
        cell_id_col=args.cell_id_col,
        cell_type_col=args.cell_type_col,
        max_cells_per_type=args.max_cells_per_type,
        seed=args.seed,
    )

    adata = ad.AnnData(
        X=spatial.to_numpy(dtype=np.float32, copy=True),
        obs=pd.DataFrame({"sample": args.sample_name}, index=spatial.index),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    Cell2location.setup_anndata(adata, batch_key="sample")

    use_gpu = args.accelerator == "gpu" and torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    device = 1 if use_gpu else "auto"
    if use_gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model = Cell2location(
        adata,
        cell_state_df=signatures,
        N_cells_per_location=args.n_cells_per_location,
        detection_alpha=args.detection_alpha,
    )
    model.train(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_size=1,
        lr=args.lr,
        accelerator=accelerator,
        device=device,
        enable_progress_bar=args.enable_progress_bar,
        logger=False,
    )
    posterior_kwargs = {
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "accelerator": accelerator,
        "device": device,
    }
    adata = model.export_posterior(adata, sample_kwargs=posterior_kwargs)
    runtime_seconds = time.perf_counter() - start
    peak_cuda_memory_mb = float(torch.cuda.max_memory_allocated() / (1024**2)) if use_gpu else 0.0

    abundance = _extract_abundance(adata, list(spatial.index.astype(str)), cell_types)
    proportions = _to_proportions(abundance)
    abundance_path = output_dir / "cell2location_abundances.csv"
    proportions_path = output_dir / "cell2location_proportions.csv"
    signatures_path = output_dir / "cell2location_reference_signatures.csv"
    abundance.to_csv(abundance_path)
    proportions.to_csv(proportions_path)
    signatures.to_csv(signatures_path)
    history_path = _write_history(model, output_dir)

    manifest = {
        "method": "cell2location",
        "spatial_expression_path": str(args.spatial_expression),
        "scrna_expression_path": str(args.scrna_expression),
        "scrna_labels_path": str(args.scrna_labels),
        "genes_path": str(args.genes),
        "cell_types_path": str(args.cell_types),
        "output_dir": str(output_dir),
        "num_spots": int(spatial.shape[0]),
        "num_genes": int(spatial.shape[1]),
        "cell_types": cell_types,
        "reference_summary": reference_summary,
        "n_cells_per_location": float(args.n_cells_per_location),
        "detection_alpha": float(args.detection_alpha),
        "max_epochs": int(args.max_epochs),
        "batch_size": int(args.batch_size),
        "num_samples": int(args.num_samples),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "device": "cuda" if use_gpu else "cpu",
        "runtime_seconds": float(runtime_seconds),
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "abundance_path": str(abundance_path),
        "proportions_path": str(proportions_path),
        "signatures_path": str(signatures_path),
        "history_path": history_path,
        "versions": {
            "cell2location": _package_version("cell2location"),
            "scvi-tools": _package_version("scvi-tools"),
            "torch": torch.__version__,
            "anndata": _package_version("anndata"),
            "scanpy": _package_version("scanpy"),
        },
    }
    manifest_path = output_dir / "cell2location_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cell2location from an isolated environment.")
    parser.add_argument("--spatial-expression", required=True)
    parser.add_argument("--scrna-expression", required=True)
    parser.add_argument("--scrna-labels", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--cell-types", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spot-id-col", default="spot_id")
    parser.add_argument("--cell-id-col", default="cell_id")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--sample-name", default="sample")
    parser.add_argument("--n-cells-per-location", type=float, default=8.0)
    parser.add_argument("--detection-alpha", type=float, default=20.0)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--max-cells-per-type", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--enable-progress-bar", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
