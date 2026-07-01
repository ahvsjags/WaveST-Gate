"""Tangram baseline runner for WaveST-Gate benchmark datasets."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import summarize_proportion_metrics


def _require_optional_packages() -> tuple[Any, Any]:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("Tangram baseline requires optional package 'anndata'.") from exc
    try:
        import tangram as tg
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("Tangram baseline requires optional package 'tangram-sc'.") from exc
    return ad, tg


def _read_scrna(
    expression_path: str | Path,
    labels_path: str | Path,
    genes: list[str],
    cell_types: list[str],
    cell_id_col: str = "cell_id",
    cell_type_col: str = "cell_type",
    max_cells_per_type: int | None = None,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    expr = pd.read_csv(expression_path)
    labels = pd.read_csv(labels_path)
    missing_genes = [gene for gene in genes if gene not in expr.columns]
    if missing_genes:
        raise ValueError(f"scRNA expression is missing benchmark genes: {missing_genes[:10]}")
    for col, path in [(cell_id_col, expression_path), (cell_id_col, labels_path), (cell_type_col, labels_path)]:
        if col not in (expr.columns if path == expression_path else labels.columns):
            raise ValueError(f"{path} is missing required column '{col}'")

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

    expression = merged[genes].astype(np.float32)
    expression.index = merged[cell_id_col].astype(str).to_numpy()
    cell_type_series = pd.Series(
        merged[cell_type_col].astype(str).to_numpy(),
        index=expression.index,
        name=cell_type_col,
    )
    summary = {
        "num_cells_used": int(expression.shape[0]),
        "num_genes": int(expression.shape[1]),
        "cell_type_counts": {str(k): int(v) for k, v in cell_type_series.value_counts().sort_index().items()},
        "max_cells_per_type": max_cells_per_type,
    }
    return expression, cell_type_series, summary


def _build_anndata(
    prepared_path: str | Path,
    scrna_expression_path: str | Path,
    scrna_labels_path: str | Path,
    device: str,
    cell_id_col: str,
    cell_type_col: str,
    max_cells_per_type: int | None,
    seed: int,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    ad, tg = _require_optional_packages()
    batch, metadata, extra = load_prepared_dataset(prepared_path, device="cpu")
    genes = list(metadata.gene_names)
    cell_types = list(metadata.cell_types)
    sc_expr, sc_types, sc_summary = _read_scrna(
        scrna_expression_path,
        scrna_labels_path,
        genes=genes,
        cell_types=cell_types,
        cell_id_col=cell_id_col,
        cell_type_col=cell_type_col,
        max_cells_per_type=max_cells_per_type,
        seed=seed,
    )
    spatial_x = batch.st_expression.detach().cpu().numpy().astype(np.float32)
    adata_sp = ad.AnnData(
        X=spatial_x,
        obs=pd.DataFrame(index=pd.Index(metadata.spot_ids, name="spot_id")),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    adata_sc = ad.AnnData(
        X=sc_expr.to_numpy(dtype=np.float32, copy=False),
        obs=pd.DataFrame({cell_type_col: sc_types.to_numpy()}, index=pd.Index(sc_expr.index, name=cell_id_col)),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    run_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    manifest = {
        "prepared_path": str(prepared_path),
        "scrna_expression_path": str(scrna_expression_path),
        "scrna_labels_path": str(scrna_labels_path),
        "device": run_device,
        "prepared_extra": extra,
        "num_spots": int(spatial_x.shape[0]),
        "num_genes": int(spatial_x.shape[1]),
        "cell_types": cell_types,
        "scrna_summary": sc_summary,
    }
    return adata_sc, adata_sp, batch, metadata, manifest


def _prediction_frame(adata_sp: Any, spot_ids: list[str], cell_types: list[str], output_key: str) -> pd.DataFrame:
    if output_key not in adata_sp.obsm:
        available = sorted(str(key) for key in adata_sp.obsm.keys())
        raise KeyError(f"Tangram output key '{output_key}' not found. Available obsm keys: {available}")
    raw = adata_sp.obsm[output_key]
    if isinstance(raw, pd.DataFrame):
        frame = raw.copy()
    else:
        frame = pd.DataFrame(np.asarray(raw), index=spot_ids)
    frame.index = frame.index.astype(str)
    frame = frame.reindex(index=spot_ids)
    for cell_type in cell_types:
        if cell_type not in frame.columns:
            frame[cell_type] = 0.0
    frame = frame[cell_types].astype(float).clip(lower=0.0)
    values = frame.to_numpy(dtype=np.float64)
    row_sums = values.sum(axis=1, keepdims=True)
    empty = row_sums[:, 0] <= 1e-12
    if empty.any():
        values[empty, :] = 1.0 / len(cell_types)
        row_sums = values.sum(axis=1, keepdims=True)
    values = values / np.maximum(row_sums, 1e-12)
    return pd.DataFrame(values, index=spot_ids, columns=cell_types)


def _write_training_history(adata_map: Any, output_dir: Path) -> str | None:
    history = getattr(adata_map, "uns", {}).get("training_history")
    if history is None:
        return None
    path = output_dir / "tangram_training_history.csv"
    if isinstance(history, pd.DataFrame):
        history.to_csv(path, index=False)
    elif isinstance(history, dict):
        pd.DataFrame(history).to_csv(path, index=False)
    else:
        pd.DataFrame({"value": list(history)}).to_csv(path, index=False)
    return str(path)


def run_tangram_baseline(
    prepared_path: str | Path,
    scrna_expression_path: str | Path,
    scrna_labels_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    num_epochs: int = 500,
    density_prior: str = "rna_count_based",
    cell_id_col: str = "cell_id",
    cell_type_col: str = "cell_type",
    output_key: str = "tangram_ct_pred",
    max_cells_per_type: int | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    """Run Tangram in cluster mode and evaluate against Xenium spot proportions."""

    _, tg = _require_optional_packages()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adata_sc, adata_sp, batch, metadata, manifest = _build_anndata(
        prepared_path,
        scrna_expression_path,
        scrna_labels_path,
        device=device,
        cell_id_col=cell_id_col,
        cell_type_col=cell_type_col,
        max_cells_per_type=max_cells_per_type,
        seed=seed,
    )
    run_device = manifest["device"]
    manifest.update(
        {
            "method": "Tangram",
            "mode": "clusters",
            "num_epochs": int(num_epochs),
            "density_prior": density_prior,
            "cell_type_col": cell_type_col,
            "output_key": output_key,
            "seed": int(seed),
        }
    )

    if run_device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tg.pp_adatas(adata_sc, adata_sp, genes=list(metadata.gene_names), gene_to_lowercase=False)
    adata_map = tg.map_cells_to_space(
        adata_sc,
        adata_sp,
        mode="clusters",
        cluster_label=cell_type_col,
        device=run_device,
        num_epochs=int(num_epochs),
        density_prior=density_prior,
    )
    tg.project_cell_annotations(adata_map, adata_sp, annotation=cell_type_col)
    runtime_seconds = time.perf_counter() - start
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if run_device == "cuda" and torch.cuda.is_available() else 0.0
    )

    predictions = _prediction_frame(adata_sp, list(metadata.spot_ids), list(metadata.cell_types), output_key=output_key)
    pred_path = output_dir / "tangram_proportions.csv"
    predictions.to_csv(pred_path, index_label="spot_id")
    metrics = summarize_proportion_metrics(
        torch.as_tensor(predictions.to_numpy(dtype=np.float32)),
        batch.proportion_gt.detach().cpu(),
    )
    history_path = _write_training_history(adata_map, output_dir)
    row: dict[str, Any] = {
        "method": "Tangram",
        "runtime_seconds": float(runtime_seconds),
        "peak_cuda_memory_mb": peak_memory_mb,
        "device": run_device,
        "predictions_path": str(pred_path),
    }
    row.update(metrics)

    metrics_path = output_dir / "tangram_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    manifest.update(
        {
            "runtime_seconds": float(runtime_seconds),
            "peak_cuda_memory_mb": peak_memory_mb,
            "predictions_path": str(pred_path),
            "metrics_path": str(metrics_path),
            "training_history_path": history_path,
            "metrics": metrics,
        }
    )
    manifest_path = output_dir / "tangram_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tangram cluster-mode baseline on a prepared WaveST-Gate benchmark.")
    parser.add_argument("--prepared", required=True, help="Prepared dataset with proportion_gt.")
    parser.add_argument("--scrna-expression", required=True, help="scRNA expression table with cell_id and gene columns.")
    parser.add_argument("--scrna-labels", required=True, help="scRNA cell label table.")
    parser.add_argument("--output-dir", required=True, help="Output directory for Tangram predictions and metrics.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--density-prior", default="rna_count_based")
    parser.add_argument("--cell-id-col", default="cell_id")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--output-key", default="tangram_ct_pred")
    parser.add_argument("--max-cells-per-type", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    row = run_tangram_baseline(
        prepared_path=args.prepared,
        scrna_expression_path=args.scrna_expression,
        scrna_labels_path=args.scrna_labels,
        output_dir=args.output_dir,
        device=args.device,
        num_epochs=args.num_epochs,
        density_prior=args.density_prior,
        cell_id_col=args.cell_id_col,
        cell_type_col=args.cell_type_col,
        output_key=args.output_key,
        max_cells_per_type=args.max_cells_per_type,
        seed=args.seed,
    )
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
