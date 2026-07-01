"""RCTD baseline runner backed by the R package spacexr."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import summarize_proportion_metrics
from wavestgate.evaluation.run_baselines import _load_prediction_matrix


def _write_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def _check_rctd_ready() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("RCTD baseline requires Rscript.")
    command = [rscript, "-e", "quit(status = ifelse(requireNamespace('spacexr', quietly = TRUE), 0, 1))"]
    if subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("RCTD baseline requires the R package 'spacexr'.")


def run_rctd_baseline(
    prepared_path: str | Path,
    spatial_expression_path: str | Path,
    spatial_coords_path: str | Path,
    scrna_expression_path: str | Path,
    scrna_labels_path: str | Path,
    output_dir: str | Path,
    doublet_mode: str = "multi",
    max_cores: int = 8,
    n_max_cells: int = 30000,
    min_umi_reference: int = 1,
    umi_min_spatial: int = 1,
    max_multi_types: int = 4,
    spot_id_col: str = "spot_id",
    cell_id_col: str = "cell_id",
    cell_type_col: str = "cell_type",
    x_col: str = "x",
    y_col: str = "y",
) -> dict[str, Any]:
    """Run RCTD and evaluate its proportions against the prepared benchmark."""

    _check_rctd_ready()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch, metadata, extra = load_prepared_dataset(prepared_path, device="cpu")
    genes_path = output_dir / "rctd_genes.txt"
    cell_types_path = output_dir / "rctd_cell_types.txt"
    _write_list(genes_path, list(metadata.gene_names))
    _write_list(cell_types_path, list(metadata.cell_types))

    r_script = Path(__file__).with_name("rctd_baseline.R")
    command = [
        shutil.which("Rscript") or "Rscript",
        str(r_script),
        "--spatial-expression",
        str(spatial_expression_path),
        "--spatial-coords",
        str(spatial_coords_path),
        "--scrna-expression",
        str(scrna_expression_path),
        "--scrna-labels",
        str(scrna_labels_path),
        "--genes",
        str(genes_path),
        "--cell-types",
        str(cell_types_path),
        "--output-dir",
        str(output_dir),
        "--doublet-mode",
        doublet_mode,
        "--max-cores",
        str(max_cores),
        "--n-max-cells",
        str(n_max_cells),
        "--min-umi-reference",
        str(min_umi_reference),
        "--umi-min-spatial",
        str(umi_min_spatial),
        "--max-multi-types",
        str(max_multi_types),
        "--spot-id-col",
        spot_id_col,
        "--cell-id-col",
        cell_id_col,
        "--cell-type-col",
        cell_type_col,
        "--x-col",
        x_col,
        "--y-col",
        y_col,
    ]
    start = time.perf_counter()
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    runtime_seconds = time.perf_counter() - start
    log_path = output_dir / "rctd_run.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"RCTD failed with exit code {completed.returncode}. See {log_path}")

    predictions_path = output_dir / "rctd_proportions.csv"
    predictions = _load_prediction_matrix(predictions_path, metadata.spot_ids, metadata.cell_types)
    metrics = summarize_proportion_metrics(predictions, batch.proportion_gt.detach().cpu())
    row: dict[str, Any] = {
        "method": f"RCTD ({doublet_mode})",
        "runtime_seconds": float(runtime_seconds),
        "peak_cuda_memory_mb": 0.0,
        "device": "R/cpu",
        "predictions_path": str(predictions_path),
    }
    row.update(metrics)

    metrics_path = output_dir / "rctd_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    manifest_path = output_dir / "rctd_python_manifest.json"
    manifest = {
        "prepared_path": str(prepared_path),
        "spatial_expression_path": str(spatial_expression_path),
        "spatial_coords_path": str(spatial_coords_path),
        "scrna_expression_path": str(scrna_expression_path),
        "scrna_labels_path": str(scrna_labels_path),
        "output_dir": str(output_dir),
        "prepared_extra": extra,
        "command": command,
        "returncode": completed.returncode,
        "runtime_seconds": float(runtime_seconds),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "log_path": str(log_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RCTD/spacexr baseline on a WaveST-Gate benchmark.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--spatial-expression", required=True)
    parser.add_argument("--spatial-coords", required=True)
    parser.add_argument("--scrna-expression", required=True)
    parser.add_argument("--scrna-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--doublet-mode", default="multi", choices=["doublet", "multi", "full"])
    parser.add_argument("--max-cores", type=int, default=8)
    parser.add_argument("--n-max-cells", type=int, default=30000)
    parser.add_argument("--min-umi-reference", type=int, default=1)
    parser.add_argument("--umi-min-spatial", type=int, default=1)
    parser.add_argument("--max-multi-types", type=int, default=4)
    parser.add_argument("--spot-id-col", default="spot_id")
    parser.add_argument("--cell-id-col", default="cell_id")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--x-col", default="x")
    parser.add_argument("--y-col", default="y")
    args = parser.parse_args()
    row = run_rctd_baseline(
        prepared_path=args.prepared,
        spatial_expression_path=args.spatial_expression,
        spatial_coords_path=args.spatial_coords,
        scrna_expression_path=args.scrna_expression,
        scrna_labels_path=args.scrna_labels,
        output_dir=args.output_dir,
        doublet_mode=args.doublet_mode,
        max_cores=args.max_cores,
        n_max_cells=args.n_max_cells,
        min_umi_reference=args.min_umi_reference,
        umi_min_spatial=args.umi_min_spatial,
        max_multi_types=args.max_multi_types,
        spot_id_col=args.spot_id_col,
        cell_id_col=args.cell_id_col,
        cell_type_col=args.cell_type_col,
        x_col=args.x_col,
        y_col=args.y_col,
    )
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
