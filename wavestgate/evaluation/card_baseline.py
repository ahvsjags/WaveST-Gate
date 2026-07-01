"""CARD baseline runner backed by the R package CARD."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import summarize_proportion_metrics
from wavestgate.evaluation.run_baselines import _load_prediction_matrix


def _write_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def _check_card_ready() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("CARD baseline requires Rscript.")
    command = [rscript, "-e", "quit(status = ifelse(requireNamespace('CARD', quietly = TRUE), 0, 1))"]
    if subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("CARD baseline requires the R package 'CARD'.")


def run_card_baseline(
    prepared_path: str | Path,
    spatial_expression_path: str | Path,
    spatial_coords_path: str | Path,
    scrna_expression_path: str | Path,
    scrna_labels_path: str | Path,
    output_dir: str | Path,
    min_count_gene: int = 1,
    min_count_spot: int = 1,
    spot_id_col: str = "spot_id",
    cell_id_col: str = "cell_id",
    cell_type_col: str = "cell_type",
    sample_col: str = "sample_id",
    x_col: str = "x",
    y_col: str = "y",
) -> dict[str, Any]:
    """Run CARD and evaluate its proportions against the prepared benchmark."""

    _check_card_ready()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch, metadata, extra = load_prepared_dataset(prepared_path, device="cpu")
    genes_path = output_dir / "card_genes.txt"
    cell_types_path = output_dir / "card_cell_types.txt"
    _write_list(genes_path, list(metadata.gene_names))
    _write_list(cell_types_path, list(metadata.cell_types))

    r_script = Path(__file__).with_name("card_baseline.R")
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
        "--min-count-gene",
        str(min_count_gene),
        "--min-count-spot",
        str(min_count_spot),
        "--spot-id-col",
        spot_id_col,
        "--cell-id-col",
        cell_id_col,
        "--cell-type-col",
        cell_type_col,
        "--sample-col",
        sample_col,
        "--x-col",
        x_col,
        "--y-col",
        y_col,
    ]
    start = time.perf_counter()
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    runtime_seconds = time.perf_counter() - start
    log_path = output_dir / "card_run.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"CARD failed with exit code {completed.returncode}. See {log_path}")

    predictions_path = output_dir / "card_proportions.csv"
    predictions = _load_prediction_matrix(predictions_path, metadata.spot_ids, metadata.cell_types)
    metrics = summarize_proportion_metrics(predictions, batch.proportion_gt.detach().cpu())
    row: dict[str, Any] = {
        "method": "CARD",
        "runtime_seconds": float(runtime_seconds),
        "peak_cuda_memory_mb": 0.0,
        "device": "R/cpu",
        "predictions_path": str(predictions_path),
    }
    row.update(metrics)

    metrics_path = output_dir / "card_metrics.csv"
    pd.DataFrame([row]).to_csv(metrics_path, index=False)

    manifest_path = output_dir / "card_python_manifest.json"
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
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CARD baseline on a WaveST-Gate benchmark.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--spatial-expression", required=True)
    parser.add_argument("--spatial-coords", required=True)
    parser.add_argument("--scrna-expression", required=True)
    parser.add_argument("--scrna-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-count-gene", type=int, default=1)
    parser.add_argument("--min-count-spot", type=int, default=1)
    parser.add_argument("--spot-id-col", default="spot_id")
    parser.add_argument("--cell-id-col", default="cell_id")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--x-col", default="x")
    parser.add_argument("--y-col", default="y")
    args = parser.parse_args()
    row = run_card_baseline(
        prepared_path=args.prepared,
        spatial_expression_path=args.spatial_expression,
        spatial_coords_path=args.spatial_coords,
        scrna_expression_path=args.scrna_expression,
        scrna_labels_path=args.scrna_labels,
        output_dir=args.output_dir,
        min_count_gene=args.min_count_gene,
        min_count_spot=args.min_count_spot,
        spot_id_col=args.spot_id_col,
        cell_id_col=args.cell_id_col,
        cell_type_col=args.cell_type_col,
        sample_col=args.sample_col,
        x_col=args.x_col,
        y_col=args.y_col,
    )
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
