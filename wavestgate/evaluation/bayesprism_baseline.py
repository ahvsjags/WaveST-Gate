"""BayesPrism baseline runner backed by the R package BayesPrism."""

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


def _check_bayesprism_ready() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("BayesPrism baseline requires Rscript.")
    command = [rscript, "-e", "quit(status = ifelse(requireNamespace('BayesPrism', quietly = TRUE), 0, 1))"]
    if subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("BayesPrism baseline requires the R package 'BayesPrism'.")


def run_bayesprism_baseline(
    prepared_path: str | Path,
    spatial_expression_path: str | Path,
    scrna_expression_path: str | Path,
    scrna_labels_path: str | Path,
    output_dir: str | Path,
    input_type: str = "GEP",
    max_cells_per_type: int | None = None,
    max_spots: int | None = None,
    n_cores: int = 1,
    chain_length: int = 1000,
    burn_in: int = 500,
    thinning: int = 2,
    seed: int = 123,
    alpha: float = 1.0,
    outlier_cut: float = 1.0,
    outlier_fraction: float = 1.0,
    pseudo_min: float = 1e-8,
    optimizer: str = "MLE",
    maxit: int = 100000,
    update_gibbs: bool = True,
    which_theta: str | None = None,
    key: str | None = None,
    spot_id_col: str = "spot_id",
    cell_id_col: str = "cell_id",
    cell_type_col: str = "cell_type",
) -> dict[str, Any]:
    """Run BayesPrism and evaluate its proportions against the prepared benchmark."""

    if input_type not in {"GEP", "count.matrix"}:
        raise ValueError("input_type must be 'GEP' or 'count.matrix'")
    if optimizer not in {"MLE", "MAP"}:
        raise ValueError("optimizer must be 'MLE' or 'MAP'")
    if which_theta is not None and which_theta not in {"first", "final"}:
        raise ValueError("which_theta must be 'first', 'final', or None")
    if which_theta == "final" and not update_gibbs:
        raise ValueError("which_theta='final' requires update_gibbs=True")

    _check_bayesprism_ready()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch, metadata, extra = load_prepared_dataset(prepared_path, device="cpu")
    genes_path = output_dir / "bayesprism_genes.txt"
    cell_types_path = output_dir / "bayesprism_cell_types.txt"
    _write_list(genes_path, list(metadata.gene_names))
    _write_list(cell_types_path, list(metadata.cell_types))

    r_script = Path(__file__).with_name("bayesprism_baseline.R")
    command = [
        shutil.which("Rscript") or "Rscript",
        str(r_script),
        "--spatial-expression",
        str(spatial_expression_path),
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
        "--spot-id-col",
        spot_id_col,
        "--cell-id-col",
        cell_id_col,
        "--cell-type-col",
        cell_type_col,
        "--input-type",
        input_type,
        "--n-cores",
        str(n_cores),
        "--chain-length",
        str(chain_length),
        "--burn-in",
        str(burn_in),
        "--thinning",
        str(thinning),
        "--seed",
        str(seed),
        "--alpha",
        str(alpha),
        "--outlier-cut",
        str(outlier_cut),
        "--outlier-fraction",
        str(outlier_fraction),
        "--pseudo-min",
        str(pseudo_min),
        "--optimizer",
        optimizer,
        "--maxit",
        str(maxit),
        "--update-gibbs",
        "true" if update_gibbs else "false",
    ]
    if which_theta is not None:
        command.extend(["--which-theta", which_theta])
    if key:
        command.extend(["--key", key])
    if max_cells_per_type is not None:
        command.extend(["--max-cells-per-type", str(max_cells_per_type)])
    if max_spots is not None:
        command.extend(["--max-spots", str(max_spots)])

    start = time.perf_counter()
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    runtime_seconds = time.perf_counter() - start
    log_path = output_dir / "bayesprism_run.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"BayesPrism failed with exit code {completed.returncode}. See {log_path}")

    predictions_path = output_dir / "bayesprism_proportions.csv"
    predictions = _load_prediction_matrix(predictions_path, metadata.spot_ids, metadata.cell_types)
    metrics = summarize_proportion_metrics(predictions, batch.proportion_gt.detach().cpu())
    row: dict[str, Any] = {
        "method": "BayesPrism",
        "runtime_seconds": float(runtime_seconds),
        "peak_cuda_memory_mb": 0.0,
        "device": "R/cpu",
        "predictions_path": str(predictions_path),
    }
    row.update(metrics)

    metrics_path = output_dir / "bayesprism_metrics.csv"
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    r_manifest_path = output_dir / "bayesprism_manifest.json"
    manifest = {
        "prepared_path": str(prepared_path),
        "spatial_expression_path": str(spatial_expression_path),
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
        "r_manifest_path": str(r_manifest_path),
        "r_manifest": json.loads(r_manifest_path.read_text(encoding="utf-8")) if r_manifest_path.exists() else None,
    }
    (output_dir / "bayesprism_python_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BayesPrism baseline on a WaveST-Gate benchmark.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--spatial-expression", required=True)
    parser.add_argument("--scrna-expression", required=True)
    parser.add_argument("--scrna-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-type", choices=["GEP", "count.matrix"], default="GEP")
    parser.add_argument("--max-cells-per-type", type=int, default=None)
    parser.add_argument("--max-spots", type=int, default=None)
    parser.add_argument("--n-cores", type=int, default=1)
    parser.add_argument("--chain-length", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--thinning", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--outlier-cut", type=float, default=1.0)
    parser.add_argument("--outlier-fraction", type=float, default=1.0)
    parser.add_argument("--pseudo-min", type=float, default=1e-8)
    parser.add_argument("--optimizer", choices=["MLE", "MAP"], default="MLE")
    parser.add_argument("--maxit", type=int, default=100000)
    parser.add_argument("--update-gibbs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--which-theta", choices=["first", "final"], default=None)
    parser.add_argument("--key", default=None)
    parser.add_argument("--spot-id-col", default="spot_id")
    parser.add_argument("--cell-id-col", default="cell_id")
    parser.add_argument("--cell-type-col", default="cell_type")
    args = parser.parse_args()
    row = run_bayesprism_baseline(
        prepared_path=args.prepared,
        spatial_expression_path=args.spatial_expression,
        scrna_expression_path=args.scrna_expression,
        scrna_labels_path=args.scrna_labels,
        output_dir=args.output_dir,
        input_type=args.input_type,
        max_cells_per_type=args.max_cells_per_type,
        max_spots=args.max_spots,
        n_cores=args.n_cores,
        chain_length=args.chain_length,
        burn_in=args.burn_in,
        thinning=args.thinning,
        seed=args.seed,
        alpha=args.alpha,
        outlier_cut=args.outlier_cut,
        outlier_fraction=args.outlier_fraction,
        pseudo_min=args.pseudo_min,
        optimizer=args.optimizer,
        maxit=args.maxit,
        update_gibbs=args.update_gibbs,
        which_theta=args.which_theta,
        key=args.key,
        spot_id_col=args.spot_id_col,
        cell_id_col=args.cell_id_col,
        cell_type_col=args.cell_type_col,
    )
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
