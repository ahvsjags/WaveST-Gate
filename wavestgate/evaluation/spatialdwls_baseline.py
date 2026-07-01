"""SpatialDWLS baseline runner using a standalone quadprog DWLS implementation."""

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


def _check_spatialdwls_ready() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("SpatialDWLS baseline requires Rscript.")
    command = [
        rscript,
        "-e",
        "quit(status = ifelse(requireNamespace('quadprog', quietly = TRUE) && requireNamespace('Matrix', quietly = TRUE), 0, 1))",
    ]
    if subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("SpatialDWLS baseline requires R packages 'quadprog' and 'Matrix'.")


def run_spatialdwls_baseline(
    prepared_path: str | Path,
    spatial_expression_path: str | Path,
    scrna_expression_path: str | Path,
    scrna_labels_path: str | Path,
    output_dir: str | Path,
    top_markers: int = 25,
    n_cell: float = 50.0,
    dampening_j: float = 2.0,
    max_iter: int = 100,
    tol: float = 0.01,
    pseudo_count: float = 1e-6,
    eps: float = 1e-8,
    max_spots: int | None = None,
    spot_id_col: str = "spot_id",
    cell_id_col: str = "cell_id",
    cell_type_col: str = "cell_type",
) -> dict[str, Any]:
    """Run SpatialDWLS and evaluate its proportions against the prepared benchmark."""

    _check_spatialdwls_ready()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch, metadata, extra = load_prepared_dataset(prepared_path, device="cpu")
    genes_path = output_dir / "spatialdwls_genes.txt"
    cell_types_path = output_dir / "spatialdwls_cell_types.txt"
    _write_list(genes_path, list(metadata.gene_names))
    _write_list(cell_types_path, list(metadata.cell_types))

    r_script = Path(__file__).with_name("spatialdwls_baseline.R")
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
        "--top-markers",
        str(top_markers),
        "--n-cell",
        str(n_cell),
        "--dampening-j",
        str(dampening_j),
        "--max-iter",
        str(max_iter),
        "--tol",
        str(tol),
        "--pseudo-count",
        str(pseudo_count),
        "--eps",
        str(eps),
    ]
    if max_spots is not None:
        command.extend(["--max-spots", str(max_spots)])

    start = time.perf_counter()
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    runtime_seconds = time.perf_counter() - start
    log_path = output_dir / "spatialdwls_run.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"SpatialDWLS failed with exit code {completed.returncode}. See {log_path}")

    predictions_path = output_dir / "spatialdwls_proportions.csv"
    predictions = _load_prediction_matrix(predictions_path, metadata.spot_ids, metadata.cell_types)
    metrics = summarize_proportion_metrics(predictions, batch.proportion_gt.detach().cpu())
    row: dict[str, Any] = {
        "method": "SpatialDWLS",
        "runtime_seconds": float(runtime_seconds),
        "peak_cuda_memory_mb": 0.0,
        "device": "R/cpu",
        "predictions_path": str(predictions_path),
    }
    row.update(metrics)

    metrics_path = output_dir / "spatialdwls_metrics.csv"
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    r_manifest_path = output_dir / "spatialdwls_manifest.json"
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
    (output_dir / "spatialdwls_python_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SpatialDWLS baseline on a WaveST-Gate benchmark.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--spatial-expression", required=True)
    parser.add_argument("--scrna-expression", required=True)
    parser.add_argument("--scrna-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-markers", type=int, default=25)
    parser.add_argument("--n-cell", type=float, default=50.0)
    parser.add_argument("--dampening-j", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--pseudo-count", type=float, default=1e-6)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-spots", type=int, default=None)
    parser.add_argument("--spot-id-col", default="spot_id")
    parser.add_argument("--cell-id-col", default="cell_id")
    parser.add_argument("--cell-type-col", default="cell_type")
    args = parser.parse_args()
    row = run_spatialdwls_baseline(
        prepared_path=args.prepared,
        spatial_expression_path=args.spatial_expression,
        scrna_expression_path=args.scrna_expression,
        scrna_labels_path=args.scrna_labels,
        output_dir=args.output_dir,
        top_markers=args.top_markers,
        n_cell=args.n_cell,
        dampening_j=args.dampening_j,
        max_iter=args.max_iter,
        tol=args.tol,
        pseudo_count=args.pseudo_count,
        eps=args.eps,
        max_spots=args.max_spots,
        spot_id_col=args.spot_id_col,
        cell_id_col=args.cell_id_col,
        cell_type_col=args.cell_type_col,
    )
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
