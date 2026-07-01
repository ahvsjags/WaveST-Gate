"""SPOTlight baseline runner backed by the R package SPOTlight."""

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


def _check_spotlight_ready() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("SPOTlight baseline requires Rscript.")
    command = [rscript, "-e", "quit(status = ifelse(requireNamespace('SPOTlight', quietly = TRUE), 0, 1))"]
    if subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("SPOTlight baseline requires the R package 'SPOTlight'.")


def run_spotlight_baseline(
    prepared_path: str | Path,
    spatial_expression_path: str | Path,
    scrna_expression_path: str | Path,
    scrna_labels_path: str | Path,
    output_dir: str | Path,
    top_markers: int = 25,
    max_cells_per_type: int | None = None,
    nrun: int = 1,
    max_iter: int = 200,
    min_prop: float = 0.0,
    seed: int = 7,
    spot_id_col: str = "spot_id",
    cell_id_col: str = "cell_id",
    cell_type_col: str = "cell_type",
) -> dict[str, Any]:
    """Run SPOTlight and evaluate its proportions against the prepared benchmark."""

    _check_spotlight_ready()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch, metadata, extra = load_prepared_dataset(prepared_path, device="cpu")
    genes_path = output_dir / "spotlight_genes.txt"
    cell_types_path = output_dir / "spotlight_cell_types.txt"
    _write_list(genes_path, list(metadata.gene_names))
    _write_list(cell_types_path, list(metadata.cell_types))

    r_script = Path(__file__).with_name("spotlight_baseline.R")
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
        "--nrun",
        str(nrun),
        "--max-iter",
        str(max_iter),
        "--min-prop",
        str(min_prop),
        "--seed",
        str(seed),
    ]
    if max_cells_per_type is not None:
        command.extend(["--max-cells-per-type", str(max_cells_per_type)])

    start = time.perf_counter()
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    runtime_seconds = time.perf_counter() - start
    log_path = output_dir / "spotlight_run.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"SPOTlight failed with exit code {completed.returncode}. See {log_path}")

    predictions_path = output_dir / "spotlight_proportions.csv"
    predictions = _load_prediction_matrix(predictions_path, metadata.spot_ids, metadata.cell_types)
    metrics = summarize_proportion_metrics(predictions, batch.proportion_gt.detach().cpu())
    row: dict[str, Any] = {
        "method": "SPOTlight",
        "runtime_seconds": float(runtime_seconds),
        "peak_cuda_memory_mb": 0.0,
        "device": "R/cpu",
        "predictions_path": str(predictions_path),
    }
    row.update(metrics)

    metrics_path = output_dir / "spotlight_metrics.csv"
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    r_manifest_path = output_dir / "spotlight_manifest.json"
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
    (output_dir / "spotlight_python_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SPOTlight baseline on a WaveST-Gate benchmark.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--spatial-expression", required=True)
    parser.add_argument("--scrna-expression", required=True)
    parser.add_argument("--scrna-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-markers", type=int, default=25)
    parser.add_argument("--max-cells-per-type", type=int, default=None)
    parser.add_argument("--nrun", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--min-prop", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--spot-id-col", default="spot_id")
    parser.add_argument("--cell-id-col", default="cell_id")
    parser.add_argument("--cell-type-col", default="cell_type")
    args = parser.parse_args()
    row = run_spotlight_baseline(
        prepared_path=args.prepared,
        spatial_expression_path=args.spatial_expression,
        scrna_expression_path=args.scrna_expression,
        scrna_labels_path=args.scrna_labels,
        output_dir=args.output_dir,
        top_markers=args.top_markers,
        max_cells_per_type=args.max_cells_per_type,
        nrun=args.nrun,
        max_iter=args.max_iter,
        min_prop=args.min_prop,
        seed=args.seed,
        spot_id_col=args.spot_id_col,
        cell_id_col=args.cell_id_col,
        cell_type_col=args.cell_type_col,
    )
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
