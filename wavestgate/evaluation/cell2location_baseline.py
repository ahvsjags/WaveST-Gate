"""cell2location baseline runner for WaveST-Gate benchmark datasets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import summarize_proportion_metrics
from wavestgate.evaluation.run_baselines import _load_prediction_matrix


DEFAULT_CELL2LOCATION_PYTHON = Path("/root/miniconda3/envs/cell2loc_env/bin/python")


def _write_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def _default_python() -> str:
    return os.environ.get("CELL2LOCATION_PYTHON", str(DEFAULT_CELL2LOCATION_PYTHON))


def _check_cell2location_ready(python_executable: str) -> None:
    executable = Path(python_executable)
    if not executable.exists():
        raise RuntimeError(
            "cell2location baseline requires an environment Python. "
            f"Expected {executable}; set CELL2LOCATION_PYTHON or pass --cell2location-python."
        )
    command = [
        str(executable),
        "-c",
        "import cell2location, scvi, torch; raise SystemExit(0 if torch.cuda.is_available() else 0)",
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if completed.returncode != 0:
        raise RuntimeError(f"cell2location baseline requires importable cell2location in {executable}.")


def run_cell2location_baseline(
    prepared_path: str | Path,
    spatial_expression_path: str | Path,
    scrna_expression_path: str | Path,
    scrna_labels_path: str | Path,
    output_dir: str | Path,
    cell2location_python: str | None = None,
    max_epochs: int = 1000,
    batch_size: int = 512,
    num_samples: int = 500,
    n_cells_per_location: float = 8.0,
    detection_alpha: float = 20.0,
    lr: float = 0.002,
    max_cells_per_type: int | None = None,
    accelerator: str = "gpu",
    seed: int = 7,
    spot_id_col: str = "spot_id",
    cell_id_col: str = "cell_id",
    cell_type_col: str = "cell_type",
) -> dict[str, Any]:
    """Run cell2location and evaluate its proportions against the prepared benchmark."""

    python_executable = cell2location_python or _default_python()
    _check_cell2location_ready(python_executable)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch, metadata, extra = load_prepared_dataset(prepared_path, device="cpu")

    genes_path = output_dir / "cell2location_genes.txt"
    cell_types_path = output_dir / "cell2location_cell_types.txt"
    _write_list(genes_path, list(metadata.gene_names))
    _write_list(cell_types_path, list(metadata.cell_types))

    worker = Path(__file__).with_name("cell2location_worker.py")
    command = [
        str(python_executable),
        str(worker),
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
        "--n-cells-per-location",
        str(n_cells_per_location),
        "--detection-alpha",
        str(detection_alpha),
        "--max-epochs",
        str(max_epochs),
        "--batch-size",
        str(batch_size),
        "--num-samples",
        str(num_samples),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--accelerator",
        accelerator,
    ]
    if max_cells_per_type is not None:
        command.extend(["--max-cells-per-type", str(max_cells_per_type)])

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "True"
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    log_path = output_dir / "cell2location_run.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"cell2location failed with exit code {completed.returncode}. See {log_path}")

    worker_manifest_path = output_dir / "cell2location_manifest.json"
    worker_manifest = json.loads(worker_manifest_path.read_text(encoding="utf-8"))
    predictions_path = output_dir / "cell2location_proportions.csv"
    predictions = _load_prediction_matrix(predictions_path, metadata.spot_ids, metadata.cell_types)
    metrics = summarize_proportion_metrics(predictions, batch.proportion_gt.detach().cpu())

    row: dict[str, Any] = {
        "method": "cell2location",
        "runtime_seconds": float(worker_manifest.get("runtime_seconds", 0.0)),
        "peak_cuda_memory_mb": float(worker_manifest.get("peak_cuda_memory_mb", 0.0)),
        "device": str(worker_manifest.get("device", "unknown")),
        "predictions_path": str(predictions_path),
    }
    row.update(metrics)
    metrics_path = output_dir / "cell2location_metrics.csv"
    pd.DataFrame([row]).to_csv(metrics_path, index=False)

    manifest = {
        "prepared_path": str(prepared_path),
        "spatial_expression_path": str(spatial_expression_path),
        "scrna_expression_path": str(scrna_expression_path),
        "scrna_labels_path": str(scrna_labels_path),
        "cell2location_python": str(python_executable),
        "output_dir": str(output_dir),
        "prepared_extra": extra,
        "command": command,
        "returncode": completed.returncode,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "log_path": str(log_path),
        "worker_manifest_path": str(worker_manifest_path),
        "worker_manifest": worker_manifest,
    }
    (output_dir / "cell2location_python_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cell2location baseline on a WaveST-Gate benchmark.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--spatial-expression", required=True)
    parser.add_argument("--scrna-expression", required=True)
    parser.add_argument("--scrna-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cell2location-python", default=None)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--n-cells-per-location", type=float, default=8.0)
    parser.add_argument("--detection-alpha", type=float, default=20.0)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--max-cells-per-type", type=int, default=None)
    parser.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--spot-id-col", default="spot_id")
    parser.add_argument("--cell-id-col", default="cell_id")
    parser.add_argument("--cell-type-col", default="cell_type")
    args = parser.parse_args()
    row = run_cell2location_baseline(
        prepared_path=args.prepared,
        spatial_expression_path=args.spatial_expression,
        scrna_expression_path=args.scrna_expression,
        scrna_labels_path=args.scrna_labels,
        output_dir=args.output_dir,
        cell2location_python=args.cell2location_python,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        n_cells_per_location=args.n_cells_per_location,
        detection_alpha=args.detection_alpha,
        lr=args.lr,
        max_cells_per_type=args.max_cells_per_type,
        accelerator=args.accelerator,
        seed=args.seed,
        spot_id_col=args.spot_id_col,
        cell_id_col=args.cell_id_col,
        cell_type_col=args.cell_type_col,
    )
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
