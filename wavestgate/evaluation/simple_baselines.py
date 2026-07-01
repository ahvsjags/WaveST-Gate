"""Lightweight deconvolution baselines that run without optional bio packages."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import summarize_proportion_metrics


def uniform_baseline(num_spots: int, num_cell_types: int, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.full((num_spots, num_cell_types), 1.0 / num_cell_types, device=device)


def cosine_reference_baseline(expression: torch.Tensor, reference: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    spot = F.normalize(torch.log1p(expression.clamp_min(0.0)), dim=1)
    ref = F.normalize(torch.log1p(reference.clamp_min(0.0)), dim=1)
    logits = spot @ ref.transpose(0, 1) / max(float(temperature), 1e-6)
    return logits.softmax(dim=1)


def nnls_reference_baseline(expression: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Non-negative least-squares deconvolution against reference prototypes."""

    from scipy.optimize import nnls

    expr = torch.log1p(expression.clamp_min(0.0)).detach().cpu().numpy()
    ref = torch.log1p(reference.clamp_min(0.0)).detach().cpu().numpy().T
    rows = []
    for row in expr:
        weights, _ = nnls(ref, row)
        total = weights.sum()
        if total <= 1e-12:
            weights = np.full(reference.size(0), 1.0 / reference.size(0), dtype=float)
        else:
            weights = weights / total
        rows.append(weights)
    return torch.as_tensor(np.vstack(rows), dtype=torch.float32, device=expression.device)


BASELINES = {
    "uniform": uniform_baseline,
    "reference_cosine": cosine_reference_baseline,
    "reference_nnls": nnls_reference_baseline,
}


def run_simple_baselines(
    prepared_path: str | Path,
    output_dir: str | Path,
    methods: list[str] | None = None,
    device: str = "cpu",
) -> list[dict[str, float | str]]:
    run_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    batch, metadata, extra = load_prepared_dataset(prepared_path, device=run_device)
    if batch.proportion_gt is None:
        raise ValueError("Prepared dataset must contain proportion_gt for baseline evaluation")

    methods = methods or list(BASELINES)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    for method in methods:
        if method not in BASELINES:
            raise ValueError(f"Unknown simple baseline: {method}")
        start = time.perf_counter()
        if method == "uniform":
            predicted = uniform_baseline(batch.st_expression.size(0), batch.reference_prototypes.size(0), run_device)
        else:
            predicted = BASELINES[method](batch.st_expression, batch.reference_prototypes)
        elapsed = time.perf_counter() - start
        metrics = summarize_proportion_metrics(predicted, batch.proportion_gt)
        pred_path = output_dir / f"{method}_proportions.csv"
        pd.DataFrame(
            predicted.detach().cpu().numpy(),
            index=metadata.spot_ids,
            columns=metadata.cell_types,
        ).to_csv(pred_path, index_label="spot_id")
        row: dict[str, float | str] = {
            "method": method,
            "runtime_seconds": float(elapsed),
            "device": str(run_device),
            "predictions_path": str(pred_path),
        }
        row.update(metrics)
        rows.append(row)

    metrics_path = output_dir / "simple_baseline_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "prepared_path": str(prepared_path),
        "output_dir": str(output_dir),
        "methods": methods,
        "dataset_extra": extra,
        "metrics_path": str(metrics_path),
    }
    (output_dir / "simple_baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight deconvolution baselines.")
    parser.add_argument("--prepared", required=True, help="Prepared dataset with proportion_gt.")
    parser.add_argument("--output-dir", required=True, help="Output directory for predictions and metrics.")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional subset of methods.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    rows = run_simple_baselines(args.prepared, args.output_dir, methods=args.methods, device=args.device)
    for row in rows:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
