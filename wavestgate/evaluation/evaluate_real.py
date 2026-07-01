"""Evaluate real-data WaveST-Gate predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch

from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import summarize_proportion_metrics


def _read_prediction_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "spot_id" in df.columns:
        df = df.set_index("spot_id")
    else:
        df = df.set_index(df.columns[0])
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _tensor_from_aligned(df: pd.DataFrame, spot_ids: list[str], columns: list[str]) -> torch.Tensor:
    missing_spots = [spot_id for spot_id in spot_ids if spot_id not in df.index]
    if missing_spots:
        raise ValueError(f"Prediction/ground-truth table is missing spot ids: {missing_spots[:5]}")
    aligned = df.reindex(index=spot_ids, columns=columns, fill_value=0.0)
    return torch.as_tensor(aligned.values, dtype=torch.float32)


def _uncertainty_from_table(path: str | Path, spot_ids: list[str]) -> torch.Tensor:
    df = _read_prediction_table(path)
    missing_spots = [spot_id for spot_id in spot_ids if spot_id not in df.index]
    if missing_spots:
        raise ValueError(f"Uncertainty table is missing spot ids: {missing_spots[:5]}")
    column = "spot_uncertainty" if "spot_uncertainty" in df.columns else df.columns[0]
    aligned = df.reindex(index=spot_ids)[column]
    return torch.as_tensor(aligned.values, dtype=torch.float32)


def evaluate_real(
    predictions_path: str | Path,
    output_metrics_path: str | Path,
    prepared_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    uncertainty_path: str | Path | None = None,
) -> dict[str, float]:
    """Evaluate predicted proportions against prepared or explicit ground truth."""

    if prepared_path is None and ground_truth_path is None:
        raise ValueError("Either prepared_path or ground_truth_path is required")

    pred_df = _read_prediction_table(predictions_path)
    if prepared_path is not None:
        batch, metadata, _ = load_prepared_dataset(prepared_path)
        spot_ids = metadata.spot_ids
        cell_types = metadata.cell_types
        if ground_truth_path is None:
            if batch.proportion_gt is None:
                raise ValueError("Prepared dataset does not contain proportion_gt; provide ground_truth_path")
            target = batch.proportion_gt.cpu()
        else:
            gt_df = _read_prediction_table(ground_truth_path)
            target = _tensor_from_aligned(gt_df, spot_ids, cell_types)
    else:
        gt_df = _read_prediction_table(ground_truth_path)
        spot_ids = [spot_id for spot_id in gt_df.index if spot_id in pred_df.index]
        if not spot_ids:
            raise ValueError("No overlapping spot ids between predictions and ground truth")
        cell_types = [col for col in gt_df.columns if col in pred_df.columns]
        if not cell_types:
            raise ValueError("No overlapping cell-type columns between predictions and ground truth")
        target = _tensor_from_aligned(gt_df, spot_ids, cell_types)

    predicted = _tensor_from_aligned(pred_df, spot_ids, cell_types)
    uncertainty = _uncertainty_from_table(uncertainty_path, spot_ids) if uncertainty_path is not None else None
    metrics = summarize_proportion_metrics(predicted, target, uncertainty)

    output_metrics_path = Path(output_metrics_path)
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with output_metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WaveST-Gate real-data prediction tables.")
    parser.add_argument("--predictions", required=True, help="Predicted proportions CSV.")
    parser.add_argument("--output-metrics", required=True, help="Output metrics CSV.")
    parser.add_argument("--prepared", default=None, help="Prepared dataset containing metadata and optional proportion_gt.")
    parser.add_argument("--ground-truth", default=None, help="Optional explicit ground-truth proportions CSV.")
    parser.add_argument("--uncertainty", default=None, help="Optional spot-level uncertainty CSV.")
    args = parser.parse_args()
    metrics = evaluate_real(
        predictions_path=args.predictions,
        output_metrics_path=args.output_metrics,
        prepared_path=args.prepared,
        ground_truth_path=args.ground_truth,
        uncertainty_path=args.uncertainty,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
