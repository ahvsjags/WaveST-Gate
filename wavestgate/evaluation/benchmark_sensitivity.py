"""Radius and cell-count confidence sensitivity for Xenium-derived benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from wavestgate.data.ground_truth import count_cells_in_spots, proportions_from_counts
from wavestgate.data.preprocess_st import read_delimited_table
from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import summarize_proportion_metrics


def _load_predictions(path: str | Path, spot_ids: list[str], cell_types: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    missing = [spot_id for spot_id in spot_ids if spot_id not in frame.index]
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} prediction spot ids")
    for cell_type in cell_types:
        if cell_type not in frame.columns:
            frame[cell_type] = 0.0
    return frame.loc[spot_ids, cell_types].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _target_from_proportions(proportions: pd.DataFrame, spot_ids: list[str], cell_types: list[str]) -> torch.Tensor:
    frame = proportions.copy()
    frame.index = frame.index.astype(str)
    for cell_type in cell_types:
        if cell_type not in frame.columns:
            frame[cell_type] = 0.0
    return torch.as_tensor(frame.reindex(spot_ids).fillna(0.0)[cell_types].to_numpy(), dtype=torch.float32)


def run_benchmark_sensitivity(
    prepared_path: str | Path,
    predictions_path: str | Path,
    benchmark_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    radii: list[float] | None = None,
    min_cell_counts: list[int] | None = None,
) -> dict[str, str]:
    """Evaluate existing predictions across radius and Xenium cell-count thresholds."""

    radii = radii or [45.0, 55.0, 65.0, 75.0]
    min_cell_counts = min_cell_counts or [1, 5, 10, 20, 50]
    output_dir = Path(output_dir)
    radius_dir = output_dir / "radius_artifacts"
    radius_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(benchmark_manifest_path).read_text(encoding="utf-8"))
    columns = manifest.get("columns", {})
    cells = read_delimited_table(manifest["cells_path"])
    spots = read_delimited_table(manifest["spots_path"])
    batch, metadata, _ = load_prepared_dataset(prepared_path, device="cpu")
    spot_ids = [str(spot_id) for spot_id in metadata.spot_ids]
    predictions_df = _load_predictions(predictions_path, spot_ids, metadata.cell_types)
    predictions = torch.as_tensor(predictions_df.to_numpy(), dtype=torch.float32)

    rows: list[dict[str, float | int | str]] = []
    radius_rows: list[dict[str, float | int | str]] = []
    for radius in radii:
        counts = count_cells_in_spots(
            cells=cells,
            spots=spots,
            spot_radius=float(radius),
            cell_type_col=columns.get("cell_type_col", "cell_type"),
            cell_x_col=columns.get("cell_x_col", "x"),
            cell_y_col=columns.get("cell_y_col", "y"),
            spot_id_col=columns.get("spot_id_col", "spot_id"),
            spot_x_col=columns.get("spot_x_col", "x"),
            spot_y_col=columns.get("spot_y_col", "y"),
        )
        proportions = proportions_from_counts(counts)
        counts_path = radius_dir / f"xenium_cell_counts_radius{int(radius)}.csv"
        proportions_path = radius_dir / f"xenium_cell_proportions_radius{int(radius)}.csv"
        counts.to_csv(counts_path, index_label="spot_id")
        proportions.to_csv(proportions_path, index_label="spot_id")
        count_totals = counts.sum(axis=1).reindex(spot_ids).fillna(0.0)
        target = _target_from_proportions(proportions, spot_ids, metadata.cell_types)
        radius_rows.append(
            {
                "radius": float(radius),
                "num_spots_with_cells": int((count_totals > 0).sum()),
                "mean_cell_count_on_covered_spots": float(count_totals[count_totals > 0].mean()) if (count_totals > 0).any() else 0.0,
                "median_cell_count_on_covered_spots": float(count_totals[count_totals > 0].median()) if (count_totals > 0).any() else 0.0,
                "counts_path": str(counts_path),
                "proportions_path": str(proportions_path),
            }
        )
        for min_count in min_cell_counts:
            mask = torch.as_tensor((count_totals >= int(min_count)).to_numpy(), dtype=torch.bool)
            metrics = summarize_proportion_metrics(predictions[mask], target[mask])
            rows.append(
                {
                    "radius": float(radius),
                    "min_xenium_cells": int(min_count),
                    "num_spots_passing_threshold": int(mask.sum().item()),
                    **metrics,
                }
            )

    metrics_path = output_dir / "radius_cell_count_sensitivity.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    radius_summary_path = output_dir / "radius_coverage_summary.csv"
    pd.DataFrame(radius_rows).to_csv(radius_summary_path, index=False)
    manifest_payload = {
        "prepared_path": str(prepared_path),
        "predictions_path": str(predictions_path),
        "benchmark_manifest_path": str(benchmark_manifest_path),
        "output_dir": str(output_dir),
        "radii": radii,
        "min_cell_counts": min_cell_counts,
        "metrics_path": str(metrics_path),
        "radius_summary_path": str(radius_summary_path),
        "radius_artifact_dir": str(radius_dir),
        "interpretation": (
            "Sensitivity rows recompute Xenium-derived target proportions at each spot radius and evaluate only spots "
            "meeting a minimum Xenium cell-count threshold. Higher thresholds are confidence filters, not new training data."
        ),
    }
    manifest_path = output_dir / "radius_cell_count_sensitivity_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    markdown_path = output_dir / "radius_cell_count_sensitivity.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Radius And Cell-Count Sensitivity",
                "",
                f"- Metrics: `{metrics_path}`",
                f"- Radius coverage: `{radius_summary_path}`",
                f"- Radii: `{', '.join(str(value) for value in radii)}`",
                f"- Minimum cell-count thresholds: `{', '.join(str(value) for value in min_cell_counts)}`",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "manifest_path": str(manifest_path),
        "metrics_path": str(metrics_path),
        "radius_summary_path": str(radius_summary_path),
        "markdown_path": str(markdown_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run radius/cell-count confidence sensitivity for existing predictions.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radii", nargs="*", type=float, default=[45.0, 55.0, 65.0, 75.0])
    parser.add_argument("--min-cell-counts", nargs="*", type=int, default=[1, 5, 10, 20, 50])
    args = parser.parse_args()
    outputs = run_benchmark_sensitivity(
        prepared_path=args.prepared,
        predictions_path=args.predictions,
        benchmark_manifest_path=args.benchmark_manifest,
        output_dir=args.output_dir,
        radii=args.radii,
        min_cell_counts=args.min_cell_counts,
    )
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
