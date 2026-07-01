"""Aggregate baseline comparisons across independent matched-GT datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRICS = [
    "jsd",
    "spotwise_cosine",
    "mean_celltype_pearson",
    "rmse",
    "num_supervised_spots",
    "runtime_seconds",
    "peak_cuda_memory_mb",
]


def _parse_dataset_comparison(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use DATASET=PATH for each --comparison entry.")
    dataset, path = value.split("=", 1)
    dataset = dataset.strip()
    path = path.strip()
    if not dataset:
        raise argparse.ArgumentTypeError("Dataset name cannot be empty.")
    if not path:
        raise argparse.ArgumentTypeError("Comparison path cannot be empty.")
    return dataset, Path(path)


def _read_comparison(dataset: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "method" not in frame.columns:
        raise ValueError(f"{path} is missing a method column")
    frame = frame.copy()
    frame.insert(0, "dataset", dataset)
    frame["comparison_path"] = str(path)
    for metric in METRICS:
        if metric in frame.columns:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    return frame


def _summarize(dataset_metrics: pd.DataFrame, min_datasets: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for method, group in dataset_metrics.groupby("method", sort=False):
        row: dict[str, float | int | str] = {
            "method": str(method),
            "num_datasets": int(group["dataset"].nunique()),
            "datasets": "|".join(sorted(group["dataset"].astype(str).unique())),
            "included_in_all_required_datasets": bool(group["dataset"].nunique() >= min_datasets),
        }
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_median"] = float(values.median())
        rows.append(row)
    summary = pd.DataFrame(rows)
    if "jsd_mean" in summary.columns:
        summary = summary.sort_values(["jsd_mean", "method"], ascending=[True, True]).reset_index(drop=True)
        summary.insert(0, "rank_by_mean_jsd", np.arange(1, len(summary) + 1))
    return summary


def _paired_dataset_improvement(dataset_metrics: pd.DataFrame, model_method: str) -> pd.DataFrame:
    if model_method not in set(dataset_metrics["method"].astype(str)):
        return pd.DataFrame()
    model = (
        dataset_metrics[dataset_metrics["method"].astype(str) == model_method]
        .set_index("dataset")["jsd"]
        .dropna()
    )
    rows = []
    for method, group in dataset_metrics.groupby("method", sort=False):
        method = str(method)
        if method == model_method:
            continue
        aligned = group.set_index("dataset")["jsd"].dropna().reindex(model.index).dropna()
        model_aligned = model.reindex(aligned.index)
        diff = aligned - model_aligned
        if diff.empty:
            continue
        rows.append(
            {
                "method": method,
                "num_paired_datasets": int(len(diff)),
                "datasets": "|".join(diff.index.astype(str).tolist()),
                "mean_jsd_improvement_vs_wavestgate": float(diff.mean()),
                "std_jsd_improvement_vs_wavestgate": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                "min_jsd_improvement_vs_wavestgate": float(diff.min()),
                "max_jsd_improvement_vs_wavestgate": float(diff.max()),
                "wavestgate_better_fraction": float(np.mean(diff > 0)),
                "dataset_sign_p_baseline_not_worse": float((np.count_nonzero(diff <= 0) + 1) / (len(diff) + 1)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("mean_jsd_improvement_vs_wavestgate", ascending=False).reset_index(drop=True)


def collect_matched_multisample_baselines(
    comparisons: Iterable[tuple[str, str | Path]],
    output_dir: str | Path,
    model_method: str = "WaveST-Gate",
    min_datasets: int | None = None,
) -> dict[str, str]:
    """Collect comparison tables and compute independent dataset mean/std evidence."""

    comparison_pairs = [(dataset, Path(path)) for dataset, path in comparisons]
    if not comparison_pairs:
        raise ValueError("At least one comparison table is required.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    min_datasets = int(min_datasets or len(comparison_pairs))

    dataset_metrics = pd.concat(
        [_read_comparison(dataset, path) for dataset, path in comparison_pairs],
        ignore_index=True,
        sort=False,
    )
    dataset_path = output_dir / "matched_multisample_baseline_dataset_metrics.csv"
    dataset_metrics.to_csv(dataset_path, index=False)

    summary = _summarize(dataset_metrics, min_datasets=min_datasets)
    summary_path = output_dir / "matched_multisample_baseline_summary.csv"
    summary.to_csv(summary_path, index=False)

    paired = _paired_dataset_improvement(dataset_metrics, model_method=model_method)
    paired_path = output_dir / "matched_multisample_baseline_paired_improvement.csv"
    paired.to_csv(paired_path, index=False)

    complete_methods = summary.loc[
        summary.get("num_datasets", pd.Series(dtype=int)) >= min_datasets,
        "method",
    ].astype(str).tolist()
    manifest = {
        "num_datasets": int(len(comparison_pairs)),
        "min_datasets_for_complete_method": int(min_datasets),
        "datasets": [{"dataset": dataset, "comparison_path": str(path)} for dataset, path in comparison_pairs],
        "num_methods": int(summary["method"].nunique()) if "method" in summary.columns else 0,
        "methods_in_all_required_datasets": complete_methods,
        "model_method": model_method,
        "dataset_metrics_path": str(dataset_path),
        "summary_path": str(summary_path),
        "paired_improvement_path": str(paired_path),
        "notes": (
            "Rows summarize independent matched cell-type ground-truth datasets, not bootstrap "
            "replicates. Use this table for formal multi-sample mean/std reporting."
        ),
    }
    manifest_path = output_dir / "matched_multisample_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "dataset_metrics_path": str(dataset_path),
        "summary_path": str(summary_path),
        "paired_improvement_path": str(paired_path),
        "manifest_path": str(manifest_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate matched-GT baseline comparison tables.")
    parser.add_argument(
        "--comparison",
        action="append",
        type=_parse_dataset_comparison,
        required=True,
        help="Dataset comparison in DATASET=PATH form. Repeat for each matched-GT dataset.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-method", default="WaveST-Gate")
    parser.add_argument("--min-datasets", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    outputs = collect_matched_multisample_baselines(
        comparisons=args.comparison,
        output_dir=args.output_dir,
        model_method=args.model_method,
        min_datasets=args.min_datasets,
    )
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
