"""Split and bootstrap statistics for baseline comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import summarize_proportion_metrics


METRIC_COLUMNS = ["spotwise_cosine", "mean_celltype_pearson", "rmse", "jsd", "num_supervised_spots"]


def _read_prediction_table(path: str | Path, spot_ids: list[str], cell_types: list[str]) -> torch.Tensor:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    for cell_type in cell_types:
        if cell_type not in frame.columns:
            frame[cell_type] = 0.0
    values = frame.reindex(index=spot_ids, columns=cell_types).fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32)
    totals = values.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= 1e-12
    if empty.any():
        values[empty, :] = 1.0 / len(cell_types)
        totals = values.sum(axis=1, keepdims=True)
    values = values / np.maximum(totals, 1e-12)
    return torch.as_tensor(values, dtype=torch.float32)


def _load_methods(comparison_path: str | Path) -> pd.DataFrame:
    comparison = pd.read_csv(comparison_path)
    required = {"method", "predictions_path"}
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(f"Baseline comparison table is missing columns: {sorted(missing)}")
    comparison = comparison.copy()
    comparison["method"] = comparison["method"].astype(str)
    comparison["predictions_path"] = comparison["predictions_path"].astype(str)
    comparison = comparison[comparison["predictions_path"].str.len() > 0]
    return comparison


def _load_split_masks(
    split_path: str | Path | None,
    spot_ids: list[str],
    supervised_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    masks = {"all_supervised": supervised_mask.detach().cpu().clone()}
    if split_path is None or not Path(split_path).exists():
        return masks
    split_frame = pd.read_csv(split_path)
    if "spot_id" not in split_frame.columns or "split" not in split_frame.columns:
        return masks
    split_frame["spot_id"] = split_frame["spot_id"].astype(str)
    split_by_spot = split_frame.set_index("spot_id")["split"].reindex(spot_ids)
    for split_name in sorted(split_by_spot.dropna().astype(str).unique()):
        split_mask = torch.as_tensor((split_by_spot.astype(str) == split_name).to_numpy(), dtype=torch.bool)
        mask = supervised_mask.detach().cpu() & split_mask
        if bool(mask.sum()):
            masks[f"manifest_{split_name}"] = mask
    return masks


def _metric_row(method: str, cohort: str, pred: torch.Tensor, target: torch.Tensor, selector: torch.Tensor, replicate: int | None = None) -> dict[str, float | str | int]:
    metrics = summarize_proportion_metrics(pred[selector], target[selector])
    row: dict[str, float | str | int] = {"method": method, "cohort": cohort}
    if replicate is not None:
        row["replicate"] = replicate
    row.update(metrics)
    return row


def _bootstrap_indices(supervised_mask: torch.Tensor, n_bootstraps: int, seed: int) -> list[torch.Tensor]:
    supervised_indices = torch.where(supervised_mask.detach().cpu())[0].numpy()
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(n_bootstraps)):
        sampled = rng.choice(supervised_indices, size=len(supervised_indices), replace=True)
        samples.append(torch.as_tensor(sampled, dtype=torch.long))
    return samples


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in metrics.groupby("method"):
        row: dict[str, float | str] = {"method": method, "num_cohorts": float(group["cohort"].nunique()), "num_rows": float(len(group))}
        for metric in METRIC_COLUMNS:
            if metric not in group.columns:
                continue
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{metric}_median"] = float(group[metric].median())
        rows.append(row)
    summary = pd.DataFrame(rows)
    if "jsd_mean" in summary.columns:
        summary = summary.sort_values("jsd_mean", ascending=True).reset_index(drop=True)
        summary.insert(0, "rank_by_mean_jsd", np.arange(1, len(summary) + 1))
    return summary


def _paired_bootstrap_against_model(metrics: pd.DataFrame, model_method: str = "WaveST-Gate") -> pd.DataFrame:
    bootstrap = metrics[metrics["cohort"].str.startswith("bootstrap_")].copy()
    if bootstrap.empty or model_method not in set(bootstrap["method"]):
        return pd.DataFrame()
    model = bootstrap[bootstrap["method"] == model_method].set_index("cohort")["jsd"]
    rows = []
    for method, group in bootstrap.groupby("method"):
        if method == model_method:
            continue
        aligned = group.set_index("cohort")["jsd"].reindex(model.index).dropna()
        model_aligned = model.reindex(aligned.index)
        diff = aligned - model_aligned
        if diff.empty:
            continue
        rows.append(
            {
                "method": method,
                "bootstrap_replicates": int(len(diff)),
                "mean_jsd_improvement_vs_wavestgate": float(diff.mean()),
                "std_jsd_improvement_vs_wavestgate": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                "ci95_low": float(diff.quantile(0.025)),
                "ci95_high": float(diff.quantile(0.975)),
                "bootstrap_p_baseline_not_worse": float((np.count_nonzero(diff <= 0) + 1) / (len(diff) + 1)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_jsd_improvement_vs_wavestgate", ascending=False)


def collect_baseline_statistics(
    prepared_path: str | Path,
    comparison_path: str | Path,
    output_dir: str | Path,
    split_path: str | Path | None = None,
    n_bootstraps: int = 200,
    seed: int = 2026,
) -> dict[str, str]:
    """Compute split-wise and bootstrap mean/std evidence for all baseline predictions."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch, metadata, _ = load_prepared_dataset(prepared_path, device="cpu")
    if batch.proportion_gt is None:
        raise ValueError("Baseline statistics require prepared proportion_gt")
    target = batch.proportion_gt.detach().cpu()
    supervised_mask = target.sum(dim=1) > 1e-8
    methods = _load_methods(comparison_path)
    predictions = {
        str(row["method"]): _read_prediction_table(row["predictions_path"], metadata.spot_ids, metadata.cell_types)
        for _, row in methods.iterrows()
    }

    metric_rows: list[dict[str, float | str | int]] = []
    split_masks = _load_split_masks(split_path, metadata.spot_ids, supervised_mask)
    for method, pred in predictions.items():
        for cohort, mask in split_masks.items():
            metric_rows.append(_metric_row(method, cohort, pred, target, mask))

    bootstrap_samples = _bootstrap_indices(supervised_mask, n_bootstraps=n_bootstraps, seed=seed)
    for replicate, sample_idx in enumerate(bootstrap_samples):
        cohort = f"bootstrap_{replicate:04d}"
        for method, pred in predictions.items():
            metric_rows.append(_metric_row(method, cohort, pred, target, sample_idx, replicate=replicate))

    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "baseline_split_bootstrap_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    summary = _summarize_metrics(metrics)
    summary_path = output_dir / "baseline_split_bootstrap_summary.csv"
    summary.to_csv(summary_path, index=False)
    paired = _paired_bootstrap_against_model(metrics)
    paired_path = output_dir / "baseline_bootstrap_paired_improvement.csv"
    paired.to_csv(paired_path, index=False)
    manifest = {
        "prepared_path": str(prepared_path),
        "comparison_path": str(comparison_path),
        "split_path": str(split_path) if split_path is not None else "",
        "output_dir": str(output_dir),
        "n_methods": int(len(predictions)),
        "methods": list(predictions),
        "n_supervised_spots": int(supervised_mask.sum().item()),
        "cohorts": sorted(metrics["cohort"].unique().tolist()),
        "n_bootstraps": int(n_bootstraps),
        "seed": int(seed),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "paired_improvement_path": str(paired_path),
        "notes": (
            "These are split-wise and paired bootstrap statistics on the current matched Xenium-to-Visium benchmark. "
            "They do not replace independent multi-sample matched-GT validation."
        ),
    }
    manifest_path = output_dir / "baseline_statistics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "paired_improvement_path": str(paired_path),
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute split/bootstrap baseline statistics.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--n-bootstraps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    outputs = collect_baseline_statistics(
        prepared_path=args.prepared,
        comparison_path=args.comparison,
        output_dir=args.output_dir,
        split_path=args.splits,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
    )
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
