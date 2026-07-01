"""Supplementary split-sensitivity analysis for Xenium-supervised spots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import summarize_proportion_metrics


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _read_prediction_table(path: str | Path, spot_ids: list[str], cell_types: list[str]) -> torch.Tensor:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    missing_spots = [spot_id for spot_id in spot_ids if spot_id not in frame.index]
    if missing_spots:
        raise ValueError(f"{path} is missing {len(missing_spots)} prediction spot ids")
    for cell_type in cell_types:
        if cell_type not in frame.columns:
            frame[cell_type] = 0.0
    frame = frame.loc[spot_ids, cell_types].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return torch.as_tensor(frame.to_numpy(), dtype=torch.float32)


def _split_counts(split_frame: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    merged = split_frame.merge(qc[["spot_id", "has_xenium_ground_truth"]], on="spot_id", how="left")
    merged["has_xenium_ground_truth"] = _truthy(merged["has_xenium_ground_truth"])
    return (
        merged.groupby("split", dropna=False)
        .agg(num_spots=("spot_id", "count"), num_supervised_spots=("has_xenium_ground_truth", "sum"))
        .reset_index()
        .sort_values("split")
    )


def _make_gt_stratified_split(
    original_splits: pd.DataFrame,
    qc: pd.DataFrame,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> pd.DataFrame:
    split = original_splits.copy()
    qc = qc.copy()
    qc["has_xenium_ground_truth"] = _truthy(qc["has_xenium_ground_truth"])
    gt = qc.loc[qc["has_xenium_ground_truth"], ["spot_id", "dominant_cell_type", "xenium_cell_count"]].copy()
    if gt.empty:
        return split
    rng = pd.Series(range(len(gt))).sample(frac=1.0, random_state=seed).index.to_numpy()
    gt = gt.iloc[rng].reset_index(drop=True)
    n_gt = len(gt)
    n_val = max(1, int(round(n_gt * val_fraction)))
    n_test = max(1, int(round(n_gt * test_fraction)))
    if n_val + n_test >= n_gt:
        n_val = max(1, n_gt // 5)
        n_test = max(1, n_gt // 5)
    labels = pd.Series("train", index=gt["spot_id"].astype(str))
    labels.iloc[:n_val] = "val"
    labels.iloc[n_val : n_val + n_test] = "test"
    split["spot_id"] = split["spot_id"].astype(str)
    split.loc[split["spot_id"].isin(labels.index), "split"] = split["spot_id"].map(labels).fillna(split["split"])
    return split


def _evaluate_split(
    split_frame: pd.DataFrame,
    split_name: str,
    predictions: torch.Tensor,
    target: torch.Tensor,
    spot_ids: list[str],
) -> dict[str, float | str | int]:
    split_by_spot = split_frame.set_index("spot_id")["split"].astype(str).reindex(spot_ids)
    mask = torch.as_tensor((split_by_spot == split_name).to_numpy(), dtype=torch.bool)
    metrics = summarize_proportion_metrics(predictions[mask], target[mask])
    return {"split": split_name, "num_spots": int(mask.sum().item()), **metrics}


def run_split_sensitivity(
    prepared_path: str | Path,
    predictions_path: str | Path,
    original_splits_path: str | Path,
    qc_path: str | Path,
    output_dir: str | Path,
    *,
    seeds: list[int] | None = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, str]:
    """Create GT-stratified split sensitivity tables for existing predictions."""

    seeds = seeds or [11, 17, 23, 31, 43]
    output_dir = Path(output_dir)
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    batch, metadata, _ = load_prepared_dataset(prepared_path, device="cpu")
    if batch.proportion_gt is None:
        raise ValueError("split sensitivity requires proportion_gt")
    spot_ids = [str(spot_id) for spot_id in metadata.spot_ids]
    target = batch.proportion_gt.float()
    predictions = _read_prediction_table(predictions_path, spot_ids, metadata.cell_types)
    original_splits = pd.read_csv(original_splits_path)
    original_splits["spot_id"] = original_splits["spot_id"].astype(str)
    qc = pd.read_csv(qc_path)
    qc["spot_id"] = qc["spot_id"].astype(str)

    original_summary = _split_counts(original_splits, qc)
    original_summary_path = output_dir / "original_split_gt_summary.csv"
    original_summary.to_csv(original_summary_path, index=False)

    metric_rows: list[dict[str, float | str | int]] = []
    generated_split_rows: list[dict[str, str | int]] = []
    for seed in seeds:
        split = _make_gt_stratified_split(original_splits, qc, seed, val_fraction, test_fraction)
        split_path = split_dir / f"gt_stratified_seed{seed}.csv"
        split.to_csv(split_path, index=False)
        counts = _split_counts(split, qc)
        for _, row in counts.iterrows():
            generated_split_rows.append(
                {
                    "seed": int(seed),
                    "split": str(row["split"]),
                    "num_spots": int(row["num_spots"]),
                    "num_supervised_spots": int(row["num_supervised_spots"]),
                    "split_path": str(split_path),
                }
            )
        for split_name in ["train", "val", "test"]:
            row = _evaluate_split(split, split_name, predictions, target, spot_ids)
            row["seed"] = int(seed)
            row["split_path"] = str(split_path)
            metric_rows.append(row)

    generated_summary_path = output_dir / "gt_stratified_split_summary.csv"
    pd.DataFrame(generated_split_rows).to_csv(generated_summary_path, index=False)
    metrics_path = output_dir / "split_sensitivity_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    metric_frame = pd.DataFrame(metric_rows)
    aggregate = (
        metric_frame.groupby("split")
        .agg(
            num_seeds=("seed", "count"),
            mean_supervised_spots=("num_supervised_spots", "mean"),
            mean_jsd=("jsd", "mean"),
            sd_jsd=("jsd", "std"),
            mean_spotwise_cosine=("spotwise_cosine", "mean"),
            mean_celltype_pearson=("mean_celltype_pearson", "mean"),
        )
        .reset_index()
    )
    aggregate_path = output_dir / "split_sensitivity_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)
    val_gt_original = int(original_summary.loc[original_summary["split"].astype(str).eq("val"), "num_supervised_spots"].sum())
    manifest = {
        "prepared_path": str(prepared_path),
        "predictions_path": str(predictions_path),
        "original_splits_path": str(original_splits_path),
        "qc_path": str(qc_path),
        "output_dir": str(output_dir),
        "seeds": seeds,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "original_val_supervised_spots": val_gt_original,
        "interpretation": (
            "The original spatial holdout has no Xenium-supervised validation spots. "
            "Supplementary GT-stratified splits therefore test metric stability without changing the primary trained model."
        ),
        "original_summary_path": str(original_summary_path),
        "generated_summary_path": str(generated_summary_path),
        "metrics_path": str(metrics_path),
        "aggregate_path": str(aggregate_path),
    }
    manifest_path = output_dir / "split_sensitivity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    markdown_path = output_dir / "split_sensitivity.md"
    lines = [
        "# Split Sensitivity",
        "",
        f"Original validation supervised spots: `{val_gt_original}`.",
        "",
        "Supplementary GT-stratified splits were generated only for sensitivity analysis; they do not replace the primary benchmark split.",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Aggregate: `{aggregate_path}`",
        f"- Generated split summary: `{generated_summary_path}`",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "original_summary_path": str(original_summary_path),
        "generated_summary_path": str(generated_summary_path),
        "metrics_path": str(metrics_path),
        "aggregate_path": str(aggregate_path),
        "markdown_path": str(markdown_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GT-stratified split sensitivity for existing predictions.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--qc", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 17, 23, 31, 43])
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    outputs = run_split_sensitivity(
        prepared_path=args.prepared,
        predictions_path=args.predictions,
        original_splits_path=args.splits,
        qc_path=args.qc,
        output_dir=args.output_dir,
        seeds=args.seeds,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
