"""Quantify whether H&E image information improves WaveST-Gate predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wavestgate.data.prepare_dataset import load_prepared_dataset


def _load_table(path: str | Path, expected_index: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    missing = [spot_id for spot_id in expected_index if spot_id not in frame.index]
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} prepared spot ids")
    return frame.loc[expected_index]


def _normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return values / np.clip(values.sum(axis=1, keepdims=True), eps, None)


def _per_spot_jsd(predicted: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    predicted = _normalize_rows(np.clip(predicted, eps, None), eps)
    target = _normalize_rows(np.clip(target, eps, None), eps)
    midpoint = 0.5 * (predicted + target)
    kl_pred = np.sum(predicted * np.log(predicted / np.clip(midpoint, eps, None)), axis=1)
    kl_target = np.sum(target * np.log(target / np.clip(midpoint, eps, None)), axis=1)
    return 0.5 * (kl_pred + kl_target)


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    if left.nunique(dropna=True) <= 1 or right.nunique(dropna=True) <= 1:
        return 0.0
    value = left.corr(right, method=method)
    return float(0.0 if pd.isna(value) else value)


def _run_comparison_row(label: str, run_dir: Path) -> dict[str, float | str]:
    metrics_path = run_dir / "metrics.csv"
    gates_path = run_dir / "gate_weights.csv"
    raw_gates_path = run_dir / "raw_gate_weights.csv"
    row: dict[str, float | str] = {"setting": label, "run_dir": str(run_dir)}
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path).tail(1).iloc[0]
        for key in ["jsd", "mean_celltype_pearson", "spotwise_cosine", "uncertainty_error_pearson"]:
            row[key] = float(metrics[key]) if key in metrics else float("nan")
    if gates_path.exists():
        gates = pd.read_csv(gates_path, index_col=0)
        row["mean_image_gate"] = float(gates["image"].mean())
        row["median_image_gate"] = float(gates["image"].median())
    if raw_gates_path.exists():
        raw_gates = pd.read_csv(raw_gates_path, index_col=0)
        row["mean_raw_image_gate"] = float(raw_gates["image"].mean())
    return row


def run_image_contribution_analysis(
    prepared_path: str | Path,
    image_run_dir: str | Path,
    no_image_run_dir: str | Path,
    output_dir: str | Path,
    *,
    baseline_run_dir: str | Path | None = None,
) -> dict[str, float | str | int]:
    """Write image-contribution summary files and return the main summary."""

    batch, metadata, _ = load_prepared_dataset(prepared_path, device="cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spot_ids = [str(spot_id) for spot_id in metadata.spot_ids]
    image_run_dir = Path(image_run_dir)
    no_image_run_dir = Path(no_image_run_dir)
    image_pred = _load_table(image_run_dir / "predicted_proportions.csv", spot_ids)
    no_image_pred = _load_table(no_image_run_dir / "predicted_proportions.csv", spot_ids)
    image_gate = _load_table(image_run_dir / "gate_weights.csv", spot_ids)
    raw_gate_path = image_run_dir / "raw_gate_weights.csv"
    raw_image_gate = _load_table(raw_gate_path, spot_ids)["image"] if raw_gate_path.exists() else image_gate["image"]

    gt = batch.proportion_gt
    if gt is None:
        raise ValueError("prepared dataset does not contain Xenium-derived proportion_gt")
    gt_frame = pd.DataFrame(gt.numpy(), index=spot_ids, columns=metadata.cell_types)
    supervised = gt_frame.sum(axis=1) > 1e-8
    if not supervised.any():
        raise ValueError("prepared dataset has no supervised spots")

    image_jsd = pd.Series(
        _per_spot_jsd(image_pred.loc[supervised].to_numpy(), gt_frame.loc[supervised].to_numpy()),
        index=gt_frame.index[supervised],
        name="imagegate_jsd",
    )
    no_image_jsd = pd.Series(
        _per_spot_jsd(no_image_pred.loc[supervised].to_numpy(), gt_frame.loc[supervised].to_numpy()),
        index=gt_frame.index[supervised],
        name="noimage_jsd",
    )
    patch_texture = pd.Series(batch.image_patches.float().std(dim=(1, 2, 3)).numpy(), index=spot_ids, name="patch_texture")
    patch_intensity = pd.Series(batch.image_patches.float().mean(dim=(1, 2, 3)).numpy(), index=spot_ids, name="patch_intensity")
    supervised_texture = patch_texture.loc[supervised]
    high_texture = supervised_texture >= supervised_texture.quantile(0.75)
    low_texture = supervised_texture <= supervised_texture.quantile(0.25)
    delta = no_image_jsd - image_jsd

    per_spot = pd.DataFrame(
        {
            "patch_texture": patch_texture.loc[supervised],
            "patch_intensity": patch_intensity.loc[supervised],
            "image_gate": image_gate.loc[supervised, "image"],
            "raw_image_gate": raw_image_gate.loc[supervised],
            "imagegate_jsd": image_jsd,
            "noimage_jsd": no_image_jsd,
            "paired_jsd_improvement": delta,
        }
    )
    baseline_delta_mean = None
    if baseline_run_dir is not None:
        baseline_pred = _load_table(Path(baseline_run_dir) / "predicted_proportions.csv", spot_ids)
        baseline_jsd = pd.Series(
            _per_spot_jsd(baseline_pred.loc[supervised].to_numpy(), gt_frame.loc[supervised].to_numpy()),
            index=gt_frame.index[supervised],
            name="baseline_jsd",
        )
        per_spot["baseline_jsd"] = baseline_jsd
        per_spot["paired_jsd_improvement_vs_baseline"] = baseline_jsd - image_jsd
        baseline_delta_mean = float(per_spot["paired_jsd_improvement_vs_baseline"].mean())

    per_spot_path = output_dir / "image_contribution_per_spot.csv"
    per_spot.to_csv(per_spot_path, index_label="spot_id")

    texture_groups = pd.DataFrame(
        [
            {
                "group": "low_texture_q1",
                "num_spots": int(low_texture.sum()),
                "mean_image_gate": float(per_spot.loc[low_texture, "image_gate"].mean()),
                "mean_raw_image_gate": float(per_spot.loc[low_texture, "raw_image_gate"].mean()),
                "mean_paired_jsd_improvement": float(delta.loc[low_texture.index[low_texture]].mean()),
                "mean_imagegate_jsd": float(image_jsd.loc[low_texture.index[low_texture]].mean()),
                "mean_noimage_jsd": float(no_image_jsd.loc[low_texture.index[low_texture]].mean()),
            },
            {
                "group": "high_texture_q4",
                "num_spots": int(high_texture.sum()),
                "mean_image_gate": float(per_spot.loc[high_texture, "image_gate"].mean()),
                "mean_raw_image_gate": float(per_spot.loc[high_texture, "raw_image_gate"].mean()),
                "mean_paired_jsd_improvement": float(delta.loc[high_texture.index[high_texture]].mean()),
                "mean_imagegate_jsd": float(image_jsd.loc[high_texture.index[high_texture]].mean()),
                "mean_noimage_jsd": float(no_image_jsd.loc[high_texture.index[high_texture]].mean()),
            },
        ]
    )
    texture_groups_path = output_dir / "image_contribution_texture_groups.csv"
    texture_groups.to_csv(texture_groups_path, index=False)
    run_rows = [
        _run_comparison_row("imagegate_enhanced", image_run_dir),
        _run_comparison_row("matched_no_image_control", no_image_run_dir),
    ]
    if baseline_run_dir is not None:
        run_rows.insert(0, _run_comparison_row("baseline_main", Path(baseline_run_dir)))
    run_comparison_path = output_dir / "imagegate_run_comparison.csv"
    pd.DataFrame(run_rows).to_csv(run_comparison_path, index=False)

    summary: dict[str, float | str | int] = {
        "num_supervised_spots": int(supervised.sum()),
        "mean_image_gate": float(image_gate["image"].mean()),
        "median_image_gate": float(image_gate["image"].median()),
        "mean_raw_image_gate": float(raw_image_gate.mean()),
        "mean_imagegate_jsd": float(image_jsd.mean()),
        "mean_noimage_jsd": float(no_image_jsd.mean()),
        "mean_paired_jsd_improvement_noimage_minus_imagegate": float(delta.mean()),
        "high_texture_mean_paired_jsd_improvement": float(delta.loc[high_texture.index[high_texture]].mean()),
        "low_texture_mean_paired_jsd_improvement": float(delta.loc[low_texture.index[low_texture]].mean()),
        "image_gate_patch_texture_pearson": _safe_corr(image_gate["image"], patch_texture, "pearson"),
        "image_gate_patch_texture_spearman": _safe_corr(image_gate["image"], patch_texture, "spearman"),
        "supervised_image_gate_error_spearman": _safe_corr(per_spot["image_gate"], per_spot["imagegate_jsd"], "spearman"),
        "per_spot_path": str(per_spot_path),
        "texture_groups_path": str(texture_groups_path),
        "run_comparison_path": str(run_comparison_path),
    }
    if baseline_delta_mean is not None:
        summary["mean_paired_jsd_improvement_baseline_minus_imagegate"] = baseline_delta_mean

    summary_path = output_dir / "image_contribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze H&E image contribution against a no-image control.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--image-run-dir", required=True)
    parser.add_argument("--no-image-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-run-dir", default=None)
    args = parser.parse_args()
    summary = run_image_contribution_analysis(
        prepared_path=args.prepared,
        image_run_dir=args.image_run_dir,
        no_image_run_dir=args.no_image_run_dir,
        output_dir=args.output_dir,
        baseline_run_dir=args.baseline_run_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
