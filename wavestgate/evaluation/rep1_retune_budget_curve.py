"""Rep1 no-retuning vs minimal-retuning budget curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import summarize_proportion_metrics
from wavestgate.training.train import load_config
from wavestgate.training.train_real import train_real_from_config


def _read_predictions(path: str | Path, spot_ids: list[str], cell_types: list[str]) -> torch.Tensor:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    for cell_type in cell_types:
        if cell_type not in frame.columns:
            frame[cell_type] = 0.0
    return torch.as_tensor(frame.reindex(spot_ids).fillna(0.0)[cell_types].to_numpy(), dtype=torch.float32)


def _eval_prediction_path(
    predictions_path: str | Path,
    prepared_path: str | Path,
    split_path: str | Path,
    eval_split: str,
) -> dict[str, float]:
    batch, metadata, _ = load_prepared_dataset(prepared_path, device="cpu")
    if batch.proportion_gt is None:
        raise ValueError("Rep1 budget curve requires proportion_gt")
    spot_ids = [str(spot_id) for spot_id in metadata.spot_ids]
    predictions = _read_predictions(predictions_path, spot_ids, metadata.cell_types)
    split = pd.read_csv(split_path)
    split["spot_id"] = split["spot_id"].astype(str)
    split_by_spot = split.set_index("spot_id")["split"].astype(str).reindex(spot_ids)
    mask = torch.as_tensor((split_by_spot == eval_split).to_numpy(), dtype=torch.bool)
    return summarize_proportion_metrics(predictions[mask], batch.proportion_gt.float()[mask])


def _best_baseline(comparison_path: str | Path) -> dict[str, Any]:
    frame = pd.read_csv(comparison_path)
    if frame.empty:
        return {"method": "", "jsd": 0.0}
    non_model = frame.loc[~frame["method"].astype(str).eq("WaveST-Gate")].copy()
    if non_model.empty:
        return {"method": "", "jsd": 0.0}
    row = non_model.sort_values("jsd", ascending=True).iloc[0].to_dict()
    return {"method": str(row.get("method", "")), "jsd": float(row.get("jsd", 0.0))}


def _budget_config(base_config: dict[str, Any], budget: int, run_dir: Path) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config.setdefault("training", {})
    config["training"]["steps"] = int(budget)
    config["training"]["checkpoint_path"] = str(run_dir / "checkpoint.pt")
    config["training"]["metrics_path"] = str(run_dir / "metrics.csv")
    config["training"]["history_path"] = str(run_dir / "training_history.csv")
    config["predictions"] = {
        "proportions_path": str(run_dir / "predicted_proportions.csv"),
        "gates_path": str(run_dir / "gate_weights.csv"),
        "uncertainty_path": str(run_dir / "spot_uncertainty.csv"),
        "modality_reliability_path": str(run_dir / "modality_reliability.csv"),
        "reconstructed_expression_path": str(run_dir / "reconstructed_expression.csv"),
        "agent_attention_path": str(run_dir / "agent_attention.csv"),
    }
    return config


def run_rep1_retune_budget_curve(
    base_config_path: str | Path,
    no_retune_predictions_path: str | Path,
    baseline_comparison_path: str | Path,
    output_dir: str | Path,
    *,
    budgets: list[int] | None = None,
    eval_split: str = "test",
    reuse_existing: bool = True,
) -> dict[str, str]:
    """Train/evaluate minimal-retuning budgets and write a curve."""

    budgets = budgets or [0, 25, 50, 100, 250, 500]
    output_dir = Path(output_dir)
    run_root = output_dir / "runs"
    config_root = output_dir / "configs"
    run_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    base_config = load_config(base_config_path)
    prepared_path = base_config["data"]["prepared_path"]
    split_path = base_config["data"]["split_path"]
    best_baseline = _best_baseline(baseline_comparison_path)
    rows: list[dict[str, float | int | str | bool]] = []
    for budget in budgets:
        if int(budget) == 0:
            metrics = _eval_prediction_path(no_retune_predictions_path, prepared_path, split_path, eval_split)
            predictions_path = str(no_retune_predictions_path)
            run_dir = ""
        else:
            run_dir_path = run_root / f"steps_{int(budget)}"
            config = _budget_config(base_config, int(budget), run_dir_path)
            config_path = config_root / f"steps_{int(budget)}.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            metrics_path = run_dir_path / "metrics.csv"
            if reuse_existing and metrics_path.exists():
                metrics = pd.read_csv(metrics_path).tail(1).iloc[0].to_dict()
            else:
                metrics = train_real_from_config(config_path)
            predictions_path = config["predictions"]["proportions_path"]
            run_dir = str(run_dir_path)
        jsd = float(metrics.get("jsd", 0.0))
        rows.append(
            {
                "budget_steps": int(budget),
                "run_dir": run_dir,
                "predictions_path": predictions_path,
                "best_baseline_method": best_baseline["method"],
                "best_baseline_jsd": float(best_baseline["jsd"]),
                "beats_best_baseline": bool(jsd < float(best_baseline["jsd"])) if best_baseline["method"] else False,
                "jsd_margin_vs_best_baseline": float(best_baseline["jsd"]) - jsd if best_baseline["method"] else 0.0,
                **{key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
            }
        )
    curve = pd.DataFrame(rows).sort_values("budget_steps")
    curve_path = output_dir / "rep1_minimal_retune_budget_curve.csv"
    curve.to_csv(curve_path, index=False)
    crossing = curve.loc[curve["beats_best_baseline"]]
    min_budget = int(crossing["budget_steps"].min()) if not crossing.empty else None
    manifest = {
        "base_config_path": str(base_config_path),
        "prepared_path": str(prepared_path),
        "split_path": str(split_path),
        "eval_split": eval_split,
        "budgets": budgets,
        "no_retune_predictions_path": str(no_retune_predictions_path),
        "baseline_comparison_path": str(baseline_comparison_path),
        "best_baseline": best_baseline,
        "minimum_budget_beating_best_baseline": min_budget,
        "curve_path": str(curve_path),
        "interpretation": (
            "Rep1 no-retuning is reported honestly as a domain-shift case. "
            "The budget curve quantifies how little Rep1 supervision is needed for the Rep2-initialized model to recover."
        ),
    }
    manifest_path = output_dir / "rep1_minimal_retune_budget_curve_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    markdown_path = output_dir / "rep1_minimal_retune_budget_curve.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Rep1 Minimal-Retuning Budget Curve",
                "",
                f"Best held-out baseline: `{best_baseline['method']}` with JSD `{best_baseline['jsd']}`.",
                f"Minimum budget beating best baseline: `{min_budget}` steps.",
                "",
                f"- Curve: `{curve_path}`",
                f"- Manifest: `{manifest_path}`",
            ]
        ),
        encoding="utf-8",
    )
    return {"manifest_path": str(manifest_path), "curve_path": str(curve_path), "markdown_path": str(markdown_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Rep1 minimal-retuning budget curve.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--no-retune-predictions", required=True)
    parser.add_argument("--baseline-comparison", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", nargs="*", type=int, default=[0, 25, 50, 100, 250, 500])
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--no-reuse-existing", action="store_true")
    args = parser.parse_args()
    outputs = run_rep1_retune_budget_curve(
        base_config_path=args.base_config,
        no_retune_predictions_path=args.no_retune_predictions,
        baseline_comparison_path=args.baseline_comparison,
        output_dir=args.output_dir,
        budgets=args.budgets,
        eval_split=args.eval_split,
        reuse_existing=not args.no_reuse_existing,
    )
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
