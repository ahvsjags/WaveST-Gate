"""Ablation suite generation and execution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from wavestgate.training.train import load_config
from wavestgate.training.train_real import train_real_from_config


ABLATION_OVERRIDES: dict[str, dict[str, Any]] = {
    "full": {},
    "no_wavelet_cnn_replacement": {"image_encoder_type": "cnn"},
    "no_image_branch": {"image_encoder_type": "none", "use_local_refinement": False},
    "no_celltype_agents": {"use_celltype_agents": False},
    "no_gate_mean_fusion": {"use_cross_modal_gate": False, "use_uncertainty_calibration": False},
    "raw_gate_without_uncertainty": {"use_uncertainty_calibration": False, "loss_uncertainty_weight": 0.0},
    "no_boundary_loss": {"loss_boundary_weight": 0.0},
    "normal_smoothness_only": {"loss_boundary_weight": 0.0, "loss_spatial_weight": 0.05},
    "no_local_refinement": {"use_local_refinement": False},
    "expression_only": {
        "image_encoder_type": "none",
        "use_local_refinement": False,
        "use_celltype_agents": False,
        "use_cross_modal_gate": False,
        "use_uncertainty_calibration": False,
        "loss_contrast_weight": 0.0,
        "loss_boundary_weight": 0.0,
    },
    "image_only": {
        "use_expression_branch": False,
        "use_celltype_agents": False,
        "use_cross_modal_gate": False,
        "use_uncertainty_calibration": False,
        "loss_expr_weight": 0.0,
        "loss_contrast_weight": 0.0,
    },
    "reference_only": {
        "image_encoder_type": "none",
        "use_expression_branch": False,
        "use_local_refinement": False,
        "use_cross_modal_gate": False,
        "use_uncertainty_calibration": False,
        "loss_expr_weight": 0.0,
        "loss_contrast_weight": 0.0,
        "loss_boundary_weight": 0.0,
    },
}


def list_default_ablations() -> list[str]:
    return list(ABLATION_OVERRIDES)


def _with_nested_update(config: dict[str, Any], section: str, updates: dict[str, Any]) -> None:
    config.setdefault(section, {})
    config[section].update(updates)


def build_ablation_config(base_config: dict[str, Any], ablation_name: str, output_root: Path) -> dict[str, Any]:
    if ablation_name not in ABLATION_OVERRIDES:
        raise ValueError(f"Unknown ablation: {ablation_name}")
    config = json.loads(json.dumps(base_config))
    _with_nested_update(config, "model", ABLATION_OVERRIDES[ablation_name])
    run_dir = output_root / ablation_name
    config.setdefault("training", {})
    config["training"]["checkpoint_path"] = str(run_dir / "checkpoint.pt")
    config["training"]["metrics_path"] = str(run_dir / "metrics.csv")
    config["training"]["history_path"] = str(run_dir / "training_history.csv")
    config["predictions"] = {
        "proportions_path": str(run_dir / "predicted_proportions.csv"),
        "gates_path": str(run_dir / "gate_weights.csv"),
        "raw_gates_path": str(run_dir / "raw_gate_weights.csv"),
        "uncertainty_path": str(run_dir / "spot_uncertainty.csv"),
        "modality_reliability_path": str(run_dir / "modality_reliability.csv"),
        "reconstructed_expression_path": str(run_dir / "reconstructed_expression.csv"),
        "agent_attention_path": str(run_dir / "agent_attention.csv"),
    }
    return config


def write_ablation_configs(
    base_config_path: str | Path,
    output_root: str | Path,
    ablations: list[str] | None = None,
) -> list[Path]:
    base = load_config(base_config_path)
    output_root = Path(output_root)
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ablations or list_default_ablations():
        config = build_ablation_config(base, name, output_root)
        path = config_dir / f"{name}.yaml"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def run_ablation_suite(
    base_config_path: str | Path,
    output_root: str | Path,
    ablations: list[str] | None = None,
) -> list[dict[str, Any]]:
    config_paths = write_ablation_configs(base_config_path, output_root, ablations)
    rows = []
    for config_path in config_paths:
        metrics = train_real_from_config(config_path)
        row = {"ablation": config_path.stem, "config_path": str(config_path)}
        row.update(metrics)
        rows.append(row)
    summary_path = Path(output_root) / "ablation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize_existing_ablation_metrics(output_root: str | Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(Path(output_root).glob("*/metrics.csv")):
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        row = df.iloc[-1].to_dict()
        row["ablation"] = metrics_path.parent.name
        row["metrics_path"] = str(metrics_path)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or run WaveST-Gate ablation configs.")
    parser.add_argument("--base-config", required=True, help="Base train_real config.")
    parser.add_argument("--output-root", required=True, help="Output root for ablation runs.")
    parser.add_argument("--ablations", nargs="*", default=None, help="Optional subset of ablation names.")
    parser.add_argument("--run", action="store_true", help="Run the generated ablations.")
    args = parser.parse_args()
    if args.run:
        rows = run_ablation_suite(args.base_config, args.output_root, args.ablations)
        for row in rows:
            print(json.dumps(row, indent=2), flush=True)
    else:
        paths = write_ablation_configs(args.base_config, args.output_root, args.ablations)
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
