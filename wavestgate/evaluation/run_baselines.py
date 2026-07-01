"""Utilities for baseline audit and result aggregation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import per_spot_jensen_shannon_divergence, summarize_proportion_metrics


PLANNED_BASELINES = [
    "cell2location",
    "RCTD",
    "CARD",
    "Tangram",
    "SpatialDWLS",
    "SpatialDWLS/Seurat",
    "BayesPrism",
    "DestVI",
    "SPOTlight",
    "reference_cosine",
    "reference_nnls",
    "uniform",
]

PYTHON_BASELINE_PACKAGES = {
    "Tangram": ["tangram", "anndata", "scanpy"],
    "cell2location": ["cell2location", "anndata", "scanpy"],
}

EXTERNAL_PYTHON_BASELINES = {
    "cell2location": {
        "env_var": "CELL2LOCATION_PYTHON",
        "default_python": "/root/miniconda3/envs/cell2loc_env/bin/python",
        "packages": ["cell2location", "scvi", "torch", "anndata", "scanpy"],
    }
}

R_BASELINE_PACKAGES = {
    "RCTD": ["spacexr"],
    "CARD": ["CARD", "MuSiC"],
    "SpatialDWLS": ["Matrix", "quadprog"],
    "SpatialDWLS/Seurat": ["Seurat", "Giotto", "GiottoClass", "GiottoUtils", "GiottoVisuals", "Rfast", "quadprog"],
    "BayesPrism": ["BayesPrism"],
    "SPOTlight": ["SPOTlight"],
}


def list_planned_baselines() -> list[str]:
    return list(PLANNED_BASELINES)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _module_available_with_python(python_executable: str | Path, name: str) -> bool:
    executable = Path(python_executable)
    if not executable.exists():
        return False
    cmd = [str(executable), "-c", f"import {name}; raise SystemExit(0)"]
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "True"
    return subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env).returncode == 0


def _r_package_available(package: str) -> bool | None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None
    cmd = [
        rscript,
        "-e",
        f"quit(status = ifelse(requireNamespace('{package}', quietly = TRUE), 0, 1))",
    ]
    return subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def audit_baseline_environment(output_path: str | Path | None = None) -> dict[str, Any]:
    """Record which formal baseline environments are currently runnable."""

    python = {
        method: {package: _module_available(package) for package in packages}
        for method, packages in PYTHON_BASELINE_PACKAGES.items()
    }
    external_python = {}
    for method, spec in EXTERNAL_PYTHON_BASELINES.items():
        executable = os.environ.get(str(spec["env_var"]), str(spec["default_python"]))
        external_python[method] = {
            "python": executable,
            "packages": {
                package: _module_available_with_python(executable, package) for package in spec["packages"]
            },
        }
    rscript = shutil.which("Rscript")
    r = {
        method: {package: _r_package_available(package) for package in packages}
        for method, packages in R_BASELINE_PACKAGES.items()
    }
    status = {}
    for method, packages in python.items():
        external = external_python.get(method, {})
        external_packages = external.get("packages", {})
        status[method] = (
            "ready"
            if all(packages.values()) or (external_packages and all(external_packages.values()))
            else "missing_python_packages"
        )
    for method, packages in r.items():
        if rscript is None:
            status[method] = "missing_R"
        elif all(value is True for value in packages.values()):
            status[method] = "ready"
        else:
            status[method] = "missing_R_packages"
    result = {
        "planned_baselines": PLANNED_BASELINES,
        "python_packages": python,
        "external_python_packages": external_python,
        "r_executable": shutil.which("R"),
        "rscript_executable": rscript,
        "r_packages": r,
        "status": status,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _read_csv_row(path: str | Path) -> dict[str, Any]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[-1]


def _read_tangram_label(metrics_path: str | Path) -> str:
    manifest_path = Path(metrics_path).with_name("tangram_manifest.json")
    if not manifest_path.exists():
        return "Tangram"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior = manifest.get("density_prior")
    epochs = manifest.get("num_epochs")
    if prior and epochs:
        return f"Tangram ({prior}, {epochs} epochs)"
    return "Tangram"


def _coerce_metric_row(row: dict[str, Any], method: str, source: str) -> dict[str, Any]:
    metric_keys = [
        "spotwise_cosine",
        "mean_celltype_pearson",
        "rmse",
        "jsd",
        "num_supervised_spots",
        "runtime_seconds",
        "peak_cuda_memory_mb",
        "device",
        "predictions_path",
    ]
    out: dict[str, Any] = {"method": method, "source": source}
    for key in metric_keys:
        value = row.get(key, "")
        if key in {"device", "predictions_path"}:
            out[key] = value
        elif value == "" or value is None:
            out[key] = ""
        else:
            out[key] = float(value)
    return out


def _load_prediction_matrix(path: str | Path, spot_ids: list[str], cell_types: list[str]) -> torch.Tensor:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame = frame.reindex(index=spot_ids)
    for cell_type in cell_types:
        if cell_type not in frame.columns:
            frame[cell_type] = 0.0
    values = frame[cell_types].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32)
    sums = values.sum(axis=1, keepdims=True)
    empty = sums[:, 0] <= 1e-12
    if empty.any():
        values[empty, :] = 1.0 / len(cell_types)
        sums = values.sum(axis=1, keepdims=True)
    values = values / np.maximum(sums, 1e-12)
    return torch.as_tensor(values, dtype=torch.float32)


def _as_split_list(value: list[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _select_spots_by_split(
    spot_ids: list[str],
    target: torch.Tensor | None,
    split_path: str | Path | None,
    eval_splits: list[str] | str | None,
) -> tuple[list[str], torch.Tensor | None, torch.Tensor | None, list[str] | None]:
    requested = _as_split_list(eval_splits)
    if split_path is None or requested is None:
        return spot_ids, target, None, requested
    frame = pd.read_csv(split_path)
    required = {"spot_id", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Split file must contain columns {sorted(required)}")
    frame["spot_id"] = frame["spot_id"].astype(str)
    split_by_spot = frame.set_index("spot_id")["split"].astype(str)
    keep = torch.as_tensor([str(split_by_spot.get(str(spot_id), "")) in set(requested) for spot_id in spot_ids], dtype=torch.bool)
    if not bool(keep.any()):
        raise ValueError(f"No spots matched requested eval splits: {requested}")
    selected_spot_ids = [spot_id for spot_id, selected in zip(spot_ids, keep.tolist()) if selected]
    selected_target = target[keep] if target is not None else None
    return selected_spot_ids, selected_target, keep, requested


def _paired_jsd_test(
    model_pred: torch.Tensor,
    baseline_pred: torch.Tensor,
    target: torch.Tensor,
    permutations: int = 10000,
    seed: int = 13,
) -> dict[str, float]:
    mask = target.sum(dim=1) > 1e-8
    model_errors = per_spot_jensen_shannon_divergence(model_pred[mask], target[mask]).detach().cpu().numpy()
    baseline_errors = per_spot_jensen_shannon_divergence(baseline_pred[mask], target[mask]).detach().cpu().numpy()
    diff = baseline_errors - model_errors
    observed = float(diff.mean())
    if len(diff) == 0:
        return {"mean_jsd_improvement_vs_wavestgate": 0.0, "paired_permutation_p": 1.0}
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(permutations), len(diff)))
    null_means = (signs * diff[None, :]).mean(axis=1)
    p_value = float((np.count_nonzero(null_means >= observed) + 1) / (len(null_means) + 1))
    return {
        "mean_jsd_improvement_vs_wavestgate": observed,
        "paired_permutation_p": p_value,
    }


def collect_baseline_comparison(
    prepared_path: str | Path,
    model_metrics_path: str | Path,
    model_predictions_path: str | Path,
    output_dir: str | Path,
    simple_metrics_path: str | Path | None = None,
    tangram_metrics_paths: list[str | Path] | None = None,
    baseline_metrics_paths: list[str | Path] | None = None,
    split_path: str | Path | None = None,
    eval_splits: list[str] | str | None = None,
    permutations: int = 10000,
    seed: int = 13,
) -> pd.DataFrame:
    """Merge WaveST-Gate and baseline metrics into one comparison table."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch, metadata, _ = load_prepared_dataset(prepared_path, device="cpu")
    selected_spot_ids, selected_target, split_mask, requested_splits = _select_spots_by_split(
        metadata.spot_ids,
        batch.proportion_gt.detach().cpu() if batch.proportion_gt is not None else None,
        split_path,
        eval_splits,
    )
    model_metrics = _read_csv_row(model_metrics_path)
    rows = [
        _coerce_metric_row(
            {
                **model_metrics,
                "predictions_path": str(model_predictions_path),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            method="WaveST-Gate",
            source="main_model",
        )
    ]

    if simple_metrics_path is not None and Path(simple_metrics_path).exists():
        simple = pd.read_csv(simple_metrics_path)
        for _, row in simple.iterrows():
            rows.append(_coerce_metric_row(row.to_dict(), method=str(row["method"]), source="simple_baseline"))

    for path in tangram_metrics_paths or []:
        if Path(path).exists():
            rows.append(_coerce_metric_row(_read_csv_row(path), method=_read_tangram_label(path), source="Tangram"))

    for path in baseline_metrics_paths or []:
        if Path(path).exists():
            row = _read_csv_row(path)
            rows.append(_coerce_metric_row(row, method=str(row.get("method", Path(path).parent.name)), source="formal_baseline"))

    model_pred = _load_prediction_matrix(model_predictions_path, selected_spot_ids, metadata.cell_types)
    for row in rows:
        pred_path = row.get("predictions_path")
        if selected_target is not None and pred_path:
            row.update(summarize_proportion_metrics(_load_prediction_matrix(pred_path, selected_spot_ids, metadata.cell_types), selected_target))
        if row["method"] == "WaveST-Gate" or not pred_path:
            row["mean_jsd_improvement_vs_wavestgate"] = 0.0
            row["paired_permutation_p"] = ""
            continue
        baseline_pred = _load_prediction_matrix(pred_path, selected_spot_ids, metadata.cell_types)
        row.update(
            _paired_jsd_test(
                model_pred,
                baseline_pred,
                selected_target if selected_target is not None else batch.proportion_gt.detach().cpu(),
                permutations=permutations,
                seed=seed,
            )
        )

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values("jsd", ascending=True).reset_index(drop=True)
    comparison.insert(0, "rank_by_jsd", np.arange(1, len(comparison) + 1))
    path = output_dir / "baseline_comparison.csv"
    comparison.to_csv(path, index=False)

    manifest = {
        "prepared_path": str(prepared_path),
        "model_metrics_path": str(model_metrics_path),
        "model_predictions_path": str(model_predictions_path),
        "simple_metrics_path": str(simple_metrics_path) if simple_metrics_path is not None else None,
        "tangram_metrics_paths": [str(p) for p in tangram_metrics_paths or []],
        "baseline_metrics_paths": [str(p) for p in baseline_metrics_paths or []],
        "split_path": str(split_path) if split_path is not None else None,
        "eval_splits": requested_splits,
        "num_evaluation_spots": int(len(selected_spot_ids)),
        "split_mask_applied": split_mask is not None,
        "permutations": int(permutations),
        "seed": int(seed),
        "comparison_path": str(path),
        "notes": (
            "Paired p-values use supervised Xenium-covered spots and test per-spot JSD "
            "differences against WaveST-Gate. Multi-sample mean/std remains required for "
            "the final submission-scale baseline panel."
        ),
    }
    (output_dir / "baseline_comparison_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return comparison


def _main_audit(args: argparse.Namespace) -> None:
    result = audit_baseline_environment(args.output)
    print(json.dumps(result, indent=2), flush=True)


def _main_collect(args: argparse.Namespace) -> None:
    comparison = collect_baseline_comparison(
        prepared_path=args.prepared,
        model_metrics_path=args.model_metrics,
        model_predictions_path=args.model_predictions,
        output_dir=args.output_dir,
        simple_metrics_path=args.simple_metrics,
        tangram_metrics_paths=args.tangram_metrics,
        baseline_metrics_paths=args.baseline_metrics,
        split_path=args.splits,
        eval_splits=args.eval_splits or None,
        permutations=args.permutations,
        seed=args.seed,
    )
    print(comparison.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and aggregate WaveST-Gate baseline results.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit optional baseline environments.")
    audit.add_argument("--output", default=None, help="Optional JSON output path.")
    audit.set_defaults(func=_main_audit)

    collect = subparsers.add_parser("collect", help="Collect model and baseline metrics.")
    collect.add_argument("--prepared", required=True)
    collect.add_argument("--model-metrics", required=True)
    collect.add_argument("--model-predictions", required=True)
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--simple-metrics", default=None)
    collect.add_argument("--tangram-metrics", action="append", default=[])
    collect.add_argument("--baseline-metrics", action="append", default=[])
    collect.add_argument("--splits", default=None, help="Optional spot split CSV with spot_id and split columns.")
    collect.add_argument("--eval-splits", action="append", default=[], help="Evaluate and recompute metrics only on these split names.")
    collect.add_argument("--permutations", type=int, default=10000)
    collect.add_argument("--seed", type=int, default=13)
    collect.set_defaults(func=_main_collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
