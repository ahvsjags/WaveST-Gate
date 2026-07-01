"""Leakage and baseline-fairness audit for WaveST-Gate submission evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wavestgate.training.train import load_config


def _read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _cell_type_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != "spot_id"]


def _normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.clip(values.astype(np.float64), 0.0, None)
    sums = values.sum(axis=1, keepdims=True)
    empty = sums[:, 0] <= eps
    if empty.any():
        values[empty, :] = 1.0 / values.shape[1]
        sums = values.sum(axis=1, keepdims=True)
    return values / np.maximum(sums, eps)


def _jsd(predicted: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    p = _normalize(predicted, eps=eps)
    q = _normalize(target, eps=eps)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    values = 0.5 * ((p * np.log(p / m)).sum(axis=1) + (q * np.log(q / m)).sum(axis=1))
    return float(values.mean()) if len(values) else 0.0


def _prediction_values(path: str | Path, spot_ids: list[str], cell_types: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    frame = _read_csv(path)
    if "spot_id" not in frame.columns:
        first = frame.columns[0]
        frame = frame.rename(columns={first: "spot_id"})
    frame["spot_id"] = frame["spot_id"].astype(str)
    missing_spots = sorted(set(spot_ids) - set(frame["spot_id"]))
    missing_cell_types = [cell_type for cell_type in cell_types if cell_type not in frame.columns]
    for cell_type in missing_cell_types:
        frame[cell_type] = 0.0
    aligned = frame.set_index("spot_id").reindex(spot_ids)
    values = aligned[cell_types].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    diagnostics = {
        "path": str(path),
        "num_prediction_rows": int(len(frame)),
        "num_required_spots": int(len(spot_ids)),
        "num_missing_required_spots": int(len(missing_spots)),
        "missing_required_spots_preview": missing_spots[:5],
        "num_missing_cell_types": int(len(missing_cell_types)),
        "missing_cell_types": missing_cell_types,
        "min_row_sum": float(np.nanmin(values.sum(axis=1))) if values.size else 0.0,
        "max_row_sum": float(np.nanmax(values.sum(axis=1))) if values.size else 0.0,
    }
    return _normalize(values), diagnostics


def _load_ground_truth(ground_truth_path: str | Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    gt = _read_csv(ground_truth_path)
    if "spot_id" not in gt.columns:
        raise ValueError("ground truth table must contain a spot_id column")
    gt["spot_id"] = gt["spot_id"].astype(str)
    cell_types = _cell_type_columns(gt)
    values = gt[cell_types].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    supervised_mask = values.sum(axis=1) > 1e-8
    supervised_spots = gt.loc[supervised_mask, "spot_id"].astype(str).tolist()
    return gt, cell_types, supervised_spots


def _add_check(
    rows: list[dict[str, Any]],
    check: str,
    status: str,
    detail: str,
    metric: str = "",
    evidence: str = "",
) -> None:
    rows.append({"check": check, "status": status, "detail": detail, "metric": metric, "evidence": evidence})


def _split_audit(
    split_path: Path,
    qc_path: Path,
    supervised_spots: list[str],
    checklist: list[dict[str, Any]],
) -> dict[str, Any]:
    split = _read_csv(split_path)
    qc = _read_csv(qc_path)
    split["spot_id"] = split["spot_id"].astype(str)
    qc["spot_id"] = qc["spot_id"].astype(str)
    duplicate_spots = int(split["spot_id"].duplicated().sum())
    split_sets = {name: set(group["spot_id"]) for name, group in split.groupby("split")}
    overlaps = {
        f"{left}_vs_{right}": len(split_sets[left] & split_sets[right])
        for idx, left in enumerate(sorted(split_sets))
        for right in sorted(split_sets)[idx + 1 :]
    }
    merged = split.merge(qc[["spot_id", "has_xenium_ground_truth"]], on="spot_id", how="left")
    merged["has_xenium_ground_truth"] = _truthy(merged["has_xenium_ground_truth"])
    counts = (
        merged.groupby("split", dropna=False)
        .agg(num_spots=("spot_id", "count"), num_supervised_spots=("has_xenium_ground_truth", "sum"))
        .reset_index()
        .sort_values("split")
    )
    supervised_from_gt = set(supervised_spots)
    supervised_from_qc = set(qc.loc[_truthy(qc["has_xenium_ground_truth"]), "spot_id"].astype(str))
    _add_check(
        checklist,
        "spot_split_uniqueness",
        "pass" if duplicate_spots == 0 else "fail",
        "Every spot should have one fixed split assignment.",
        f"duplicate_spots={duplicate_spots}",
        str(split_path),
    )
    max_overlap = max(overlaps.values()) if overlaps else 0
    _add_check(
        checklist,
        "spot_split_disjointness",
        "pass" if max_overlap == 0 else "fail",
        "Train/validation/test split spot ids should be disjoint.",
        "; ".join(f"{key}={value}" for key, value in overlaps.items()),
        str(split_path),
    )
    _add_check(
        checklist,
        "gt_inventory_consistency",
        "pass" if supervised_from_gt == supervised_from_qc else "warn",
        "Supervised spots inferred from proportions should match the QC ground-truth inventory.",
        f"gt_table={len(supervised_from_gt)}; qc={len(supervised_from_qc)}; symmetric_difference={len(supervised_from_gt ^ supervised_from_qc)}",
        str(qc_path),
    )
    return {
        "split_counts": counts.to_dict(orient="records"),
        "duplicate_split_spots": duplicate_spots,
        "split_overlaps": overlaps,
    }


def _config_split_audit(
    primary_config_path: Path,
    split_guarded_config_path: Path,
    split_guarded_metrics_path: Path,
    checklist: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = load_config(primary_config_path) if primary_config_path.exists() else {}
    primary_data = primary.get("data", {})
    primary_training = primary.get("training", {})
    primary_eval = primary.get("evaluation", {})
    primary_has_split = bool(primary_data.get("split_path"))
    primary_eval_splits = primary_eval.get("eval_splits", primary_training.get("eval_splits", primary_data.get("eval_splits")))
    primary_train_splits = primary_training.get("train_splits", primary_data.get("train_splits"))
    _add_check(
        checklist,
        "primary_config_split_guard",
        "warn" if not primary_has_split else "pass",
        (
            "The primary matched-benchmark run is treated as a full-benchmark fit when no split_path is configured; "
            "it should not be described as an independent held-out test."
        ),
        f"split_path_present={primary_has_split}; train_splits={primary_train_splits}; eval_splits={primary_eval_splits}",
        str(primary_config_path),
    )

    guarded = load_config(split_guarded_config_path) if split_guarded_config_path.exists() else {}
    guarded_data = guarded.get("data", {})
    guarded_training = guarded.get("training", {})
    guarded_eval = guarded.get("evaluation", {})
    guarded_train = set(guarded_training.get("train_splits", guarded_data.get("train_splits", [])) or [])
    guarded_test = set(guarded_eval.get("eval_splits", guarded_training.get("eval_splits", guarded_data.get("eval_splits", []))) or [])
    guarded_disjoint = bool(guarded_train) and bool(guarded_test) and not (guarded_train & guarded_test)
    metrics_exist = split_guarded_metrics_path.exists()
    _add_check(
        checklist,
        "split_guarded_config",
        "pass" if guarded_disjoint and metrics_exist else "partial",
        "Supplementary leakage-resistant run must train and evaluate on disjoint fixed splits.",
        f"train_splits={sorted(guarded_train)}; eval_splits={sorted(guarded_test)}; metrics_exist={metrics_exist}",
        str(split_guarded_config_path),
    )
    metrics = {}
    if metrics_exist:
        frame = _read_csv(split_guarded_metrics_path)
        if not frame.empty:
            metrics = frame.iloc[-1].to_dict()
    return {
        "primary_has_split_guard": primary_has_split,
        "primary_train_splits": primary_train_splits,
        "primary_eval_splits": primary_eval_splits,
        "split_guarded_train_splits": sorted(guarded_train),
        "split_guarded_eval_splits": sorted(guarded_test),
        "split_guarded_metrics": metrics,
    }


def _baseline_fairness_audit(
    comparison_path: Path,
    ground_truth: pd.DataFrame,
    cell_types: list[str],
    supervised_spots: list[str],
    checklist: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    comparison = _read_csv(comparison_path)
    target = ground_truth.set_index("spot_id").loc[supervised_spots, cell_types].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    all_pass = True
    for _, row in comparison.iterrows():
        method = str(row.get("method", ""))
        pred_path = str(row.get("predictions_path", ""))
        pred_file = Path(pred_path)
        if not pred_path or not pred_file.exists():
            all_pass = False
            rows.append(
                {
                    "method": method,
                    "status": "fail",
                    "predictions_path": pred_path,
                    "num_supervised_spots": len(supervised_spots),
                    "num_missing_required_spots": len(supervised_spots),
                    "num_missing_cell_types": len(cell_types),
                    "reported_jsd": row.get("jsd", np.nan),
                    "recomputed_jsd": np.nan,
                    "abs_jsd_delta": np.nan,
                }
            )
            continue
        pred, diagnostics = _prediction_values(pred_file, supervised_spots, cell_types)
        recomputed_jsd = _jsd(pred, target)
        reported_jsd = float(row.get("jsd", np.nan))
        delta = abs(recomputed_jsd - reported_jsd) if np.isfinite(reported_jsd) else np.nan
        status = "pass"
        if diagnostics["num_missing_required_spots"] or diagnostics["num_missing_cell_types"]:
            status = "fail"
        elif np.isfinite(delta) and delta > 5e-4:
            status = "warn"
        all_pass = all_pass and status != "fail"
        rows.append(
            {
                "method": method,
                "status": status,
                "predictions_path": pred_path,
                "num_prediction_rows": diagnostics["num_prediction_rows"],
                "num_supervised_spots": len(supervised_spots),
                "num_missing_required_spots": diagnostics["num_missing_required_spots"],
                "num_missing_cell_types": diagnostics["num_missing_cell_types"],
                "reported_jsd": reported_jsd,
                "recomputed_jsd": recomputed_jsd,
                "abs_jsd_delta": delta,
            }
        )
    _add_check(
        checklist,
        "baseline_prediction_inputs",
        "pass" if all_pass else "fail",
        "Every baseline in the comparison table should cover the same supervised spots and cell-type panel.",
        f"methods={len(rows)}; supervised_spots={len(supervised_spots)}; failed={sum(1 for row in rows if row['status'] == 'fail')}",
        str(comparison_path),
    )
    return rows, {
        "num_methods": len(rows),
        "num_methods_pass": sum(1 for row in rows if row["status"] == "pass"),
        "num_methods_warn": sum(1 for row in rows if row["status"] == "warn"),
        "num_methods_fail": sum(1 for row in rows if row["status"] == "fail"),
    }


def _permutation_control(
    predictions_path: Path,
    ground_truth: pd.DataFrame,
    cell_types: list[str],
    supervised_spots: list[str],
    num_permutations: int,
    seed: int,
    checklist: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = ground_truth.set_index("spot_id").loc[supervised_spots, cell_types].to_numpy(dtype=np.float64)
    predictions, diagnostics = _prediction_values(predictions_path, supervised_spots, cell_types)
    observed = _jsd(predictions, target)
    rng = np.random.default_rng(seed)
    rows = []
    permuted_values = []
    for permutation in range(num_permutations):
        order = rng.permutation(len(supervised_spots))
        if len(order) > 1:
            attempts = 0
            while np.array_equal(order, np.arange(len(supervised_spots))) and attempts < 10:
                order = rng.permutation(len(supervised_spots))
                attempts += 1
        value = _jsd(predictions, target[order])
        permuted_values.append(value)
        rows.append({"permutation": permutation, "jsd": value})
    permuted = np.asarray(permuted_values, dtype=np.float64)
    empirical_p = float((np.sum(permuted <= observed) + 1) / (len(permuted) + 1)) if len(permuted) else 1.0
    summary = {
        "observed_jsd": observed,
        "num_permutations": int(num_permutations),
        "permuted_jsd_mean": float(permuted.mean()) if len(permuted) else 0.0,
        "permuted_jsd_min": float(permuted.min()) if len(permuted) else 0.0,
        "permuted_jsd_max": float(permuted.max()) if len(permuted) else 0.0,
        "empirical_p_lower_or_equal": empirical_p,
        "prediction_diagnostics": diagnostics,
    }
    _add_check(
        checklist,
        "label_permutation_negative_control",
        "pass" if len(permuted) and observed < float(permuted.min()) else "warn",
        "Correct spot-label pairing should outperform shuffled Xenium labels.",
        f"observed_jsd={observed:.6g}; min_permuted_jsd={summary['permuted_jsd_min']:.6g}; empirical_p={empirical_p:.6g}",
        str(predictions_path),
    )
    return rows, summary


def _write_markdown(summary: dict[str, Any], checklist: list[dict[str, Any]], path: Path) -> None:
    failed = [row for row in checklist if row["status"] == "fail"]
    warnings = [row for row in checklist if row["status"] == "warn"]
    guarded = summary.get("config_audit", {}).get("split_guarded_metrics", {})
    lines = [
        "# Leakage And Fairness Audit",
        "",
        f"Generated UTC: `{summary['generated_at_utc']}`",
        f"Overall status: `{summary['overall_status']}`",
        "",
        "## Key Results",
        "",
        f"- Fixed split overlap failures: `{summary['split_audit']['split_overlaps']}`",
        f"- Supervised Xenium spots used for matched evaluation: `{summary['num_supervised_spots']}`",
        f"- Baseline methods audited: `{summary['baseline_audit']['num_methods']}`",
        f"- Baseline input failures: `{summary['baseline_audit']['num_methods_fail']}`",
        f"- Label permutation observed JSD: `{summary['permutation_control']['observed_jsd']:.6g}`",
        f"- Label permutation minimum shuffled JSD: `{summary['permutation_control']['permuted_jsd_min']:.6g}`",
    ]
    if guarded:
        lines.extend(
            [
                f"- Split-guarded held-out test JSD: `{float(guarded.get('jsd', 0.0)):.6g}`",
                f"- Split-guarded train spots: `{int(float(guarded.get('num_train_spots', 0)))}`",
                f"- Split-guarded eval spots: `{int(float(guarded.get('num_eval_spots', 0)))}`",
                f"- Split-guarded supervised test spots: `{int(float(guarded.get('num_supervised_spots', 0)))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The primary WaveST-Gate run should be described as the matched Xenium-supervised benchmark result, not as a strict independent-test estimate when its config lacks a split guard. The split-guarded supplementary run provides the held-out sanity check: training uses only the fixed train split and evaluation uses the fixed test split.",
            "",
            "All formal baselines are audited against the same Xenium-supervised spot set and the same cell-type panel. The label-permutation negative control verifies that the reported matched performance is not reproduced by shuffled spot-label pairings.",
            "",
            "## Checklist",
            "",
            "| Check | Status | Metric |",
            "| --- | --- | --- |",
        ]
    )
    for row in checklist:
        lines.append(f"| {row['check']} | `{row['status']}` | {row['metric']} |")
    if failed or warnings:
        lines.extend(["", "## Caveats", ""])
        for row in failed + warnings:
            lines.append(f"- `{row['check']}`: {row['detail']} ({row['metric']})")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_leakage_fairness_audit(
    benchmark_dir: str | Path = "data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55",
    run_dir: str | Path = "results/nature_main/cytassist_rep2_radius55",
    output_dir: str | Path = "results/nature_leakage_fairness_audit",
    primary_config: str | Path = "experiments/nature_main/train_cytassist_rep2_radius55.yaml",
    split_guarded_config: str | Path = "experiments/nature_main/train_cytassist_rep2_radius55_split_guarded.yaml",
    split_guarded_run_dir: str | Path = "results/nature_main/cytassist_rep2_radius55_split_guarded",
    num_permutations: int = 200,
    seed: int = 13,
) -> dict[str, Any]:
    """Build machine-readable leakage and fairness evidence for reviewers."""

    benchmark_dir = Path(benchmark_dir)
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    split_guarded_run_dir = Path(split_guarded_run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_path = benchmark_dir / "xenium_cell_proportions.csv"
    split_path = benchmark_dir / "spot_splits.csv"
    qc_path = benchmark_dir / "spot_ground_truth_qc.csv"
    comparison_path = run_dir / "baseline_comparison" / "baseline_comparison.csv"
    predictions_path = run_dir / "predicted_proportions.csv"
    split_guarded_metrics_path = split_guarded_run_dir / "metrics.csv"

    ground_truth, cell_types, supervised_spots = _load_ground_truth(ground_truth_path)
    checklist: list[dict[str, Any]] = []
    split_audit = _split_audit(split_path, qc_path, supervised_spots, checklist)
    config_audit = _config_split_audit(
        Path(primary_config),
        Path(split_guarded_config),
        split_guarded_metrics_path,
        checklist,
    )
    baseline_rows, baseline_summary = _baseline_fairness_audit(
        comparison_path,
        ground_truth,
        cell_types,
        supervised_spots,
        checklist,
    )
    permutation_rows, permutation_summary = _permutation_control(
        predictions_path,
        ground_truth,
        cell_types,
        supervised_spots,
        num_permutations=num_permutations,
        seed=seed,
        checklist=checklist,
    )

    hard_failures = [row for row in checklist if row["status"] == "fail"]
    missing_guarded = not split_guarded_metrics_path.exists()
    if hard_failures:
        overall_status = "fail"
    elif missing_guarded:
        overall_status = "partial_missing_split_guarded_run"
    else:
        overall_status = "complete_with_primary_benchmark_caveat"

    baseline_table_path = output_dir / "baseline_fairness_table.csv"
    pd.DataFrame(baseline_rows).to_csv(baseline_table_path, index=False)
    permutation_table_path = output_dir / "label_permutation_control.csv"
    pd.DataFrame(permutation_rows).to_csv(permutation_table_path, index=False)
    checklist_path = output_dir / "leakage_fairness_checklist.csv"
    pd.DataFrame(checklist).to_csv(checklist_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "num_supervised_spots": len(supervised_spots),
        "num_cell_types": len(cell_types),
        "benchmark_dir": str(benchmark_dir),
        "run_dir": str(run_dir),
        "primary_config": str(primary_config),
        "split_guarded_config": str(split_guarded_config),
        "split_guarded_run_dir": str(split_guarded_run_dir),
        "split_audit": split_audit,
        "config_audit": config_audit,
        "baseline_audit": baseline_summary,
        "permutation_control": permutation_summary,
        "outputs": {
            "summary_json": str(output_dir / "leakage_fairness_audit.json"),
            "summary_markdown": str(output_dir / "leakage_fairness_audit.md"),
            "checklist_csv": str(checklist_path),
            "baseline_fairness_csv": str(baseline_table_path),
            "label_permutation_csv": str(permutation_table_path),
        },
    }
    summary_path = output_dir / "leakage_fairness_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path = output_dir / "leakage_fairness_audit.md"
    _write_markdown(summary, checklist, markdown_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit leakage risk and baseline fairness for WaveST-Gate.")
    parser.add_argument("--benchmark-dir", default="data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55")
    parser.add_argument("--run-dir", default="results/nature_main/cytassist_rep2_radius55")
    parser.add_argument("--output-dir", default="results/nature_leakage_fairness_audit")
    parser.add_argument("--primary-config", default="experiments/nature_main/train_cytassist_rep2_radius55.yaml")
    parser.add_argument("--split-guarded-config", default="experiments/nature_main/train_cytassist_rep2_radius55_split_guarded.yaml")
    parser.add_argument("--split-guarded-run-dir", default="results/nature_main/cytassist_rep2_radius55_split_guarded")
    parser.add_argument("--num-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    summary = build_leakage_fairness_audit(
        benchmark_dir=args.benchmark_dir,
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        primary_config=args.primary_config,
        split_guarded_config=args.split_guarded_config,
        split_guarded_run_dir=args.split_guarded_run_dir,
        num_permutations=args.num_permutations,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
