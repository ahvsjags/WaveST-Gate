"""Build manuscript-ready summary tables from WaveST-Gate evidence files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path("results/nature_main/cytassist_rep2_radius55")
DEFAULT_BENCHMARK_DIR = Path("data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55")
DEFAULT_OUTPUT_DIR = Path("results/nature_manuscript_tables")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _format_float(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.{digits}g}"


def _write_table(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def _first_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _table_benchmark(manifest_path: Path, qc_path: Path) -> pd.DataFrame:
    manifest = _read_json(manifest_path)
    qc = _read_csv(qc_path)
    gt_spots = int(manifest.get("num_spots_with_ground_truth", 0) or 0)
    total_spots = int(manifest.get("num_spots", 0) or 0)
    rows = [
        {
            "item": "spots",
            "value": total_spots,
            "notes": "All Visium/CytAssist spots in the benchmark coordinate table.",
            "evidence": str(manifest_path),
        },
        {
            "item": "xenium_supervised_spots",
            "value": gt_spots,
            "notes": "Spots with at least one aggregated Xenium cell and usable spot-level ground truth.",
            "evidence": str(manifest_path),
        },
        {
            "item": "xenium_cells",
            "value": int(manifest.get("num_cells", 0) or 0),
            "notes": "Typed Xenium cells after H&E coordinate alignment and QC.",
            "evidence": str(manifest_path),
        },
        {
            "item": "cell_types",
            "value": int(manifest.get("num_cell_types", 0) or 0),
            "notes": "Reference/ground-truth cell types used for proportion outputs.",
            "evidence": str(manifest_path),
        },
        {
            "item": "aggregation_radius",
            "value": manifest.get("spot_radius", ""),
            "notes": "Radius in aligned H&E/Xenium coordinate units.",
            "evidence": str(manifest_path),
        },
    ]
    if not qc.empty and "ground_truth_entropy" in qc.columns:
        rows.append(
            {
                "item": "mean_ground_truth_entropy",
                "value": float(qc["ground_truth_entropy"].mean()),
                "notes": "Mean entropy of spot-level Xenium cell-type proportions.",
                "evidence": str(qc_path),
            }
        )
    if not qc.empty and "xenium_cell_count" in qc.columns:
        rows.append(
            {
                "item": "median_xenium_cells_per_spot",
                "value": float(qc["xenium_cell_count"].median()),
                "notes": "Median aggregated Xenium cell count per Visium spot.",
                "evidence": str(qc_path),
            }
        )
    return pd.DataFrame(rows)


def _table_main_model(metrics_path: Path, proportion_manifest_path: Path) -> pd.DataFrame:
    metrics = _first_row(_read_csv(metrics_path))
    proportion_manifest = _read_json(proportion_manifest_path)
    rows = [
        ("training_step", metrics.get("step"), "Final training step used for the main model."),
        ("supervised_spots", metrics.get("num_supervised_spots"), "Xenium-supervised spots evaluated."),
        ("JSD", metrics.get("jsd"), "Primary cell-type proportion divergence, lower is better."),
        ("spotwise_cosine", metrics.get("spotwise_cosine"), "Spot-level proportion cosine similarity, higher is better."),
        ("mean_celltype_pearson", metrics.get("mean_celltype_pearson"), "Mean per-cell-type Pearson correlation."),
        ("proportion_RMSE", metrics.get("rmse"), "RMSE on supervised cell-type proportions."),
        ("expression_log1p_RMSE", metrics.get("expression_log1p_rmse"), "Expression reconstruction error."),
        ("uncertainty_error_pearson", metrics.get("uncertainty_error_pearson"), "Correlation of model uncertainty with spot error."),
        ("uncertainty_risk_gap", metrics.get("uncertainty_risk_gap"), "High-minus-low uncertainty JSD gap."),
        ("proportion_map_cell_types", proportion_manifest.get("num_cell_types"), "Number of cell types rendered in proportion-map analysis."),
        ("priority_cell_types_mapped", len(proportion_manifest.get("selected_cell_types", [])), "Priority breast cancer cell types rendered as manuscript maps."),
    ]
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "value": _format_float(value) if metric not in {"training_step", "supervised_spots", "proportion_map_cell_types", "priority_cell_types_mapped"} else value,
                "interpretation": notes,
                "evidence": str(metrics_path if "map" not in metric else proportion_manifest_path),
            }
            for metric, value, notes in rows
        ]
    )


def _table_baselines(comparison_path: Path, bootstrap_path: Path) -> pd.DataFrame:
    comparison = _read_csv(comparison_path)
    bootstrap = _read_csv(bootstrap_path)
    if comparison.empty:
        return pd.DataFrame()
    keep = [
        "rank_by_jsd",
        "method",
        "source",
        "jsd",
        "spotwise_cosine",
        "mean_celltype_pearson",
        "rmse",
        "runtime_seconds",
        "peak_cuda_memory_mb",
        "paired_permutation_p",
    ]
    table = comparison[[col for col in keep if col in comparison.columns]].copy()
    if not bootstrap.empty and "method" in bootstrap.columns:
        boot_keep = ["method", "jsd_mean", "jsd_std", "spotwise_cosine_mean", "spotwise_cosine_std"]
        table = table.merge(bootstrap[[col for col in boot_keep if col in bootstrap.columns]], on="method", how="left")
    for col in table.columns:
        if col not in {"method", "source"}:
            table[col] = table[col].map(lambda value: _format_float(value))
    return table


def _table_ablations(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    table = frame.copy()
    if "ablation" in table.columns:
        full = table.loc[table["ablation"].astype(str).eq("full")]
        reference = full.iloc[0].to_dict() if not full.empty else table.iloc[0].to_dict()
        for metric in ["jsd", "spotwise_cosine", "mean_celltype_pearson", "uncertainty_error_pearson"]:
            if metric in table.columns:
                ref = _float(reference.get(metric))
                table[f"delta_vs_full_{metric}"] = table[metric].map(lambda value: (_float(value) or np.nan) - (ref or np.nan))
    keep = [
        "ablation",
        "jsd",
        "delta_vs_full_jsd",
        "spotwise_cosine",
        "delta_vs_full_spotwise_cosine",
        "mean_celltype_pearson",
        "delta_vs_full_mean_celltype_pearson",
        "uncertainty_error_pearson",
        "delta_vs_full_uncertainty_error_pearson",
    ]
    table = table[[col for col in keep if col in table.columns]].copy()
    for col in table.columns:
        if col != "ablation":
            table[col] = table[col].map(lambda value: _format_float(value))
    return table


def _table_reliability_boundary_niche(nature_dir: Path, pathology_dir: Path) -> pd.DataFrame:
    reliability = _read_json(nature_dir / "reliability_summary.json")
    boundary = _read_json(nature_dir / "boundary_summary.json")
    niche = _read_json(nature_dir / "niche_summary.json")
    pathology = _read_csv(pathology_dir / "pathology_class_summary.csv")
    rows = [
        {
            "claim_area": "reliability_calibration",
            "metric": "uncertainty_error_pearson",
            "value": _format_float(reliability.get("uncertainty_error_pearson")),
            "evidence": str(nature_dir / "reliability_summary.json"),
            "interpretation": "Positive correlation means uncertainty tracks spot-level prediction error.",
        },
        {
            "claim_area": "reliability_calibration",
            "metric": "calibration_bin_pearson",
            "value": _format_float(reliability.get("calibration_bin_pearson")),
            "evidence": str(nature_dir / "reliability_summary.json"),
            "interpretation": "Calibration-bin trend supports reliability semantics.",
        },
        {
            "claim_area": "boundary_preservation",
            "metric": "boundary_to_interior_jump_ratio",
            "value": _format_float(boundary.get("boundary_to_interior_jump_ratio")),
            "evidence": str(nature_dir / "boundary_summary.json"),
            "interpretation": "Higher boundary than interior jumps indicate protected tissue transitions.",
        },
        {
            "claim_area": "biological_niche",
            "metric": "num_niches",
            "value": niche.get("num_niches", ""),
            "evidence": str(nature_dir / "niche_summary.json"),
            "interpretation": "Named tumor/stromal/immune niche programs are available for interpretation.",
        },
    ]
    if not pathology.empty and "agreement_rate" in pathology.columns:
        agreement = pathology["agreement_rate"].astype(float)
        metric_name = "mean_pathology_agreement_rate"
        if "num_spots" in pathology.columns:
            weights = pathology["num_spots"].astype(float)
            if float(weights.sum()) > 0:
                agreement_value = float(np.average(agreement, weights=weights))
                metric_name = "spot_weighted_pathology_agreement_rate"
            else:
                agreement_value = float(agreement.mean())
        else:
            agreement_value = float(agreement.mean())
        rows.append(
            {
                "claim_area": "external_pathology_validation",
                "metric": metric_name,
                "value": _format_float(agreement_value),
                "evidence": str(pathology_dir / "pathology_class_summary.csv"),
                "interpretation": "External pathology-class correspondence across public datasets.",
            }
        )
    return pd.DataFrame(rows)


def _table_external(no_retuning_path: Path, matched_gt_path: Path, minimal_multisample_path: Path) -> pd.DataFrame:
    no_retuning = _read_csv(no_retuning_path)
    matched_gt = _read_csv(matched_gt_path)
    minimal = _read_csv(minimal_multisample_path)
    rows: list[dict[str, Any]] = []
    if not no_retuning.empty:
        rows.append(
            {
                "setting": "external_no_retuning",
                "datasets": int(no_retuning["dataset"].nunique()) if "dataset" in no_retuning.columns else len(no_retuning),
                "spots": int(no_retuning["num_spots"].astype(float).sum()) if "num_spots" in no_retuning.columns else "",
                "primary_metric": "mean_expression_log1p_rmse",
                "value": _format_float(no_retuning["expression_log1p_rmse"].astype(float).mean()) if "expression_log1p_rmse" in no_retuning.columns else "",
                "evidence": str(no_retuning_path),
            }
        )
    if not matched_gt.empty:
        row = matched_gt.iloc[0].to_dict()
        rows.append(
            {
                "setting": "Rep1_no_retuning_matched_GT",
                "datasets": 1,
                "spots": row.get("num_spots", ""),
                "primary_metric": "WaveST-Gate JSD / rank",
                "value": f"{_format_float(row.get('no_retuning_model_jsd'))} / {row.get('no_retuning_wavestgate_rank_by_jsd', '')}",
                "evidence": str(matched_gt_path),
            }
        )
        rows.append(
            {
                "setting": "Rep1_minimal_retuning_matched_GT",
                "datasets": 1,
                "spots": row.get("minimal_retune_test_spots", ""),
                "primary_metric": "WaveST-Gate test JSD / rank",
                "value": f"{_format_float(row.get('minimal_retune_test_jsd'))} / {row.get('minimal_retune_wavestgate_rank_by_jsd', '')}",
                "evidence": str(matched_gt_path),
            }
        )
    if not minimal.empty:
        top = minimal.iloc[0].to_dict()
        rows.append(
            {
                "setting": "minimal_retuning_multisample_summary",
                "datasets": top.get("num_datasets", ""),
                "spots": "",
                "primary_metric": "top method mean JSD +/- SD",
                "value": f"{top.get('method', '')}: {_format_float(top.get('jsd_mean'))} +/- {_format_float(top.get('jsd_std'))}",
                "evidence": str(minimal_multisample_path),
            }
        )
    return pd.DataFrame(rows)


def _table_robustness(robustness_path: Path, patch_path: Path) -> pd.DataFrame:
    robust = _read_csv(robustness_path)
    patch = _read_csv(patch_path)
    rows: list[dict[str, Any]] = []
    if not robust.empty and "scenario" in robust.columns:
        for scenario, group in robust.groupby("scenario", dropna=False):
            row = {
                "scenario": scenario,
                "n_rows": len(group),
                "jsd_mean": _format_float(group["jsd"].astype(float).mean()) if "jsd" in group.columns else "",
                "jsd_max": _format_float(group["jsd"].astype(float).max()) if "jsd" in group.columns else "",
                "spotwise_cosine_min": _format_float(group["spotwise_cosine"].astype(float).min()) if "spotwise_cosine" in group.columns else "",
                "levels": ";".join(sorted({str(v) for v in group.get("level", pd.Series(dtype=str)).dropna().tolist() if str(v) != ""})),
                "evidence": str(robustness_path),
            }
            rows.append(row)
    if not patch.empty:
        rows.append(
            {
                "scenario": "patch_size",
                "n_rows": len(patch),
                "jsd_mean": _format_float(patch["jsd"].astype(float).mean()) if "jsd" in patch.columns else "",
                "jsd_max": _format_float(patch["jsd"].astype(float).max()) if "jsd" in patch.columns else "",
                "spotwise_cosine_min": _format_float(patch["spotwise_cosine"].astype(float).min()) if "spotwise_cosine" in patch.columns else "",
                "levels": ";".join(sorted({str(v) for v in patch.get("patch_size", pd.Series(dtype=str)).dropna().tolist()})),
                "evidence": str(patch_path),
            }
        )
    return pd.DataFrame(rows)


def _table_split_sensitivity(split_dir: Path) -> pd.DataFrame:
    original = _read_csv(split_dir / "original_split_gt_summary.csv")
    aggregate = _read_csv(split_dir / "split_sensitivity_aggregate.csv")
    rows: list[dict[str, Any]] = []
    if not original.empty:
        for row in original.to_dict(orient="records"):
            rows.append(
                {
                    "analysis": "primary_spatial_holdout",
                    "split": row.get("split", ""),
                    "num_spots": row.get("num_spots", ""),
                    "num_supervised_spots": row.get("num_supervised_spots", ""),
                    "num_seeds": "",
                    "mean_jsd": "",
                    "sd_jsd": "",
                    "mean_spotwise_cosine": "",
                    "mean_celltype_pearson": "",
                    "interpretation": "Original spatial holdout retained for primary reporting; validation has no Xenium GT and is not used as a supervised model-selection endpoint.",
                    "evidence": str(split_dir / "original_split_gt_summary.csv"),
                }
            )
    if not aggregate.empty:
        for row in aggregate.to_dict(orient="records"):
            rows.append(
                {
                    "analysis": "supplementary_gt_stratified_sensitivity",
                    "split": row.get("split", ""),
                    "num_spots": "",
                    "num_supervised_spots": _format_float(row.get("mean_supervised_spots")),
                    "num_seeds": row.get("num_seeds", ""),
                    "mean_jsd": _format_float(row.get("mean_jsd")),
                    "sd_jsd": _format_float(row.get("sd_jsd")),
                    "mean_spotwise_cosine": _format_float(row.get("mean_spotwise_cosine")),
                    "mean_celltype_pearson": _format_float(row.get("mean_celltype_pearson")),
                    "interpretation": "GT-stratified supplementary splits confirm that the main prediction error is stable when validation/test each contain supervised spots.",
                    "evidence": str(split_dir / "split_sensitivity_aggregate.csv"),
                }
            )
    return pd.DataFrame(rows)


def _table_imagegate_supplement(image_contribution_dir: Path) -> pd.DataFrame:
    comparison = _read_csv(image_contribution_dir / "imagegate_run_comparison.csv")
    texture = _read_csv(image_contribution_dir / "image_contribution_texture_groups.csv")
    rows: list[dict[str, Any]] = []
    for row in comparison.to_dict(orient="records") if not comparison.empty else []:
        rows.append(
            {
                "analysis": "run_level_imagegate_control",
                "group": row.get("run", row.get("setting", "")),
                "jsd": _format_float(row.get("jsd")),
                "spotwise_cosine": _format_float(row.get("spotwise_cosine")),
                "mean_celltype_pearson": _format_float(row.get("mean_celltype_pearson")),
                "mean_image_gate": _format_float(row.get("mean_image_gate")),
                "mean_raw_image_gate": _format_float(row.get("mean_raw_image_gate")),
                "paired_noimage_minus_image_jsd": "",
                "interpretation": "Image-gate-enhanced and matched no-image controls separate morphology contribution from expression-only performance.",
                "evidence": str(image_contribution_dir / "imagegate_run_comparison.csv"),
            }
        )
    for row in texture.to_dict(orient="records") if not texture.empty else []:
        rows.append(
            {
                "analysis": "texture_stratified_paired_error",
                "group": row.get("group", ""),
                "jsd": _format_float(row.get("mean_imagegate_jsd")),
                "spotwise_cosine": "",
                "mean_celltype_pearson": "",
                "mean_image_gate": _format_float(row.get("mean_image_gate")),
                "mean_raw_image_gate": _format_float(row.get("mean_raw_image_gate")),
                "paired_noimage_minus_image_jsd": _format_float(
                    row.get("mean_paired_jsd_improvement_noimage_minus_imagegate", row.get("mean_paired_jsd_improvement"))
                ),
                "interpretation": "Positive paired no-image-minus-image JSD means the image branch improves supervised spot errors in that morphology stratum.",
                "evidence": str(image_contribution_dir / "image_contribution_texture_groups.csv"),
            }
        )
    return pd.DataFrame(rows)


def _table_benchmark_sensitivity(benchmark_sensitivity_dir: Path) -> pd.DataFrame:
    metrics = _read_csv(benchmark_sensitivity_dir / "radius_cell_count_sensitivity.csv")
    coverage = _read_csv(benchmark_sensitivity_dir / "radius_coverage_summary.csv")
    rows: list[dict[str, Any]] = []
    if not coverage.empty:
        for row in coverage.to_dict(orient="records"):
            rows.append(
                {
                    "analysis": "radius_coverage",
                    "radius": _format_float(row.get("radius")),
                    "min_xenium_cells": "",
                    "num_spots_passing_threshold": row.get(
                        "num_spots_with_cells",
                        row.get("num_spots_with_ground_truth", row.get("num_spots_passing_threshold", "")),
                    ),
                    "jsd": "",
                    "spotwise_cosine": "",
                    "mean_celltype_pearson": "",
                    "rmse": "",
                    "interpretation": "Coverage of Xenium-supervised spots under alternative aggregation radii.",
                    "evidence": str(benchmark_sensitivity_dir / "radius_coverage_summary.csv"),
                }
            )
    if not metrics.empty:
        primary = metrics.loc[(metrics["radius"].astype(float) == 55.0) & (metrics["min_xenium_cells"].astype(float) == 1.0)]
        best = _first_row(primary) or _first_row(metrics.sort_values("jsd"))
        for row in metrics.to_dict(orient="records"):
            radius = _float(row.get("radius"))
            min_cells = _float(row.get("min_xenium_cells"))
            rows.append(
                {
                    "analysis": "radius_cell_count_metric",
                    "radius": _format_float(radius),
                    "min_xenium_cells": _format_float(min_cells),
                    "num_spots_passing_threshold": row.get("num_spots_passing_threshold", ""),
                    "jsd": _format_float(row.get("jsd")),
                    "spotwise_cosine": _format_float(row.get("spotwise_cosine")),
                    "mean_celltype_pearson": _format_float(row.get("mean_celltype_pearson")),
                    "rmse": _format_float(row.get("rmse")),
                    "interpretation": (
                        "Primary radius/min-cell setting."
                        if radius == _float(best.get("radius")) and min_cells == _float(best.get("min_xenium_cells"))
                        else "Sensitivity row for aggregation radius and Xenium cell-count confidence threshold."
                    ),
                    "evidence": str(benchmark_sensitivity_dir / "radius_cell_count_sensitivity.csv"),
                }
            )
    return pd.DataFrame(rows)


def _table_rep1_budget_curve(curve_path: Path) -> pd.DataFrame:
    curve = _read_csv(curve_path)
    if curve.empty:
        return pd.DataFrame()
    table = curve.copy()
    keep = [
        "budget_steps",
        "best_baseline_method",
        "best_baseline_jsd",
        "beats_best_baseline",
        "jsd_margin_vs_best_baseline",
        "jsd",
        "spotwise_cosine",
        "mean_celltype_pearson",
        "rmse",
        "num_supervised_spots",
        "num_train_spots",
        "num_eval_spots",
    ]
    table = table[[col for col in keep if col in table.columns]].copy()
    for col in table.columns:
        if col not in {"best_baseline_method", "beats_best_baseline"}:
            table[col] = table[col].map(lambda value: _format_float(value))
    table["interpretation"] = table["budget_steps"].astype(str).map(
        lambda budget: (
            "No-retuning domain-shift result; reported honestly even when a baseline is better."
            if budget in {"0", "0.0"}
            else "Minimal Rep1 adaptation budget evaluated on the held-out Rep1 test split."
        )
    )
    table["evidence"] = str(curve_path)
    return table


def _write_markdown_index(tables: dict[str, str], output_dir: Path) -> Path:
    lines = [
        "# WaveST-Gate Manuscript Tables",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "These tables are derived from the audited Nature-level evidence files.",
        "",
        "| Table | File | Intended manuscript use |",
        "| --- | --- | --- |",
    ]
    descriptions = {
        "benchmark_summary": "Benchmark/data protocol summary.",
        "main_model_performance": "Primary model metrics and map coverage.",
        "baseline_comparison": "Formal baseline comparison with runtime/significance/statistics.",
        "ablation_delta": "Module and modality ablation deltas.",
        "reliability_boundary_niche": "Reliability, boundary, niche, and pathology validation summary.",
        "external_generalization": "No-retuning and minimal-retuning external generalization summary.",
        "robustness_summary": "Stress-test robustness summary.",
        "split_sensitivity": "Primary split GT inventory plus supplementary GT-stratified split sensitivity.",
        "imagegate_supplement": "Image-gate-enhanced versus no-image morphology contribution controls.",
        "benchmark_sensitivity": "Xenium aggregation radius and cell-count confidence-threshold sensitivity.",
        "rep1_retune_budget_curve": "Rep1 no-retuning domain shift and minimal-retuning budget curve.",
    }
    for key, path in tables.items():
        lines.append(f"| `{key}` | `{path}` | {descriptions.get(key, '')} |")
    lines.extend(
        [
            "",
            "The Zenodo DOI is intentionally not filled here. It must come from",
            "`results/nature_release/zenodo_deposition_result.json` after a real token-backed deposition.",
        ]
    )
    path = output_dir / "manuscript_tables.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_manuscript_tables(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    nature_dir = run_dir / "nature_analysis"
    tables = {
        "benchmark_summary": _write_table(
            _table_benchmark(Path(args.benchmark_manifest), Path(args.spot_qc)),
            output_dir / "table_1_benchmark_summary.csv",
        ),
        "main_model_performance": _write_table(
            _table_main_model(Path(args.main_metrics), nature_dir / "proportion_maps" / "proportion_map_manifest.json"),
            output_dir / "table_2_main_model_performance.csv",
        ),
        "baseline_comparison": _write_table(
            _table_baselines(run_dir / "baseline_comparison" / "baseline_comparison.csv", run_dir / "baseline_comparison" / "baseline_split_bootstrap_summary.csv"),
            output_dir / "table_3_baseline_comparison.csv",
        ),
        "ablation_delta": _write_table(
            _table_ablations(run_dir / "ablations250" / "ablation_summary.csv"),
            output_dir / "table_4_ablation_delta.csv",
        ),
        "reliability_boundary_niche": _write_table(
            _table_reliability_boundary_niche(nature_dir, Path(args.external_pathology_dir)),
            output_dir / "table_5_reliability_boundary_niche.csv",
        ),
        "external_generalization": _write_table(
            _table_external(
                Path(args.external_no_retuning),
                Path(args.external_matched_gt),
                Path(args.minimal_retune_multisample),
            ),
            output_dir / "table_6_external_generalization.csv",
        ),
        "robustness_summary": _write_table(
            _table_robustness(run_dir / "robustness" / "robustness_summary.csv", run_dir / "patch_size_robustness" / "patch_size_summary.csv"),
            output_dir / "table_7_robustness_summary.csv",
        ),
        "split_sensitivity": _write_table(
            _table_split_sensitivity(Path(args.split_sensitivity_dir)),
            output_dir / "table_8_split_sensitivity.csv",
        ),
        "imagegate_supplement": _write_table(
            _table_imagegate_supplement(Path(args.image_contribution_dir)),
            output_dir / "table_9_imagegate_supplement.csv",
        ),
        "benchmark_sensitivity": _write_table(
            _table_benchmark_sensitivity(Path(args.benchmark_sensitivity_dir)),
            output_dir / "table_10_benchmark_sensitivity.csv",
        ),
        "rep1_retune_budget_curve": _write_table(
            _table_rep1_budget_curve(Path(args.rep1_budget_curve)),
            output_dir / "table_11_rep1_retune_budget_curve.csv",
        ),
    }
    index_path = _write_markdown_index(tables, output_dir)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "tables": tables,
        "markdown_index": str(index_path),
    }
    manifest_path = output_dir / "manuscript_tables_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build manuscript-ready tables from WaveST-Gate evidence files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--benchmark-manifest", default=str(DEFAULT_BENCHMARK_DIR / "xenium_visium_benchmark_manifest.json"))
    parser.add_argument("--spot-qc", default=str(DEFAULT_BENCHMARK_DIR / "spot_ground_truth_qc.csv"))
    parser.add_argument("--main-metrics", default=str(DEFAULT_RUN_DIR / "metrics.csv"))
    parser.add_argument("--external-no-retuning", default="results/nature_external_no_retuning/external_no_retuning_summary.csv")
    parser.add_argument("--external-matched-gt", default="results/nature_external_matched_gt/external_matched_gt_summary.csv")
    parser.add_argument(
        "--minimal-retune-multisample",
        default="results/nature_matched_multisample_baselines_minimal_retune/matched_multisample_baseline_summary.csv",
    )
    parser.add_argument("--external-pathology-dir", default="results/nature_external_pathology_validation")
    parser.add_argument("--split-sensitivity-dir", default=str(DEFAULT_RUN_DIR / "split_sensitivity"))
    parser.add_argument("--benchmark-sensitivity-dir", default=str(DEFAULT_RUN_DIR / "benchmark_sensitivity"))
    parser.add_argument("--image-contribution-dir", default=str(DEFAULT_RUN_DIR.parent / f"{DEFAULT_RUN_DIR.name}_imagegate" / "image_contribution"))
    parser.add_argument(
        "--rep1-budget-curve",
        default="results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/rep1_retune_budget_curve/rep1_minimal_retune_budget_curve.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = build_manuscript_tables(args)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
