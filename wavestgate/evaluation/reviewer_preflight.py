"""Generate a reviewer-facing preflight dossier for WaveST-Gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("results/nature_reviewer_preflight")


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _evidence(path: str | Path, role: str) -> dict[str, Any]:
    path = Path(path)
    return {"role": role, "path": str(path), "exists": path.exists()}


def _record_status(readiness: dict[str, Any], requirement: str) -> str:
    for record in readiness.get("records", []):
        if str(record.get("requirement", "")) == requirement:
            return str(record.get("status", ""))
    return ""


def _stage_records(readiness: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in readiness.get("records", []):
        grouped[str(record.get("stage", ""))].append(
            {
                "requirement": record.get("requirement", ""),
                "status": record.get("status", ""),
                "evidence_path": record.get("evidence_path", ""),
                "detail": record.get("detail", ""),
            }
        )
    return dict(grouped)


def _first_non_model_baseline(table: pd.DataFrame) -> dict[str, Any]:
    if table.empty or "method" not in table.columns:
        return {}
    rows = table.loc[~table["method"].astype(str).eq("WaveST-Gate")]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _headline_claims(
    readiness: dict[str, Any],
    tables_dir: Path,
    run_dir: Path,
    benchmark_dir: Path,
    release_dir: Path,
    docs_dir: Path,
    external_dir: Path,
    external_matched_gt_dir: Path,
    minimal_retune_dir: Path,
    benchmark_datasheet_dir: Path,
    completion_audit_dir: Path,
) -> list[dict[str, Any]]:
    baseline = _read_csv(tables_dir / "table_3_baseline_comparison.csv")
    model = _read_csv(tables_dir / "table_2_main_model_performance.csv")
    reliability = _read_csv(tables_dir / "table_5_reliability_boundary_niche.csv")
    split_sensitivity = _read_csv(tables_dir / "table_8_split_sensitivity.csv")
    benchmark_sensitivity = _read_csv(tables_dir / "table_10_benchmark_sensitivity.csv")
    rep1_budget = _read_csv(tables_dir / "table_11_rep1_retune_budget_curve.csv")
    best_baseline = _first_non_model_baseline(baseline)
    metrics = {str(row.get("metric")): row.get("value") for _, row in model.iterrows()} if not model.empty else {}
    rel = {str(row.get("metric")): row.get("value") for _, row in reliability.iterrows()} if not reliability.empty else {}
    split_rows = {
        (str(row.get("analysis", "")), str(row.get("split", ""))): row
        for _, row in split_sensitivity.iterrows()
    } if not split_sensitivity.empty else {}
    radius55_rows = pd.DataFrame()
    if not benchmark_sensitivity.empty:
        radius55_rows = benchmark_sensitivity.loc[
            benchmark_sensitivity.get("analysis", pd.Series(dtype=str)).astype(str).eq("radius_cell_count_metric")
            & pd.to_numeric(benchmark_sensitivity.get("radius", pd.Series(dtype=float)), errors="coerce").eq(55.0)
            & pd.to_numeric(benchmark_sensitivity.get("min_xenium_cells", pd.Series(dtype=float)), errors="coerce").eq(1.0)
        ]
    primary_radius = radius55_rows.iloc[0].to_dict() if not radius55_rows.empty else {}
    rep1_zero_rows = pd.DataFrame()
    rep1_first_beating = {}
    if not rep1_budget.empty and "budget_steps" in rep1_budget.columns:
        rep1_zero_rows = rep1_budget.loc[pd.to_numeric(rep1_budget["budget_steps"], errors="coerce").eq(0)]
        beating = rep1_budget.loc[rep1_budget.get("beats_best_baseline", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"})].copy()
        if not beating.empty:
            beating["_budget_numeric"] = pd.to_numeric(beating["budget_steps"], errors="coerce")
            rep1_first_beating = beating.sort_values("_budget_numeric").iloc[0].to_dict()
    rep1_zero = rep1_zero_rows.iloc[0].to_dict() if not rep1_zero_rows.empty else {}
    zenodo_payload = _read_json(release_dir / "zenodo_deposition_result.json")
    release_published = zenodo_payload.get("release_status") == "zenodo_published" and bool(zenodo_payload.get("doi"))
    return [
        {
            "claim": "Reproducible Xenium-to-Visium breast cancer benchmark",
            "status": _record_status(readiness, "spot-level Xenium aggregation artifacts"),
            "response_point": "The benchmark is a fixed spot-level ground-truth protocol rather than an ad hoc evaluation set.",
            "key_values": {
                "spots": _read_json(benchmark_dir / "xenium_visium_benchmark_manifest.json").get("num_spots", ""),
                "xenium_supervised_spots": _read_json(benchmark_dir / "xenium_visium_benchmark_manifest.json").get("num_spots_with_ground_truth", ""),
                "xenium_cells": _read_json(benchmark_dir / "xenium_visium_benchmark_manifest.json").get("num_cells", ""),
                "cell_types": _read_json(benchmark_dir / "xenium_visium_benchmark_manifest.json").get("num_cell_types", ""),
                "datasheet_status": _record_status(readiness, "benchmark datasheet and data dictionary"),
                "primary_val_supervised_spots": split_rows.get(("primary_spatial_holdout", "val"), {}).get("num_supervised_spots", ""),
                "supplementary_gt_stratified_val_jsd": split_rows.get(("supplementary_gt_stratified_sensitivity", "val"), {}).get("mean_jsd", ""),
                "radius55_min1_jsd": primary_radius.get("jsd", ""),
            },
            "evidence": [
                _evidence(benchmark_dir / "xenium_visium_benchmark_manifest.json", "benchmark manifest"),
                _evidence(benchmark_dir / "xenium_cell_counts.csv", "cell counts"),
                _evidence(benchmark_dir / "xenium_cell_proportions.csv", "cell proportions"),
                _evidence(benchmark_dir / "spot_ground_truth_qc.csv", "spot QC"),
                _evidence(benchmark_dir / "spot_splits.csv", "fixed splits"),
                _evidence(tables_dir / "table_8_split_sensitivity.csv", "split sensitivity table"),
                _evidence(tables_dir / "table_10_benchmark_sensitivity.csv", "radius/cell-count sensitivity table"),
                _evidence(docs_dir / "xenium_to_visium_benchmark_protocol.md", "protocol"),
                _evidence(benchmark_datasheet_dir / "benchmark_datasheet.json", "benchmark datasheet"),
                _evidence(benchmark_datasheet_dir / "benchmark_datasheet.md", "data dictionary markdown"),
            ],
        },
        {
            "claim": "Accurate real-data multimodal spatial deconvolution",
            "status": _record_status(readiness, "checkpoint, predictions, reliability maps, and training curve"),
            "response_point": "The main real-data run provides checkpoint, predictions, maps, curves, and supervised metrics.",
            "key_values": {"JSD": metrics.get("JSD", ""), "spotwise_cosine": metrics.get("spotwise_cosine", ""), "mean_celltype_pearson": metrics.get("mean_celltype_pearson", "")},
            "evidence": [
                _evidence(run_dir / "checkpoint.pt", "checkpoint"),
                _evidence(run_dir / "predicted_proportions.csv", "predicted proportions"),
                _evidence(run_dir / "metrics.csv", "metrics"),
                _evidence(run_dir / "nature_analysis" / "proportion_maps", "proportion maps"),
            ],
        },
        {
            "claim": "Fair comparison against strong deconvolution baselines",
            "status": _record_status(readiness, "formal baseline table with shared genes/reference/supervised spots"),
            "response_point": "Formal baselines use matched supervised spots, shared panels where applicable, runtime/memory reporting, and paired statistics.",
            "key_values": {"num_methods": len(baseline), "best_baseline": best_baseline.get("method", ""), "best_baseline_jsd": best_baseline.get("jsd", "")},
            "evidence": [
                _evidence(run_dir / "baseline_comparison" / "baseline_comparison.csv", "baseline comparison"),
                _evidence(run_dir / "baseline_comparison" / "baseline_split_bootstrap_summary.csv", "bootstrap summary"),
                _evidence(run_dir / "baseline_comparison" / "baseline_bootstrap_paired_improvement.csv", "paired statistics"),
            ],
        },
        {
            "claim": "Ablations support component necessity",
            "status": _record_status(readiness, "module and modality ablations"),
            "response_point": "The ablation panel isolates wavelet morphology, agents, gate, uncertainty, boundary loss, local refinement, and modality-only settings.",
            "key_values": {"num_ablations": len(_read_csv(run_dir / "ablations250" / "ablation_summary.csv"))},
            "evidence": [_evidence(run_dir / "ablations250" / "ablation_summary.csv", "ablation summary")],
        },
        {
            "claim": "Reliability gate is calibrated and interpretable",
            "status": _record_status(readiness, "uncertainty calibration and modality reliability evidence"),
            "response_point": "Uncertainty-error correlations, calibration bins, risk coverage, gate maps, and failure cases support reliability semantics.",
            "key_values": {"uncertainty_error_pearson": rel.get("uncertainty_error_pearson", ""), "calibration_bin_pearson": rel.get("calibration_bin_pearson", "")},
            "evidence": [
                _evidence(run_dir / "nature_analysis" / "reliability_summary.json", "reliability summary"),
                _evidence(run_dir / "nature_analysis" / "risk_coverage_curve.csv", "risk coverage"),
                _evidence(run_dir / "nature_analysis" / "uncertainty_calibration_bins.csv", "calibration bins"),
            ],
        },
        {
            "claim": "Morphology-aware boundary preservation",
            "status": _record_status(readiness, "typed boundary sharpness and marker validation"),
            "response_point": "Boundary sharpness, typed transitions, marker validation, H&E overlays, and no-boundary-loss comparisons address over-smoothing concerns.",
            "key_values": {"boundary_to_interior_jump_ratio": rel.get("boundary_to_interior_jump_ratio", "")},
            "evidence": [
                _evidence(run_dir / "nature_analysis" / "boundary_summary.json", "boundary summary"),
                _evidence(run_dir / "nature_analysis" / "boundary_marker_validation.csv", "boundary marker validation"),
                _evidence(run_dir / "nature_analysis" / "boundary_type_summary.csv", "boundary type summary"),
            ],
        },
        {
            "claim": "Tumor-immune-stromal niche interpretation",
            "status": _record_status(readiness, "tumor-immune-stromal niche outputs"),
            "response_point": "Niche assignments are linked to composition, marker enrichment, Xenium neighborhoods, pathology correspondence, gate reliability, and agent attention.",
            "key_values": {"num_niches": rel.get("num_niches", "")},
            "evidence": [
                _evidence(run_dir / "nature_analysis" / "niche_summary.json", "niche summary"),
                _evidence(run_dir / "nature_analysis" / "niche_biological_summary.csv", "niche biological summary"),
                _evidence(run_dir / "nature_analysis" / "niche_xenium_neighborhood_summary.csv", "Xenium neighborhood summary"),
            ],
        },
        {
            "claim": "External generalization and robustness",
            "status": "complete" if readiness.get("stage_summary", {}).get("8. External generalization", {}).get("status") == "complete" and readiness.get("stage_summary", {}).get("9. Robustness", {}).get("status") == "complete" else "partial",
            "response_point": "External no-retuning, matched-ground-truth transfer, minimal-retuning multi-sample results, and stress tests address sample-specific overfitting.",
            "key_values": {
                "rep1_no_retune_jsd": rep1_zero.get("jsd", ""),
                "rep1_best_baseline_jsd": rep1_zero.get("best_baseline_jsd", ""),
                "min_retune_budget_beating_best_baseline": rep1_first_beating.get("budget_steps", ""),
                "min_retune_budget_jsd": rep1_first_beating.get("jsd", ""),
            },
            "evidence": [
                _evidence(external_dir / "external_no_retuning_summary.csv", "external no-retuning"),
                _evidence(external_matched_gt_dir / "external_matched_gt_summary.csv", "external matched GT"),
                _evidence(minimal_retune_dir / "matched_multisample_baseline_summary.csv", "minimal-retuning multi-sample summary"),
                _evidence(tables_dir / "table_11_rep1_retune_budget_curve.csv", "Rep1 retune budget table"),
                _evidence(run_dir / "robustness" / "robustness_summary.csv", "robustness summary"),
                _evidence(run_dir / "patch_size_robustness" / "patch_size_summary.csv", "patch-size robustness"),
            ],
        },
        {
            "claim": "Code, data, and release reproducibility",
            "status": "complete" if release_published else "partial",
            "response_point": "Local release integrity, metadata, environment, statements, and handoff are complete; Zenodo must be published or a public GitHub repository must be available before submission.",
            "key_values": {
                "bundle_manifest": str(release_dir / "release_bundle_manifest.json"),
                "doi_status": _read_json(release_dir / "release_verification.json").get("doi_status", ""),
                "release_status": zenodo_payload.get("release_status", ""),
            },
            "evidence": [
                _evidence(release_dir / "release_bundle_manifest.json", "release manifest"),
                _evidence(release_dir / "release_verification.json", "release verification"),
                _evidence(release_dir / "zenodo_deposition_result.json", "Zenodo deposition result"),
                _evidence(release_dir / "final_submission_handoff.json", "final handoff"),
                _evidence(completion_audit_dir / "goal_completion_audit.json", "requirement completion audit"),
            ],
        },
    ]


def _reviewer_concerns(
    run_dir: Path,
    tables_dir: Path,
    figure_legends_dir: Path,
    methods_dir: Path,
    statements_dir: Path,
    release_dir: Path,
    external_dir: Path,
    external_matched_gt_dir: Path,
    minimal_retune_dir: Path,
    external_pathology_dir: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "concern": "The method is just a CNN/attention/gate module stack.",
            "preemptive_response": "Position the method around the benchmark, wavelet morphology, prototype-agent reference prior, calibrated reliability gate, boundary-aware objective, and biologically validated niches; cite ablations for necessity.",
            "evidence": [
                _evidence(methods_dir / "manuscript_methods.json", "model Methods"),
                _evidence(run_dir / "ablations250" / "ablation_summary.csv", "ablation panel"),
                _evidence(tables_dir / "table_4_ablation_delta.csv", "ablation manuscript table"),
            ],
        },
        {
            "concern": "Baseline comparison may be weak or unfair.",
            "preemptive_response": "Use the formal comparison table, shared supervised spots, bootstrap mean/std, paired tests, runtime/memory, and per-method manifests.",
            "evidence": [
                _evidence(tables_dir / "table_3_baseline_comparison.csv", "baseline manuscript table"),
                _evidence(run_dir / "baseline_comparison" / "baseline_comparison.csv", "baseline comparison"),
                _evidence(run_dir / "baseline_environment_audit.json", "baseline environment audit"),
            ],
        },
        {
            "concern": "The reliability gate may be ordinary attention.",
            "preemptive_response": "Use uncertainty-error correlation, calibration-bin trend, risk-coverage behavior, modality maps, and raw-gate-without-uncertainty ablation.",
            "evidence": [
                _evidence(figure_legends_dir / "figure_legends.json", "Figure 4 legend"),
                _evidence(run_dir / "nature_analysis" / "reliability_summary.json", "reliability summary"),
                _evidence(run_dir / "ablations250" / "ablation_summary.csv", "raw gate ablation"),
            ],
        },
        {
            "concern": "The primary validation split has no supervised Xenium ground truth.",
            "preemptive_response": "State that the original validation split is an unsupervised spatial holdout, keep the pre-specified spatial split for primary reporting, and cite GT-stratified supplementary splits where validation/test each contain supervised spots.",
            "evidence": [
                _evidence(tables_dir / "table_8_split_sensitivity.csv", "split sensitivity manuscript table"),
                _evidence(run_dir / "split_sensitivity" / "original_split_gt_summary.csv", "original split GT inventory"),
                _evidence(run_dir / "split_sensitivity" / "split_sensitivity_aggregate.csv", "GT-stratified sensitivity aggregate"),
                _evidence(run_dir / "split_sensitivity" / "split_sensitivity.md", "split sensitivity note"),
            ],
        },
        {
            "concern": "Xenium-to-Visium labels may depend on an arbitrary aggregation radius or low-confidence cell counts.",
            "preemptive_response": "Report radius/cell-count confidence sensitivity across radii 45/55/65/75 and minimum cell-count thresholds 1/5/10/20/50; keep radius 55 as the primary pre-specified setting and show how coverage and metrics change.",
            "evidence": [
                _evidence(tables_dir / "table_10_benchmark_sensitivity.csv", "benchmark sensitivity manuscript table"),
                _evidence(run_dir / "benchmark_sensitivity" / "radius_cell_count_sensitivity.csv", "radius/cell-count metrics"),
                _evidence(run_dir / "benchmark_sensitivity" / "radius_coverage_summary.csv", "radius coverage summary"),
                _evidence(run_dir / "benchmark_sensitivity" / "radius_cell_count_sensitivity.md", "benchmark sensitivity note"),
            ],
        },
        {
            "concern": "The mean image gate is low, so morphology may not contribute.",
            "preemptive_response": "Explain that the primary average gate is dominated by expression-rich spots, then cite the image-gate-enhanced control, matched no-image control, and texture-stratified paired error showing morphology helps most in high-texture/boundary regions.",
            "evidence": [
                _evidence(tables_dir / "table_9_imagegate_supplement.csv", "image-gate supplement manuscript table"),
                _evidence(run_dir.parent / f"{run_dir.name}_imagegate" / "image_contribution" / "image_contribution_summary.json", "paired image contribution summary"),
                _evidence(run_dir.parent / f"{run_dir.name}_imagegate" / "image_contribution" / "image_contribution_texture_groups.csv", "texture-stratified contribution"),
                _evidence(run_dir.parent / f"{run_dir.name}_imagegate_noimage" / "metrics.csv", "matched no-image control metrics"),
            ],
        },
        {
            "concern": "Spatial smoothness may over-smooth tumor-stroma or immune boundaries.",
            "preemptive_response": "Report boundary sharpness, typed boundaries, marker validation, H&E overlays, no-boundary-loss and normal-smoothness-only ablations.",
            "evidence": [
                _evidence(run_dir / "nature_analysis" / "boundary_summary.json", "boundary summary"),
                _evidence(run_dir / "nature_analysis" / "boundary_marker_validation.csv", "boundary marker validation"),
                _evidence(tables_dir / "table_5_reliability_boundary_niche.csv", "boundary manuscript table"),
            ],
        },
        {
            "concern": "Biological claims may be descriptive rather than validated.",
            "preemptive_response": "Tie niche names to cell composition, marker enrichment, H&E/pathology correspondence, Xenium neighborhood validation, gate reliability, and agent attention.",
            "evidence": [
                _evidence(run_dir / "nature_analysis" / "niche_biological_summary.csv", "niche biological summary"),
                _evidence(run_dir / "nature_analysis" / "niche_xenium_neighborhood_summary.csv", "Xenium neighborhood validation"),
                _evidence(external_pathology_dir / "pathology_niche_summary.csv", "external pathology niche summary"),
            ],
        },
        {
            "concern": "The method may work only on one sample.",
            "preemptive_response": "Use no-retuning external predictions, matched-GT transfer, minimal-retuning multi-sample summary, pathology-class validation, and the Rep1 budget curve; report direct Rep1 no-retuning as domain shift rather than hiding that RCTD is stronger there.",
            "evidence": [
                _evidence(external_dir / "external_no_retuning_summary.csv", "no-retuning external summary"),
                _evidence(external_matched_gt_dir / "external_matched_gt_summary.csv", "matched GT external summary"),
                _evidence(minimal_retune_dir / "matched_multisample_baseline_summary.csv", "minimal-retuning multi-sample summary"),
                _evidence(tables_dir / "table_11_rep1_retune_budget_curve.csv", "Rep1 retune budget table"),
            ],
        },
        {
            "concern": "Rep1 no-retuning is not ranked first.",
            "preemptive_response": "Frame this honestly as a cross-sample domain-shift result: direct transfer is reported, and a minimal-retuning budget curve quantifies that 25 adaptation steps are enough to beat the best Rep1 baseline on the held-out test split.",
            "evidence": [
                _evidence(tables_dir / "table_11_rep1_retune_budget_curve.csv", "Rep1 no-retuning/minimal-retuning table"),
                _evidence(external_matched_gt_dir / "xenium_rep1_pseudospots_radius55_common297" / "rep1_retune_budget_curve" / "rep1_minimal_retune_budget_curve.csv", "Rep1 budget curve"),
                _evidence(external_matched_gt_dir / "xenium_rep1_pseudospots_radius55_common297" / "wavestgate_minimal_retune" / "test_formal_comparison" / "baseline_comparison.csv", "minimal-retune formal baseline comparison"),
            ],
        },
        {
            "concern": "Robustness to real data perturbations is unclear.",
            "preemptive_response": "Point to patch-size, gene-panel, dropout, reference mismatch, missing-celltype, prototype perturbation, H&E perturbation, low-quality spot, and split tests.",
            "evidence": [
                _evidence(tables_dir / "table_7_robustness_summary.csv", "robustness manuscript table"),
                _evidence(tables_dir / "table_8_split_sensitivity.csv", "split sensitivity manuscript table"),
                _evidence(tables_dir / "table_10_benchmark_sensitivity.csv", "benchmark sensitivity manuscript table"),
                _evidence(run_dir / "robustness" / "robustness_summary.csv", "robustness summary"),
                _evidence(run_dir / "patch_size_robustness" / "patch_size_summary.csv", "patch-size robustness"),
            ],
        },
        {
            "concern": "Data/code availability and reproducibility are incomplete.",
            "preemptive_response": "Use release verification, environment report, availability statements, final handoff, and Zenodo deposition helper; explicitly state that a draft/reserved DOI is not public and must be published or backed by a public GitHub repository before submission.",
            "evidence": [
                _evidence(statements_dir / "manuscript_availability_statements.json", "availability statements"),
                _evidence(release_dir / "release_verification.json", "release verification"),
                _evidence(release_dir / "zenodo_deposition_result.json", "Zenodo deposition result"),
                _evidence(release_dir / "final_submission_handoff.json", "final handoff"),
            ],
        },
    ]


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# WaveST-Gate Reviewer Preflight Dossier",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        "",
        f"Overall readiness: `{payload['readiness_overall_status']}`",
        f"Release status: `{payload['release_status']}`",
        f"Project DOI status: `{payload['project_doi_status']}`",
        "",
        "## Stage Status",
        "",
    ]
    for stage, summary in payload["stage_summary"].items():
        lines.append(f"- {stage}: `{summary.get('status', '')}` ({summary.get('num_complete', 0)} complete, {summary.get('num_partial', 0)} partial, {summary.get('num_missing', 0)} missing)")
    lines.extend(["", "## Headline Claims", ""])
    for claim in payload["headline_claims"]:
        lines.extend([f"### {claim['claim']}", "", f"Status: `{claim['status']}`", "", claim["response_point"], ""])
        if claim["key_values"]:
            lines.append("Key values:")
            for key, value in claim["key_values"].items():
                lines.append(f"- `{key}`: `{value}`")
            lines.append("")
        lines.append("Evidence:")
        for evidence in claim["evidence"]:
            exists = "yes" if evidence["exists"] else "no"
            lines.append(f"- {evidence['role']}: `{evidence['path']}` (exists: {exists})")
        lines.append("")
    lines.extend(["## Reviewer Concerns", ""])
    for item in payload["reviewer_concerns"]:
        lines.extend([f"### {item['concern']}", "", item["preemptive_response"], "", "Evidence:"])
        for evidence in item["evidence"]:
            exists = "yes" if evidence["exists"] else "no"
            lines.append(f"- {evidence['role']}: `{evidence['path']}` (exists: {exists})")
        lines.append("")
    lines.extend(["## Remaining External Action", "", payload["remaining_external_action"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_reviewer_preflight(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    readiness_report_path: str | Path = "results/nature_submission_readiness/readiness_report.json",
    release_dir: str | Path = "results/nature_release",
    tables_dir: str | Path = "results/nature_manuscript_tables",
    run_dir: str | Path = "results/nature_main/cytassist_rep2_radius55",
    benchmark_dir: str | Path = "data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55",
    figure_legends_dir: str | Path = "results/nature_manuscript_figure_legends",
    methods_dir: str | Path = "results/nature_manuscript_methods",
    statements_dir: str | Path = "results/nature_manuscript_statements",
    docs_dir: str | Path = "docs",
    external_dir: str | Path = "results/nature_external_no_retuning",
    external_matched_gt_dir: str | Path = "results/nature_external_matched_gt",
    minimal_retune_dir: str | Path = "results/nature_matched_multisample_baselines_minimal_retune",
    external_pathology_dir: str | Path = "results/nature_external_pathology_validation",
    benchmark_datasheet_dir: str | Path = "results/nature_benchmark_datasheet",
    completion_audit_dir: str | Path = "results/nature_completion_audit",
) -> dict[str, Any]:
    """Generate a reviewer-preflight dossier from current evidence."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    readiness_report_path = Path(readiness_report_path)
    release_dir = Path(release_dir)
    tables_dir = Path(tables_dir)
    run_dir = Path(run_dir)
    benchmark_dir = Path(benchmark_dir)
    figure_legends_dir = Path(figure_legends_dir)
    methods_dir = Path(methods_dir)
    statements_dir = Path(statements_dir)
    docs_dir = Path(docs_dir)
    external_dir = Path(external_dir)
    external_matched_gt_dir = Path(external_matched_gt_dir)
    minimal_retune_dir = Path(minimal_retune_dir)
    external_pathology_dir = Path(external_pathology_dir)
    benchmark_datasheet_dir = Path(benchmark_datasheet_dir)
    completion_audit_dir = Path(completion_audit_dir)

    readiness = _read_json(readiness_report_path)
    release_verification = _read_json(release_dir / "release_verification.json")
    zenodo = _read_json(release_dir / "zenodo_deposition_result.json")
    claims = _headline_claims(readiness, tables_dir, run_dir, benchmark_dir, release_dir, docs_dir, external_dir, external_matched_gt_dir, minimal_retune_dir, benchmark_datasheet_dir, completion_audit_dir)
    concerns = _reviewer_concerns(run_dir, tables_dir, figure_legends_dir, methods_dir, statements_dir, release_dir, external_dir, external_matched_gt_dir, minimal_retune_dir, external_pathology_dir)
    all_evidence = [evidence for claim in claims for evidence in claim["evidence"]] + [evidence for concern in concerns for evidence in concern["evidence"]]
    missing_evidence = [item["path"] for item in all_evidence if not item["exists"]]
    partial_claims = [claim["claim"] for claim in claims if claim.get("status") == "partial"]
    status = "complete_except_project_doi" if not missing_evidence and partial_claims == ["Code, data, and release reproducibility"] else ("complete" if not missing_evidence and not partial_claims else "partial")

    json_path = output_dir / "reviewer_preflight.json"
    markdown_path = output_dir / "reviewer_preflight.md"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "readiness_overall_status": readiness.get("overall_status", ""),
        "stage_summary": readiness.get("stage_summary", {}),
        "stage_records": _stage_records(readiness),
        "release_status": zenodo.get("release_status", ""),
        "project_doi_status": release_verification.get("doi_status", ""),
        "release_verification": {
            "overall_status": release_verification.get("overall_status", ""),
            "bundle_integrity_status": release_verification.get("bundle_integrity_status", ""),
            "doi_status": release_verification.get("doi_status", ""),
            "num_failures": release_verification.get("num_failures", ""),
        },
        "headline_claims": claims,
        "reviewer_concerns": concerns,
        "missing_evidence": sorted(set(missing_evidence)),
        "remaining_external_action": "A project Zenodo DOI requires ZENODO_ACCESS_TOKEN and token-backed deposition; all local release, verification, methods, figures, legends, and readiness evidence are prepared.",
        "outputs": {"json": str(json_path), "markdown": str(markdown_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, markdown_path)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build WaveST-Gate reviewer preflight dossier.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--readiness-report", type=Path, default=Path("results/nature_submission_readiness/readiness_report.json"))
    parser.add_argument("--release-dir", type=Path, default=Path("results/nature_release"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/nature_manuscript_tables"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/nature_main/cytassist_rep2_radius55"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55"))
    parser.add_argument("--figure-legends-dir", type=Path, default=Path("results/nature_manuscript_figure_legends"))
    parser.add_argument("--methods-dir", type=Path, default=Path("results/nature_manuscript_methods"))
    parser.add_argument("--statements-dir", type=Path, default=Path("results/nature_manuscript_statements"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--external-dir", type=Path, default=Path("results/nature_external_no_retuning"))
    parser.add_argument("--external-matched-gt-dir", type=Path, default=Path("results/nature_external_matched_gt"))
    parser.add_argument("--minimal-retune-dir", type=Path, default=Path("results/nature_matched_multisample_baselines_minimal_retune"))
    parser.add_argument("--external-pathology-dir", type=Path, default=Path("results/nature_external_pathology_validation"))
    parser.add_argument("--benchmark-datasheet-dir", type=Path, default=Path("results/nature_benchmark_datasheet"))
    parser.add_argument("--completion-audit-dir", type=Path, default=Path("results/nature_completion_audit"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = build_reviewer_preflight(
        output_dir=args.output_dir,
        readiness_report_path=args.readiness_report,
        release_dir=args.release_dir,
        tables_dir=args.tables_dir,
        run_dir=args.run_dir,
        benchmark_dir=args.benchmark_dir,
        figure_legends_dir=args.figure_legends_dir,
        methods_dir=args.methods_dir,
        statements_dir=args.statements_dir,
        docs_dir=args.docs_dir,
        external_dir=args.external_dir,
        external_matched_gt_dir=args.external_matched_gt_dir,
        minimal_retune_dir=args.minimal_retune_dir,
        external_pathology_dir=args.external_pathology_dir,
        benchmark_datasheet_dir=args.benchmark_datasheet_dir,
        completion_audit_dir=args.completion_audit_dir,
    )
    print(json.dumps(payload["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
