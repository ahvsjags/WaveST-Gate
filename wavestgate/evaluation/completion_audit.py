"""Requirement-by-requirement completion audit for WaveST-Gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_BENCHMARK_DIR = Path("data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55")
DEFAULT_RUN_DIR = Path("results/nature_main/cytassist_rep2_radius55")
DEFAULT_EXTERNAL_DIR = Path("results/nature_external_no_retuning")
DEFAULT_EXTERNAL_MATCHED_GT_DIR = Path("results/nature_external_matched_gt")
DEFAULT_EXTERNAL_PATHOLOGY_DIR = Path("results/nature_external_pathology_validation")
DEFAULT_MULTISAMPLE_DIR = Path("results/nature_matched_multisample_baselines")
DEFAULT_MINIMAL_MULTISAMPLE_DIR = Path("results/nature_matched_multisample_baselines_minimal_retune")
DEFAULT_DATASHEET_DIR = Path("results/nature_benchmark_datasheet")
DEFAULT_RELEASE_DIR = Path("results/nature_release")
DEFAULT_TABLES_DIR = Path("results/nature_manuscript_tables")
DEFAULT_OUTPUT_DIR = Path("results/nature_completion_audit")


@dataclass
class AuditRequirement:
    stage: str
    requirement: str
    status: str
    detail: str
    evidence_paths: list[str]
    passed_checks: int
    failed_checks: list[str]


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


def _exists(path: str | Path) -> bool:
    return Path(path).exists()


def _all_files(paths: list[str | Path]) -> bool:
    return all(_exists(path) for path in paths)


def _status_from_checks(checks: list[tuple[str, bool]], evidence_paths: list[str | Path], allow_partial: bool = False) -> tuple[str, list[str], int]:
    failed = [name for name, passed in checks if not passed]
    passed = sum(passed for _, passed in checks)
    any_evidence = any(Path(path).exists() for path in evidence_paths)
    if not failed:
        return "complete", failed, passed
    if allow_partial or any_evidence:
        return "partial", failed, passed
    return "missing", failed, passed


def _record(
    stage: str,
    requirement: str,
    detail: str,
    evidence_paths: list[str | Path],
    checks: list[tuple[str, bool]],
    allow_partial: bool = False,
) -> AuditRequirement:
    status, failed, passed = _status_from_checks(checks, evidence_paths, allow_partial=allow_partial)
    return AuditRequirement(
        stage=stage,
        requirement=requirement,
        status=status,
        detail=detail,
        evidence_paths=[str(path) for path in evidence_paths],
        passed_checks=passed,
        failed_checks=failed,
    )


def _audit_stage_1(benchmark_dir: Path, docs_dir: Path, datasheet_dir: Path, release_dir: Path) -> list[AuditRequirement]:
    manifest_path = benchmark_dir / "xenium_visium_benchmark_manifest.json"
    counts_path = benchmark_dir / "xenium_cell_counts.csv"
    proportions_path = benchmark_dir / "xenium_cell_proportions.csv"
    qc_path = benchmark_dir / "spot_ground_truth_qc.csv"
    splits_path = benchmark_dir / "spot_splits.csv"
    protocol_path = docs_dir / "xenium_to_visium_benchmark_protocol.md"
    datasheet_path = datasheet_dir / "benchmark_datasheet.json"
    manifest = _read_json(manifest_path)
    qc = _read_csv(qc_path)
    datasheet = _read_json(datasheet_path)
    release = _read_json(release_dir / "zenodo_deposition_result.json")
    release_published = release.get("release_status") == "zenodo_published"
    required_qc = {"xenium_cell_count", "ground_truth_entropy", "dominant_cell_type", "has_xenium_ground_truth"}
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    records = [
        _record(
            "1. Xenium-to-Visium benchmark",
            "spot-level Xenium ground truth protocol",
            "Counts, proportions, QC, fixed splits, radius, coordinate/cell-type mapping, manifest, and protocol are present and content-checked.",
            [counts_path, proportions_path, qc_path, splits_path, manifest_path, protocol_path],
            [
                ("counts_table_exists", _exists(counts_path)),
                ("proportions_table_exists", _exists(proportions_path)),
                ("qc_table_exists", _exists(qc_path)),
                ("split_table_exists", _exists(splits_path)),
                ("protocol_exists", _exists(protocol_path)),
                ("manifest_has_spot_radius", bool(manifest.get("spot_radius"))),
                ("manifest_has_coordinate_and_celltype_columns", bool(manifest.get("columns"))),
                ("manifest_has_artifact_paths", bool(artifacts)),
                ("qc_has_coverage_entropy_dominant_gt_flag", required_qc.issubset(set(qc.columns))),
            ],
        ),
        _record(
            "1. Xenium-to-Visium benchmark",
            "benchmark datasheet and data dictionary",
            "Datasheet records artifact inventory, QC/split/cell-type summaries, data dictionary, and integrity checks.",
            [datasheet_dir / "benchmark_datasheet.json", datasheet_dir / "benchmark_datasheet.md"],
            [
                ("datasheet_json_exists", _exists(datasheet_dir / "benchmark_datasheet.json")),
                ("datasheet_markdown_exists", _exists(datasheet_dir / "benchmark_datasheet.md")),
                ("datasheet_status_complete", datasheet.get("status") == "complete"),
                ("no_datasheet_integrity_failures", not datasheet.get("failed_integrity_checks")),
                ("datasheet_has_cell_types", bool(datasheet.get("cell_type_summary"))),
                ("datasheet_has_column_dictionary", bool(datasheet.get("column_dictionary"))),
            ],
        ),
        _record(
            "1. Xenium-to-Visium benchmark",
            "project Zenodo DOI and deposition",
            "A real project DOI requires token-backed Zenodo deposition; dry-run evidence is not counted as complete.",
            [
                release_dir / "release_bundle_manifest.json",
                release_dir / "zenodo_metadata.json",
                release_dir / "zenodo_deposition_result.json",
                release_dir / "release_verification.json",
            ],
            [
                ("release_bundle_manifest_exists", _exists(release_dir / "release_bundle_manifest.json")),
                ("zenodo_metadata_exists", _exists(release_dir / "zenodo_metadata.json")),
                ("deposition_result_exists", _exists(release_dir / "zenodo_deposition_result.json")),
                ("project_doi_present", bool(release.get("doi"))),
                ("zenodo_deposition_id_present", bool(release.get("zenodo_deposition_id"))),
                ("zenodo_release_published", release_published),
            ],
            allow_partial=True,
        ),
    ]
    return records


def _audit_stage_2(run_dir: Path) -> list[AuditRequirement]:
    nature = run_dir / "nature_analysis"
    files = [
        run_dir / "checkpoint.pt",
        run_dir / "predicted_proportions.csv",
        run_dir / "reconstructed_expression.csv",
        run_dir / "gate_weights.csv",
        run_dir / "modality_reliability.csv",
        run_dir / "spot_uncertainty.csv",
        run_dir / "agent_attention.csv",
        run_dir / "training_history.csv",
        run_dir / "metrics.csv",
        nature / "proportion_maps" / "proportion_map_manifest.json",
        nature / "proportion_maps" / "predicted_top_celltypes_panel.png",
        nature / "proportion_maps" / "predicted_tumor_immune_stromal_group_panel.png",
        nature / "image_gate_map.png",
        nature / "expression_gate_map.png",
        nature / "reference_gate_map.png",
        nature / "spot_uncertainty_map.png",
        nature / "niche_map.png",
    ]
    metrics = _read_csv(run_dir / "metrics.csv")
    metric_cols = {"jsd", "spotwise_cosine", "mean_celltype_pearson", "num_supervised_spots"}
    return [
        _record(
            "2. Main model training",
            "real multimodal model outputs and maps",
            "Checkpoint, proportions, expression reconstruction, gate/reliability/uncertainty/agent/niche outputs, curves, and supervised metrics are available.",
            files,
            [
                ("all_required_model_files_exist", _all_files(files)),
                ("metrics_have_required_columns", metric_cols.issubset(set(metrics.columns))),
                ("metrics_have_rows", not metrics.empty),
            ],
        )
    ]


def _audit_stage_3(run_dir: Path, multisample_dir: Path) -> list[AuditRequirement]:
    comparison_path = run_dir / "baseline_comparison" / "baseline_comparison.csv"
    frame = _read_csv(comparison_path)
    method_text = " | ".join(frame.get("method", pd.Series(dtype=str)).astype(str).tolist()).lower()
    required_methods = ["cell2location", "rctd", "card", "tangram", "spatialdwls", "spatialdwls/seurat", "bayesprism"]
    statistics_files = [
        run_dir / "baseline_comparison" / "baseline_split_bootstrap_metrics.csv",
        run_dir / "baseline_comparison" / "baseline_split_bootstrap_summary.csv",
        run_dir / "baseline_comparison" / "baseline_bootstrap_paired_improvement.csv",
        run_dir / "baseline_comparison" / "baseline_statistics_manifest.json",
        multisample_dir / "matched_multisample_baseline_summary.csv",
    ]
    columns = set(frame.columns)
    return [
        _record(
            "3. Strong baseline comparison",
            "formal baselines, fairness, statistics, runtime, and memory",
            "Strong baselines are compared with shared supervised spots/genes/reference where applicable, paired statistics, bootstrap mean/std, runtime, and memory.",
            [comparison_path, run_dir / "baseline_environment_audit.json", *statistics_files],
            [
                ("baseline_comparison_exists", _exists(comparison_path)),
                ("required_baseline_methods_present", all(method in method_text for method in required_methods)),
                ("runtime_memory_significance_columns_present", {"runtime_seconds", "peak_cuda_memory_mb", "paired_permutation_p"}.issubset(columns)),
                ("bootstrap_and_multisample_statistics_exist", _all_files(statistics_files)),
                ("baseline_environment_audit_exists", _exists(run_dir / "baseline_environment_audit.json")),
            ],
        )
    ]


def _audit_stage_4(run_dir: Path) -> list[AuditRequirement]:
    path = run_dir / "ablations250" / "ablation_summary.csv"
    frame = _read_csv(path)
    required = {
        "full",
        "no_wavelet_cnn_replacement",
        "no_image_branch",
        "no_celltype_agents",
        "no_gate_mean_fusion",
        "raw_gate_without_uncertainty",
        "no_boundary_loss",
        "normal_smoothness_only",
        "no_local_refinement",
        "expression_only",
        "image_only",
        "reference_only",
    }
    observed = set(frame.get("ablation", pd.Series(dtype=str)).astype(str).tolist())
    return [
        _record(
            "4. Ablation study",
            "all requested module and modality ablations",
            "The ablation panel covers the full model, wavelet/CNN replacement, agents, gate, uncertainty, boundary loss, local refinement, and modality-only variants.",
            [path],
            [("ablation_summary_exists", _exists(path)), ("all_required_ablations_present", required.issubset(observed))],
        )
    ]


def _audit_stage_5(run_dir: Path) -> list[AuditRequirement]:
    nature = run_dir / "nature_analysis"
    summary = _read_json(nature / "reliability_summary.json")
    files = [
        nature / "reliability_summary.json",
        nature / "reliability_spot_errors.csv",
        nature / "risk_coverage_curve.csv",
        nature / "risk_coverage_curve.png",
        nature / "uncertainty_calibration_bins.csv",
        nature / "uncertainty_calibration.png",
        nature / "failure_case_candidates.csv",
        nature / "image_gate_map.png",
        nature / "expression_gate_map.png",
        nature / "reference_gate_map.png",
    ]
    return [
        _record(
            "5. Reliability and calibration",
            "calibrated reliability gate evidence",
            "Uncertainty-error correlation, calibration bins, risk coverage, gate maps, and failure cases are present.",
            files,
            [
                ("all_reliability_files_exist", _all_files(files)),
                ("uncertainty_error_pearson_present", "uncertainty_error_pearson" in summary),
                ("risk_gap_present", "risk_gap" in summary),
                ("calibration_bin_pearson_present", "calibration_bin_pearson" in summary),
            ],
        )
    ]


def _audit_stage_6(run_dir: Path, external_pathology_dir: Path) -> list[AuditRequirement]:
    nature = run_dir / "nature_analysis"
    files = [
        nature / "boundary_summary.json",
        nature / "boundary_edge_jumps.csv",
        nature / "boundary_type_summary.csv",
        nature / "boundary_marker_validation.csv",
        nature / "boundary_sharpness_map.png",
        nature / "boundary_he_overlay.png",
        nature / "boundary_he_pathology_proxy.csv",
        external_pathology_dir / "pathology_class_summary.csv",
    ]
    summary = _read_json(nature / "boundary_summary.json")
    type_summary = _read_csv(nature / "boundary_type_summary.csv")
    comparison_present = "comparison_mean_boundary_jump" in summary or {
        "mean_comparison_l1_jump",
        "boundary_preservation_delta_vs_comparison",
    }.issubset(set(type_summary.columns))
    return [
        _record(
            "6. Boundary preservation",
            "morphology-aware boundary preservation evidence",
            "Tumor-stroma, ductal, immune-edge boundaries, H&E overlay, marker validation, and pathology metadata validation are present.",
            files,
            [
                ("all_boundary_files_exist", _all_files(files)),
                ("boundary_to_interior_jump_ratio_present", "boundary_to_interior_jump_ratio" in summary),
                ("ordinary_smoothness_or_no_boundary_comparison_present", comparison_present),
            ],
        )
    ]


def _audit_stage_7(run_dir: Path, external_pathology_dir: Path) -> list[AuditRequirement]:
    nature = run_dir / "nature_analysis"
    files = [
        nature / "niche_assignments.csv",
        nature / "niche_composition.csv",
        nature / "niche_marker_enrichment.csv",
        nature / "niche_biological_summary.csv",
        nature / "niche_xenium_neighborhood_validation.csv",
        nature / "niche_xenium_neighborhood_summary.csv",
        nature / "gate_reliability_by_niche.csv",
        nature / "agent_attention_by_niche.csv",
        nature / "niche_map.png",
        external_pathology_dir / "pathology_niche_summary.csv",
        external_pathology_dir / "pathology_niche_by_class.csv",
    ]
    summary = _read_json(nature / "niche_summary.json")
    return [
        _record(
            "7. Biological niche interpretation",
            "tumor-immune-stromal niche interpretation and validation",
            "Niche composition, marker enrichment, Xenium neighborhood validation, pathology correspondence, gate reliability, and agent attention by niche are present.",
            files,
            [
                ("all_niche_files_exist", _all_files(files)),
                ("niche_summary_has_num_niches", "num_niches" in summary),
            ],
        )
    ]


def _audit_stage_8(external_dir: Path, external_matched_gt_dir: Path, minimal_multisample_dir: Path) -> list[AuditRequirement]:
    summary_path = external_dir / "external_no_retuning_summary.csv"
    external = _read_csv(summary_path)
    datasets = external.get("dataset", pd.Series(dtype=str)).astype(str).tolist()
    per_dataset = ["predicted_proportions.csv", "gate_weights.csv", "spot_uncertainty.csv", "agent_attention.csv", "aligned_prediction_metrics.csv", "aligned_prediction_manifest.json"]
    missing_per_dataset = [f"{dataset}/{name}" for dataset in datasets for name in per_dataset if not (external_dir / dataset / name).exists()]
    files = [
        summary_path,
        external_matched_gt_dir / "external_matched_gt_summary.csv",
        external_matched_gt_dir / "external_matched_gt_manifest.json",
        external_matched_gt_dir / "xenium_rep1_pseudospots_radius55_common297" / "matched_gt_metrics.csv",
        external_matched_gt_dir / "xenium_rep1_pseudospots_radius55_common297" / "wavestgate_minimal_retune" / "test_metrics.csv",
        external_matched_gt_dir / "xenium_rep1_pseudospots_radius55_common297" / "wavestgate_minimal_retune" / "test_formal_comparison" / "baseline_comparison.csv",
        external_matched_gt_dir / "xenium_rep1_pseudospots_radius55_common297" / "rep1_retune_budget_curve" / "rep1_minimal_retune_budget_curve.csv",
        external_matched_gt_dir / "xenium_rep1_pseudospots_radius55_common297" / "rep1_retune_budget_curve" / "rep1_minimal_retune_budget_curve_manifest.json",
        minimal_multisample_dir / "matched_multisample_baseline_summary.csv",
    ]
    return [
        _record(
            "8. External generalization",
            "cross-sample and cross-platform external generalization",
            "External no-retuning, Rep1 matched-GT transfer, honest no-retuning/minimal-retuning budget curve, and minimal-retuning multi-sample summaries are present.",
            files,
            [
                ("external_summary_exists", _exists(summary_path)),
                ("at_least_ten_external_datasets", len(datasets) >= 10),
                ("per_dataset_outputs_exist", not missing_per_dataset),
                ("matched_gt_and_minimal_retune_files_exist", _all_files(files[1:])),
            ],
        )
    ]


def _audit_stage_9(run_dir: Path) -> list[AuditRequirement]:
    robustness_path = run_dir / "robustness" / "robustness_summary.csv"
    patch_path = run_dir / "patch_size_robustness" / "patch_size_summary.csv"
    split_sensitivity_files = [
        run_dir / "split_sensitivity" / "original_split_gt_summary.csv",
        run_dir / "split_sensitivity" / "split_sensitivity_aggregate.csv",
        run_dir / "split_sensitivity" / "split_sensitivity_manifest.json",
    ]
    benchmark_sensitivity_files = [
        run_dir / "benchmark_sensitivity" / "radius_cell_count_sensitivity.csv",
        run_dir / "benchmark_sensitivity" / "radius_coverage_summary.csv",
        run_dir / "benchmark_sensitivity" / "radius_cell_count_sensitivity_manifest.json",
    ]
    robust = _read_csv(robustness_path)
    patch = _read_csv(patch_path)
    pairs = set(zip(robust.get("scenario", pd.Series(dtype=str)).astype(str), robust.get("level", pd.Series(dtype=str)).astype(str)))
    scenarios = set(robust.get("scenario", pd.Series(dtype=str)).astype(str))
    patch_sizes = set(patch.get("patch_size", pd.Series(dtype=str)).astype(str))
    subgroup_levels = set(robust.loc[robust.get("scenario", pd.Series(dtype=str)).astype(str) == "subgroup", "level"].astype(str).tolist()) if not robust.empty else set()
    return [
        _record(
            "9. Robustness",
            "realistic perturbation and sensitivity stress tests",
            "Patch size, gene panels/dropout, reference mismatch/removal, prototype perturbation, H&E perturbation, low-quality spots, split variation, GT-stratified split sensitivity, and radius/cell-count benchmark sensitivity are present.",
            [robustness_path, patch_path, run_dir / "robustness" / "robustness_manifest.json", *split_sensitivity_files, *benchmark_sensitivity_files],
            [
                ("robustness_summary_exists", _exists(robustness_path)),
                ("patch_size_summary_exists", _exists(patch_path)),
                ("split_sensitivity_files_exist", _all_files(split_sensitivity_files)),
                ("benchmark_sensitivity_files_exist", _all_files(benchmark_sensitivity_files)),
                ("patch_sizes_32_64_128_256_present", {"32", "64", "128", "256"}.issubset(patch_sizes)),
                ("gene_dropout_levels_present", {("gene_dropout", "0.1"), ("gene_dropout", "0.3"), ("gene_dropout", "0.5")}.issubset(pairs)),
                ("gene_panel_levels_present", {("gene_panel", "top200_variance"), ("gene_panel", "top100_variance"), ("gene_panel", "marker_only")}.issubset(pairs)),
                ("reference_missing_celltype_rows_present", len(robust[robust.get("scenario", pd.Series(dtype=str)).astype(str) == "reference_missing_celltype"]) >= 10),
                ("prototype_perturbation_rows_present", len(robust[robust.get("scenario", pd.Series(dtype=str)).astype(str) == "prototype_perturbation"]) >= 4),
                ("he_perturbation_present", "he_perturbation" in scenarios),
                ("low_quality_subgroups_present", {"low_expression_spots", "low_cell_count_spots"}.issubset(subgroup_levels)),
                ("split_variation_present", "split" in scenarios),
            ],
        )
    ]


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# WaveST-Gate Goal Completion Audit",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        "",
        f"Overall status: `{payload['overall_status']}`",
        f"Missing requirements: `{payload['num_missing']}`",
        f"Partial requirements: `{payload['num_partial']}`",
        "",
        "## Stage Summary",
        "",
        "| Stage | Status | Complete | Partial | Missing |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for stage, summary in payload["stage_summary"].items():
        lines.append(f"| {stage} | `{summary['status']}` | {summary['complete']} | {summary['partial']} | {summary['missing']} |")
    lines.extend(["", "## Requirements", ""])
    for record in payload["requirements"]:
        lines.append(f"### {record['stage']} - {record['requirement']}")
        lines.append("")
        lines.append(f"Status: `{record['status']}`")
        lines.append("")
        lines.append(record["detail"])
        lines.append("")
        lines.append(f"Passed checks: `{record['passed_checks']}`")
        if record["failed_checks"]:
            lines.append("Failed checks:")
            for check in record["failed_checks"]:
                lines.append(f"- `{check}`")
        lines.append("Evidence:")
        for evidence in record["evidence_paths"]:
            lines.append(f"- `{evidence}`")
        lines.append("")
    if payload["remaining_external_action"]:
        lines.extend(["## Remaining External Action", "", payload["remaining_external_action"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_completion_audit(
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    external_dir: str | Path = DEFAULT_EXTERNAL_DIR,
    external_matched_gt_dir: str | Path = DEFAULT_EXTERNAL_MATCHED_GT_DIR,
    external_pathology_dir: str | Path = DEFAULT_EXTERNAL_PATHOLOGY_DIR,
    multisample_dir: str | Path = DEFAULT_MULTISAMPLE_DIR,
    minimal_multisample_dir: str | Path = DEFAULT_MINIMAL_MULTISAMPLE_DIR,
    datasheet_dir: str | Path = DEFAULT_DATASHEET_DIR,
    release_dir: str | Path = DEFAULT_RELEASE_DIR,
    tables_dir: str | Path = DEFAULT_TABLES_DIR,
    docs_dir: str | Path = "docs",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build a machine-readable audit for the full user objective."""

    benchmark_dir = Path(benchmark_dir)
    run_dir = Path(run_dir)
    external_dir = Path(external_dir)
    external_matched_gt_dir = Path(external_matched_gt_dir)
    external_pathology_dir = Path(external_pathology_dir)
    multisample_dir = Path(multisample_dir)
    minimal_multisample_dir = Path(minimal_multisample_dir)
    datasheet_dir = Path(datasheet_dir)
    release_dir = Path(release_dir)
    docs_dir = Path(docs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requirements: list[AuditRequirement] = []
    requirements.extend(_audit_stage_1(benchmark_dir, docs_dir, datasheet_dir, release_dir))
    requirements.extend(_audit_stage_2(run_dir))
    requirements.extend(_audit_stage_3(run_dir, multisample_dir))
    requirements.extend(_audit_stage_4(run_dir))
    requirements.extend(_audit_stage_5(run_dir))
    requirements.extend(_audit_stage_6(run_dir, external_pathology_dir))
    requirements.extend(_audit_stage_7(run_dir, external_pathology_dir))
    requirements.extend(_audit_stage_8(external_dir, external_matched_gt_dir, minimal_multisample_dir))
    requirements.extend(_audit_stage_9(run_dir))

    stages = sorted({record.stage for record in requirements})
    stage_summary: dict[str, dict[str, Any]] = {}
    for stage in stages:
        records = [record for record in requirements if record.stage == stage]
        complete = sum(record.status == "complete" for record in records)
        partial = sum(record.status == "partial" for record in records)
        missing = sum(record.status == "missing" for record in records)
        status = "complete" if partial == 0 and missing == 0 else ("missing" if missing else "partial")
        stage_summary[stage] = {"status": status, "complete": complete, "partial": partial, "missing": missing}

    partial_requirements = [record for record in requirements if record.status == "partial"]
    missing_requirements = [record for record in requirements if record.status == "missing"]
    only_project_doi_pending = not missing_requirements and len(partial_requirements) == 1 and partial_requirements[0].requirement == "project Zenodo DOI and deposition"
    overall_status = "complete" if not partial_requirements and not missing_requirements else ("complete_except_project_doi" if only_project_doi_pending else "partial")
    remaining_external_action = (
        "A real project Zenodo DOI requires ZENODO_ACCESS_TOKEN and token-backed deposition; all other audited goal requirements are complete."
        if only_project_doi_pending
        else ""
    )
    json_path = output_dir / "goal_completion_audit.json"
    markdown_path = output_dir / "goal_completion_audit.md"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "num_requirements": len(requirements),
        "num_complete": sum(record.status == "complete" for record in requirements),
        "num_partial": len(partial_requirements),
        "num_missing": len(missing_requirements),
        "stage_summary": stage_summary,
        "requirements": [asdict(record) for record in requirements],
        "remaining_external_action": remaining_external_action,
        "outputs": {"json": str(json_path), "markdown": str(markdown_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, markdown_path)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a WaveST-Gate requirement-by-requirement completion audit.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    parser.add_argument("--external-matched-gt-dir", type=Path, default=DEFAULT_EXTERNAL_MATCHED_GT_DIR)
    parser.add_argument("--external-pathology-dir", type=Path, default=DEFAULT_EXTERNAL_PATHOLOGY_DIR)
    parser.add_argument("--multisample-dir", type=Path, default=DEFAULT_MULTISAMPLE_DIR)
    parser.add_argument("--minimal-multisample-dir", type=Path, default=DEFAULT_MINIMAL_MULTISAMPLE_DIR)
    parser.add_argument("--datasheet-dir", type=Path, default=DEFAULT_DATASHEET_DIR)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = build_completion_audit(
        benchmark_dir=args.benchmark_dir,
        run_dir=args.run_dir,
        external_dir=args.external_dir,
        external_matched_gt_dir=args.external_matched_gt_dir,
        external_pathology_dir=args.external_pathology_dir,
        multisample_dir=args.multisample_dir,
        minimal_multisample_dir=args.minimal_multisample_dir,
        datasheet_dir=args.datasheet_dir,
        release_dir=args.release_dir,
        tables_dir=args.tables_dir,
        docs_dir=args.docs_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
