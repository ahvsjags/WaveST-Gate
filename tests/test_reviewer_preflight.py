import json
from pathlib import Path

import pandas as pd

from wavestgate.evaluation.reviewer_preflight import build_reviewer_preflight


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("artifact", encoding="utf-8")


def test_build_reviewer_preflight_complete_except_project_doi(tmp_path: Path) -> None:
    readiness = {
        "overall_status": "partial",
        "stage_summary": {
            "8. External generalization": {"status": "complete"},
            "9. Robustness": {"status": "complete"},
        },
        "records": [
            {"stage": "1. Xenium-to-Visium benchmark", "requirement": "spot-level Xenium aggregation artifacts", "status": "complete"},
            {"stage": "1. Xenium-to-Visium benchmark", "requirement": "benchmark datasheet and data dictionary", "status": "complete"},
            {"stage": "2. Main model training", "requirement": "checkpoint, predictions, reliability maps, and training curve", "status": "complete"},
            {"stage": "3. Strong baseline comparison", "requirement": "formal baseline table with shared genes/reference/supervised spots", "status": "complete"},
            {"stage": "4. Ablation study", "requirement": "module and modality ablations", "status": "complete"},
            {"stage": "5. Reliability and calibration", "requirement": "uncertainty calibration and modality reliability evidence", "status": "complete"},
            {"stage": "6. Boundary preservation", "requirement": "typed boundary sharpness and marker validation", "status": "complete"},
            {"stage": "7. Biological niche interpretation", "requirement": "tumor-immune-stromal niche outputs", "status": "complete"},
        ],
    }
    readiness_path = tmp_path / "readiness" / "readiness_report.json"
    _write_json(readiness_path, readiness)

    benchmark_dir = tmp_path / "benchmark"
    _write_json(benchmark_dir / "xenium_visium_benchmark_manifest.json", {"num_spots": 10, "num_spots_with_ground_truth": 6, "num_cells": 100, "num_cell_types": 3})
    for name in ["xenium_cell_counts.csv", "xenium_cell_proportions.csv", "spot_ground_truth_qc.csv", "spot_splits.csv"]:
        _touch(benchmark_dir / name)
    docs_dir = tmp_path / "docs"
    _touch(docs_dir / "xenium_to_visium_benchmark_protocol.md")
    benchmark_datasheet_dir = tmp_path / "benchmark_datasheet"
    _touch(benchmark_datasheet_dir / "benchmark_datasheet.json")
    _touch(benchmark_datasheet_dir / "benchmark_datasheet.md")
    completion_audit_dir = tmp_path / "completion_audit"
    _touch(completion_audit_dir / "goal_completion_audit.json")
    _touch(completion_audit_dir / "goal_completion_audit.md")

    run_dir = tmp_path / "run"
    for path in [
        "checkpoint.pt",
        "predicted_proportions.csv",
        "metrics.csv",
        "nature_analysis/proportion_maps",
        "baseline_comparison/baseline_comparison.csv",
        "baseline_comparison/baseline_split_bootstrap_summary.csv",
        "baseline_comparison/baseline_bootstrap_paired_improvement.csv",
        "baseline_environment_audit.json",
        "ablations250/ablation_summary.csv",
        "nature_analysis/reliability_summary.json",
        "nature_analysis/risk_coverage_curve.csv",
        "nature_analysis/uncertainty_calibration_bins.csv",
        "nature_analysis/boundary_summary.json",
        "nature_analysis/boundary_marker_validation.csv",
        "nature_analysis/boundary_type_summary.csv",
        "nature_analysis/niche_summary.json",
        "nature_analysis/niche_biological_summary.csv",
        "nature_analysis/niche_xenium_neighborhood_summary.csv",
        "robustness/robustness_summary.csv",
        "patch_size_robustness/patch_size_summary.csv",
        "split_sensitivity/original_split_gt_summary.csv",
        "split_sensitivity/split_sensitivity_aggregate.csv",
        "split_sensitivity/split_sensitivity.md",
        "benchmark_sensitivity/radius_cell_count_sensitivity.csv",
        "benchmark_sensitivity/radius_coverage_summary.csv",
        "benchmark_sensitivity/radius_cell_count_sensitivity.md",
    ]:
        _touch(run_dir / path)

    tables_dir = tmp_path / "tables"
    _write_csv(tables_dir / "table_2_main_model_performance.csv", [{"metric": "JSD", "value": 0.1}, {"metric": "spotwise_cosine", "value": 0.9}, {"metric": "mean_celltype_pearson", "value": 0.8}])
    _write_csv(tables_dir / "table_3_baseline_comparison.csv", [{"method": "WaveST-Gate", "jsd": 0.1}, {"method": "BayesPrism", "jsd": 0.2}])
    _write_csv(tables_dir / "table_4_ablation_delta.csv", [{"ablation": "full"}])
    _write_csv(
        tables_dir / "table_5_reliability_boundary_niche.csv",
        [
            {"metric": "uncertainty_error_pearson", "value": 0.5},
            {"metric": "calibration_bin_pearson", "value": 0.9},
            {"metric": "boundary_to_interior_jump_ratio", "value": 2.0},
            {"metric": "num_niches", "value": 5},
        ],
    )
    _write_csv(tables_dir / "table_7_robustness_summary.csv", [{"scenario": "patch_size"}])
    _write_csv(
        tables_dir / "table_8_split_sensitivity.csv",
        [
            {"analysis": "primary_spatial_holdout", "split": "val", "num_supervised_spots": 0},
            {"analysis": "supplementary_gt_stratified_sensitivity", "split": "val", "mean_jsd": 0.1},
        ],
    )
    _write_csv(tables_dir / "table_9_imagegate_supplement.csv", [{"analysis": "run_level_imagegate_control"}])
    _write_csv(tables_dir / "table_10_benchmark_sensitivity.csv", [{"analysis": "radius_cell_count_metric", "radius": 55, "min_xenium_cells": 1, "jsd": 0.1}])
    _write_csv(
        tables_dir / "table_11_rep1_retune_budget_curve.csv",
        [
            {"budget_steps": 0, "best_baseline_jsd": 0.2, "beats_best_baseline": False, "jsd": 0.3},
            {"budget_steps": 25, "best_baseline_jsd": 0.2, "beats_best_baseline": True, "jsd": 0.1},
        ],
    )

    release_dir = tmp_path / "release"
    _write_json(release_dir / "release_bundle_manifest.json", {"bundle_sha256": "abc"})
    _write_json(release_dir / "release_verification.json", {"doi_status": "pending_token", "overall_status": "passed_with_warnings", "bundle_integrity_status": "passed", "num_failures": 0, "critical_artifacts_checked": 1})
    _write_json(release_dir / "zenodo_deposition_result.json", {"release_status": "dry_run_token_missing", "doi": ""})
    _touch(release_dir / "final_submission_handoff.json")

    figure_legends_dir = tmp_path / "figure_legends"
    _write_json(figure_legends_dir / "figure_legends.json", {"status": "complete"})
    methods_dir = tmp_path / "methods"
    _write_json(methods_dir / "manuscript_methods.json", {"status": "complete"})
    statements_dir = tmp_path / "statements"
    _write_json(statements_dir / "manuscript_availability_statements.json", {"status": "complete"})

    external_dir = tmp_path / "external"
    _touch(external_dir / "external_no_retuning_summary.csv")
    external_matched = tmp_path / "external_matched"
    _touch(external_matched / "external_matched_gt_summary.csv")
    minimal = tmp_path / "minimal"
    _touch(minimal / "matched_multisample_baseline_summary.csv")
    pathology = tmp_path / "pathology"
    _touch(pathology / "pathology_niche_summary.csv")
    for path in [
        tmp_path / f"{run_dir.name}_imagegate" / "image_contribution" / "image_contribution_summary.json",
        tmp_path / f"{run_dir.name}_imagegate" / "image_contribution" / "image_contribution_texture_groups.csv",
        tmp_path / f"{run_dir.name}_imagegate_noimage" / "metrics.csv",
        external_matched / "xenium_rep1_pseudospots_radius55_common297" / "rep1_retune_budget_curve" / "rep1_minimal_retune_budget_curve.csv",
        external_matched / "xenium_rep1_pseudospots_radius55_common297" / "wavestgate_minimal_retune" / "test_formal_comparison" / "baseline_comparison.csv",
    ]:
        _touch(path)

    payload = build_reviewer_preflight(
        output_dir=tmp_path / "preflight",
        readiness_report_path=readiness_path,
        release_dir=release_dir,
        tables_dir=tables_dir,
        run_dir=run_dir,
        benchmark_dir=benchmark_dir,
        figure_legends_dir=figure_legends_dir,
        methods_dir=methods_dir,
        statements_dir=statements_dir,
        docs_dir=docs_dir,
        external_dir=external_dir,
        external_matched_gt_dir=external_matched,
        minimal_retune_dir=minimal,
        external_pathology_dir=pathology,
        benchmark_datasheet_dir=benchmark_datasheet_dir,
        completion_audit_dir=completion_audit_dir,
    )

    assert payload["status"] == "complete_except_project_doi"
    assert payload["project_doi_status"] == "pending_token"
    assert payload["missing_evidence"] == []
    assert len(payload["headline_claims"]) >= 8
    assert len(payload["reviewer_concerns"]) >= 8
    assert Path(payload["outputs"]["json"]).exists()
    assert Path(payload["outputs"]["markdown"]).exists()
    preflight_text = Path(payload["outputs"]["json"]).read_text(encoding="utf-8")
    assert "critical_artifacts_checked" not in preflight_text
    assert "tar_member_count" not in preflight_text
    assert "bundle_bytes" not in preflight_text
    assert "num_files=" not in preflight_text
