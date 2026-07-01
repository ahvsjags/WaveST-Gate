import json
from pathlib import Path

import pandas as pd

from wavestgate.evaluation.submission_readiness import _release_deposition_summary, generate_submission_readiness


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("artifact", encoding="utf-8")


def test_release_deposition_summary_uses_project_doi_only(tmp_path):
    release_dir = tmp_path / "release"
    _write_json(
        release_dir / "zenodo_metadata.json",
        {
            "metadata": {
                "title": "test",
                "related_identifiers": [{"identifier": "10.5281/zenodo.4739739", "relation": "cites"}],
            }
        },
    )
    _write_json(release_dir / "release_bundle_manifest.json", {"release_status": "zenodo_ready_not_deposited", "doi": ""})

    summary = _release_deposition_summary(release_dir)
    assert summary["doi"] == ""

    _write_json(
        release_dir / "zenodo_deposition_result.json",
        {
            "release_status": "zenodo_draft_reserved",
            "doi": "10.5072/zenodo.12345",
            "zenodo_deposition_id": "12345",
        },
    )
    summary = _release_deposition_summary(release_dir)
    assert summary["doi"] == "10.5072/zenodo.12345"
    assert summary["release_status"] == "zenodo_draft_reserved"


def test_submission_readiness_report_generation(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "xenium_to_visium_benchmark_protocol.md").write_text("protocol", encoding="utf-8")

    benchmark_dir = tmp_path / "benchmark"
    _write_json(
        benchmark_dir / "xenium_visium_benchmark_manifest.json",
        {
            "num_spots": 10,
            "num_spots_with_ground_truth": 6,
            "num_cells": 100,
            "num_cell_types": 3,
            "spot_radius": 55,
        },
    )
    _write_csv(benchmark_dir / "xenium_cell_counts.csv", [{"spot_id": "s1", "A": 2}])
    _write_csv(benchmark_dir / "xenium_cell_proportions.csv", [{"spot_id": "s1", "A": 1.0}])
    _write_csv(
        benchmark_dir / "spot_ground_truth_qc.csv",
        [
            {
                "spot_id": "s1",
                "xenium_cell_count": 2,
                "ground_truth_entropy": 0.0,
                "dominant_cell_type": "A",
                "has_xenium_ground_truth": True,
            }
        ],
    )
    _write_csv(benchmark_dir / "spot_splits.csv", [{"spot_id": "s1", "split": "test"}])
    benchmark_datasheet_dir = tmp_path / "benchmark_datasheet"
    _write_json(
        benchmark_datasheet_dir / "benchmark_datasheet.json",
        {
            "status": "complete",
            "failed_integrity_checks": [],
            "artifacts": [{"path": str(benchmark_dir / "xenium_cell_counts.csv")}],
            "cell_type_summary": [{"cell_type": "A"}],
            "column_dictionary": [{"file": "xenium_cell_counts.csv", "column": "spot_id"}],
        },
    )
    _touch(benchmark_datasheet_dir / "benchmark_datasheet.md")
    completion_audit_dir = tmp_path / "completion_audit"
    _write_json(
        completion_audit_dir / "goal_completion_audit.json",
        {
            "overall_status": "complete_except_project_doi",
            "num_complete": 10,
            "num_partial": 1,
            "num_missing": 0,
            "requirements": [{"requirement": f"req_{idx}"} for idx in range(11)],
        },
    )
    _touch(completion_audit_dir / "goal_completion_audit.md")
    release_dir = tmp_path / "release"
    _write_json(
        release_dir / "release_bundle_manifest.json",
        {"bundle_path": str(release_dir / "bundle.tar.gz"), "num_files": 3, "bundle_bytes": 12},
    )
    _touch(release_dir / "bundle.tar.gz")
    _write_json(release_dir / "zenodo_metadata.json", {"metadata": {"title": "test"}})
    _write_csv(release_dir / "release_upload_manifest.csv", [{"path": "a", "bytes": 1, "sha256": "abc"}])
    _touch(release_dir / "zenodo_deposition_instructions.md")
    _write_json(
        release_dir / "release_verification.json",
        {
            "overall_status": "passed_with_warnings",
            "bundle_integrity_status": "passed",
            "doi_status": "pending_token",
            "tar_member_count": 5,
            "critical_artifacts_checked": 1,
            "num_failures": 0,
        },
    )
    _write_json(
        release_dir / "environment_report.json",
        {
            "python": {"version": "3.12.0"},
            "packages": {"torch": {"available": True, "version": "2.0"}},
            "torch": {"version": "2.0", "cuda_available": True, "device_count": 1},
        },
    )
    _touch(release_dir / "environment_report.md")

    run_dir = tmp_path / "run"
    for name in [
        "checkpoint.pt",
        "predicted_proportions.csv",
        "reconstructed_expression.csv",
        "gate_weights.csv",
        "spot_uncertainty.csv",
        "agent_attention.csv",
    ]:
        _touch(run_dir / name)
    _write_csv(run_dir / "training_history.csv", [{"step": 0, "loss_total": 1.0}])
    _write_csv(
        run_dir / "metrics.csv",
        [
            {
                "step": 4,
                "jsd": 0.1,
                "spotwise_cosine": 0.9,
                "mean_celltype_pearson": 0.8,
                "num_supervised_spots": 6,
            }
        ],
    )

    nature_dir = run_dir / "nature_analysis"
    for name in [
        "spot_uncertainty_map.png",
        "proportion_maps/proportion_map_manifest.json",
        "proportion_maps/predicted_top_celltypes_panel.png",
        "proportion_maps/predicted_tumor_immune_stromal_group_panel.png",
        "image_gate_map.png",
        "expression_gate_map.png",
        "reference_gate_map.png",
        "niche_map.png",
        "reliability_spot_errors.csv",
        "risk_coverage_curve.csv",
        "risk_coverage_curve.png",
        "uncertainty_calibration_bins.csv",
        "uncertainty_calibration.png",
        "failure_case_candidates.csv",
        "boundary_edge_jumps.csv",
        "boundary_type_summary.csv",
        "boundary_marker_validation.csv",
        "boundary_sharpness_map.png",
        "niche_assignments.csv",
        "niche_composition.csv",
        "niche_marker_enrichment.csv",
        "niche_biological_summary.csv",
        "gate_reliability_by_niche.csv",
        "agent_attention_by_niche.csv",
    ]:
        _touch(nature_dir / name)
    _write_json(
        nature_dir / "reliability_summary.json",
        {
            "uncertainty_error_pearson": 0.5,
            "uncertainty_error_spearman": 0.6,
            "risk_gap": 0.1,
            "calibration_bin_pearson": 0.9,
        },
    )
    _write_json(
        nature_dir / "boundary_summary.json",
        {
            "boundary_to_interior_jump_ratio": 2.0,
            "mean_boundary_jump": 1.0,
            "comparison_mean_boundary_jump": 0.8,
        },
    )
    _write_json(nature_dir / "niche_summary.json", {"num_niches": 3})

    figures_dir = tmp_path / "figures"
    _write_json(figures_dir / "figure_manifest.json", {"num_figures": 6, "num_pass": 6, "num_fail": 0})
    _write_csv(figures_dir / "figure_manifest.csv", [{"figure": "Figure 1", "status": "pass"}])
    _touch(figures_dir / "figure_manifest.md")
    for name in [
        "figure_1_workflow_schematic.png",
        "figure_2_spatial_cell_composition.png",
        "figure_3_baseline_performance.png",
        "figure_4_reliability_calibration.png",
        "figure_5_boundary_niche_pathology.png",
    ]:
        _touch(figures_dir / name)

    figure_legends_dir = tmp_path / "figure_legends"
    _write_json(
        figure_legends_dir / "figure_legends.json",
        {
            "status": "complete",
            "missing_evidence": [],
            "figure_manifest_status": {"num_figures": 6, "num_pass": 6, "num_fail": 0},
            "legends": [{"figure": f"Figure {idx}"} for idx in range(1, 6)] + [{"figure": "Supplementary Figure S1"}],
        },
    )
    _touch(figure_legends_dir / "figure_legends.md")

    statements_dir = tmp_path / "statements"
    _write_json(
        statements_dir / "manuscript_availability_statements.json",
        {
            "release": {"status": "dry_run_token_missing", "doi_status": "pending_token", "doi": "", "deposition_id": ""},
            "statements": [
                {"title": "Data Availability"},
                {"title": "Code Availability"},
                {"title": "Reproducibility"},
                {"title": "Computing Environment"},
            ],
        },
    )
    _touch(statements_dir / "manuscript_availability_statements.md")

    methods_dir = tmp_path / "methods"
    _write_json(
        methods_dir / "manuscript_methods.json",
        {
            "status": "complete",
            "missing_required_paths": [],
            "sections": [
                {"title": "Xenium-to-Visium benchmark construction"},
                {"title": "WaveST-Gate model"},
                {"title": "Training objective and optimization"},
                {"title": "Baseline comparison and fairness"},
                {"title": "Ablation design"},
                {"title": "Reliability, boundary, and niche analyses"},
                {"title": "External generalization and robustness"},
                {"title": "Statistical analysis, reproducibility, and compute"},
            ],
        },
    )
    _touch(methods_dir / "manuscript_methods.md")

    reviewer_preflight_dir = tmp_path / "reviewer_preflight"
    _write_json(
        reviewer_preflight_dir / "reviewer_preflight.json",
        {
            "status": "complete_except_project_doi",
            "missing_evidence": [],
            "headline_claims": [{"claim": f"claim_{idx}"} for idx in range(8)],
            "reviewer_concerns": [{"concern": f"concern_{idx}"} for idx in range(8)],
        },
    )
    _touch(reviewer_preflight_dir / "reviewer_preflight.md")

    _write_csv(
        run_dir / "baseline_comparison" / "baseline_comparison.csv",
        [
            {
                "method": method,
                "runtime_seconds": 1.0,
                "peak_cuda_memory_mb": 0.0,
                "paired_permutation_p": 0.01,
            }
            for method in [
                "WaveST-Gate",
                "cell2location",
                "RCTD",
                "CARD",
                "Tangram",
                "SpatialDWLS",
                "SpatialDWLS/Seurat",
                "BayesPrism",
                "SPOTlight",
            ]
        ],
    )
    _write_csv(run_dir / "baseline_comparison" / "baseline_split_bootstrap_metrics.csv", [{"method": "WaveST-Gate"}])
    _write_csv(run_dir / "baseline_comparison" / "baseline_split_bootstrap_summary.csv", [{"method": "WaveST-Gate"}])
    _write_csv(run_dir / "baseline_comparison" / "baseline_bootstrap_paired_improvement.csv", [{"method": "cell2location"}])
    _write_json(
        run_dir / "baseline_comparison" / "baseline_statistics_manifest.json",
        {"n_methods": 8, "n_supervised_spots": 6, "n_bootstraps": 5},
    )
    _write_json(run_dir / "baseline_environment_audit.json", {"status": {"SpatialDWLS/Seurat": "ready"}})
    _write_csv(run_dir / "spatialdwls_giotto_baseline" / "spatialdwls_giotto_metrics.csv", [{"method": "SpatialDWLS/Seurat", "jsd": 0.2, "spotwise_cosine": 0.8, "runtime_seconds": 10}])
    _write_csv(run_dir / "spatialdwls_giotto_baseline" / "spatialdwls_giotto_proportions.csv", [{"spot_id": "s1", "A": 1.0}])
    _write_json(run_dir / "spatialdwls_giotto_baseline" / "spatialdwls_giotto_manifest.json", {"method": "SpatialDWLS/Seurat"})
    _write_json(run_dir / "spatialdwls_giotto_baseline" / "spatialdwls_giotto_python_manifest.json", {"method": "SpatialDWLS/Seurat"})
    _write_csv(
        run_dir / "ablations250" / "ablation_summary.csv",
        [
            {"ablation": ablation}
            for ablation in [
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
            ]
        ],
    )

    external_dir = tmp_path / "external"
    datasets = [f"external_{i}" for i in range(4)] + [f"wu_swarbrick_{i}" for i in range(6)]
    _write_csv(external_dir / "external_no_retuning_summary.csv", [{"dataset": dataset} for dataset in datasets])
    for dataset in datasets:
        for name in [
            "predicted_proportions.csv",
            "gate_weights.csv",
            "spot_uncertainty.csv",
            "agent_attention.csv",
            "aligned_prediction_metrics.csv",
            "aligned_prediction_manifest.json",
        ]:
            _touch(external_dir / dataset / name)

    external_matched = tmp_path / "external_matched"
    matched_dataset = external_matched / "xenium_rep1_pseudospots_radius55_common297"
    _write_csv(
        external_matched / "external_matched_gt_summary.csv",
        [
            {
                "dataset": "xenium_rep1_pseudospots_radius55_common297",
                "num_spots": 3,
                "model_jsd": 0.1,
                "model_spotwise_cosine": 0.9,
                "wavestgate_rank_by_jsd": 1,
            }
        ],
    )
    _write_json(external_matched / "external_matched_gt_manifest.json", {"num_datasets": 1})
    _write_csv(matched_dataset / "matched_gt_metrics.csv", [{"jsd": 0.1}])
    _write_csv(matched_dataset / "comparison" / "baseline_comparison.csv", [{"method": "WaveST-Gate", "jsd": 0.1}])

    robustness_rows = [
        {"scenario": "gene_dropout", "level": 0.1},
        {"scenario": "gene_dropout", "level": 0.3},
        {"scenario": "gene_dropout", "level": 0.5},
        {"scenario": "he_perturbation", "level": "stain_darken"},
        {"scenario": "subgroup", "level": "low_expression_spots"},
    ]
    robustness_rows.extend({"scenario": "reference_missing_celltype", "level": f"cell_{idx}"} for idx in range(10))
    _write_csv(run_dir / "robustness" / "robustness_summary.csv", robustness_rows)
    _write_csv(
        run_dir / "patch_size_robustness" / "patch_size_summary.csv",
        [{"patch_size": size} for size in [32, 64, 128, 256]],
    )

    outputs = generate_submission_readiness(
        benchmark_dir=benchmark_dir,
        run_dir=run_dir,
        external_dir=external_dir,
        output_dir=tmp_path / "readiness",
        docs_dir=docs_dir,
        release_dir=release_dir,
        external_matched_gt_dir=external_matched,
        manuscript_figures_dir=figures_dir,
        manuscript_statements_dir=statements_dir,
        manuscript_methods_dir=methods_dir,
        manuscript_figure_legends_dir=figure_legends_dir,
        reviewer_preflight_dir=reviewer_preflight_dir,
        benchmark_datasheet_dir=benchmark_datasheet_dir,
        completion_audit_dir=completion_audit_dir,
    )

    report = json.loads(Path(outputs["readiness_report_json"]).read_text(encoding="utf-8"))
    assert report["stage_summary"]["2. Main model training"]["status"] == "complete"
    assert report["stage_summary"]["1. Xenium-to-Visium benchmark"]["status"] == "partial"
    records = {record["requirement"]: record for record in report["records"]}
    assert records["benchmark datasheet and data dictionary"]["status"] == "complete"
    assert records["requirement-by-requirement goal completion audit"]["status"] == "complete"
    assert Path(outputs["evidence_manifest"]).exists()
    release = json.loads(Path(outputs["zenodo_release_manifest"]).read_text(encoding="utf-8"))
    assert release["release_status"] == "prepared_not_deposited"
