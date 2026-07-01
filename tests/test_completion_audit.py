import json
from pathlib import Path

import pandas as pd

from wavestgate.evaluation.completion_audit import build_completion_audit


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("artifact", encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_completion_audit_complete_except_project_doi(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    docs = tmp_path / "docs"
    _touch(docs / "xenium_to_visium_benchmark_protocol.md")
    _write_json(
        benchmark / "xenium_visium_benchmark_manifest.json",
        {
            "spot_radius": 55,
            "columns": {"cell_type_col": "cell_type", "cell_x_col": "x", "cell_y_col": "y"},
            "artifacts": {"counts": "xenium_cell_counts.csv"},
        },
    )
    _write_csv(benchmark / "xenium_cell_counts.csv", [{"spot_id": "s1", "A": 2}])
    _write_csv(benchmark / "xenium_cell_proportions.csv", [{"spot_id": "s1", "A": 1.0}])
    _write_csv(
        benchmark / "spot_ground_truth_qc.csv",
        [{"spot_id": "s1", "xenium_cell_count": 2, "ground_truth_entropy": 0.0, "dominant_cell_type": "A", "has_xenium_ground_truth": True}],
    )
    _write_csv(benchmark / "spot_splits.csv", [{"spot_id": "s1", "split": "test"}])

    datasheet = tmp_path / "datasheet"
    _write_json(
        datasheet / "benchmark_datasheet.json",
        {"status": "complete", "failed_integrity_checks": [], "cell_type_summary": [{"cell_type": "A"}], "column_dictionary": [{"column": "spot_id"}]},
    )
    _touch(datasheet / "benchmark_datasheet.md")

    release = tmp_path / "release"
    for name in ["release_bundle_manifest.json", "zenodo_metadata.json", "zenodo_deposition_result.json", "release_verification.json"]:
        _write_json(release / name, {"doi": "", "zenodo_deposition_id": ""})

    run = tmp_path / "run"
    for name in [
        "checkpoint.pt",
        "predicted_proportions.csv",
        "reconstructed_expression.csv",
        "gate_weights.csv",
        "modality_reliability.csv",
        "spot_uncertainty.csv",
        "agent_attention.csv",
        "training_history.csv",
    ]:
        _touch(run / name)
    _write_csv(run / "metrics.csv", [{"jsd": 0.1, "spotwise_cosine": 0.9, "mean_celltype_pearson": 0.8, "num_supervised_spots": 1}])

    nature = run / "nature_analysis"
    for name in [
        "proportion_maps/proportion_map_manifest.json",
        "proportion_maps/predicted_top_celltypes_panel.png",
        "proportion_maps/predicted_tumor_immune_stromal_group_panel.png",
        "image_gate_map.png",
        "expression_gate_map.png",
        "reference_gate_map.png",
        "spot_uncertainty_map.png",
        "niche_map.png",
        "reliability_spot_errors.csv",
        "risk_coverage_curve.csv",
        "risk_coverage_curve.png",
        "uncertainty_calibration_bins.csv",
        "uncertainty_calibration.png",
        "failure_case_candidates.csv",
        "boundary_edge_jumps.csv",
        "boundary_marker_validation.csv",
        "boundary_sharpness_map.png",
        "boundary_he_overlay.png",
        "boundary_he_pathology_proxy.csv",
        "niche_assignments.csv",
        "niche_composition.csv",
        "niche_marker_enrichment.csv",
        "niche_biological_summary.csv",
        "niche_xenium_neighborhood_validation.csv",
        "niche_xenium_neighborhood_summary.csv",
        "gate_reliability_by_niche.csv",
        "agent_attention_by_niche.csv",
    ]:
        _touch(nature / name)
    _write_json(nature / "reliability_summary.json", {"uncertainty_error_pearson": 0.5, "risk_gap": 0.1, "calibration_bin_pearson": 0.9})
    _write_json(nature / "boundary_summary.json", {"boundary_to_interior_jump_ratio": 2.0})
    _write_csv(nature / "boundary_type_summary.csv", [{"mean_comparison_l1_jump": 0.1, "boundary_preservation_delta_vs_comparison": 0.2}])
    _write_json(nature / "niche_summary.json", {"num_niches": 3})

    methods = ["WaveST-Gate", "cell2location", "RCTD multi", "CARD", "Tangram", "SpatialDWLS", "SpatialDWLS/Seurat", "BayesPrism"]
    _write_csv(
        run / "baseline_comparison" / "baseline_comparison.csv",
        [{"method": method, "runtime_seconds": 1.0, "peak_cuda_memory_mb": 0.0, "paired_permutation_p": 0.01} for method in methods],
    )
    for name in ["baseline_split_bootstrap_metrics.csv", "baseline_split_bootstrap_summary.csv", "baseline_bootstrap_paired_improvement.csv"]:
        _write_csv(run / "baseline_comparison" / name, [{"method": "WaveST-Gate"}])
    _write_json(run / "baseline_comparison" / "baseline_statistics_manifest.json", {"n_methods": len(methods)})
    _write_json(run / "baseline_environment_audit.json", {"status": {"cell2location": "ready"}})

    ablations = [
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
    _write_csv(run / "ablations250" / "ablation_summary.csv", [{"ablation": ablation} for ablation in ablations])

    pathology = tmp_path / "pathology"
    for name in ["pathology_class_summary.csv", "pathology_niche_summary.csv", "pathology_niche_by_class.csv"]:
        _touch(pathology / name)

    external = tmp_path / "external"
    datasets = [f"external_{idx}" for idx in range(10)]
    _write_csv(external / "external_no_retuning_summary.csv", [{"dataset": dataset} for dataset in datasets])
    for dataset in datasets:
        for name in ["predicted_proportions.csv", "gate_weights.csv", "spot_uncertainty.csv", "agent_attention.csv", "aligned_prediction_metrics.csv", "aligned_prediction_manifest.json"]:
            _touch(external / dataset / name)

    matched = tmp_path / "matched"
    for path in [
        matched / "external_matched_gt_summary.csv",
        matched / "external_matched_gt_manifest.json",
        matched / "xenium_rep1_pseudospots_radius55_common297" / "matched_gt_metrics.csv",
        matched / "xenium_rep1_pseudospots_radius55_common297" / "wavestgate_minimal_retune" / "test_metrics.csv",
        matched / "xenium_rep1_pseudospots_radius55_common297" / "wavestgate_minimal_retune" / "test_formal_comparison" / "baseline_comparison.csv",
        matched / "xenium_rep1_pseudospots_radius55_common297" / "rep1_retune_budget_curve" / "rep1_minimal_retune_budget_curve.csv",
        matched / "xenium_rep1_pseudospots_radius55_common297" / "rep1_retune_budget_curve" / "rep1_minimal_retune_budget_curve_manifest.json",
    ]:
        _touch(path)
    multisample = tmp_path / "multi"
    _touch(multisample / "matched_multisample_baseline_summary.csv")
    minimal_multi = tmp_path / "minimal_multi"
    _touch(minimal_multi / "matched_multisample_baseline_summary.csv")

    robustness_rows = [
        {"scenario": "gene_dropout", "level": 0.1},
        {"scenario": "gene_dropout", "level": 0.3},
        {"scenario": "gene_dropout", "level": 0.5},
        {"scenario": "gene_panel", "level": "top200_variance"},
        {"scenario": "gene_panel", "level": "top100_variance"},
        {"scenario": "gene_panel", "level": "marker_only"},
        {"scenario": "he_perturbation", "level": "noise"},
        {"scenario": "subgroup", "level": "low_expression_spots"},
        {"scenario": "subgroup", "level": "low_cell_count_spots"},
        {"scenario": "split", "level": "random_1"},
    ]
    robustness_rows.extend({"scenario": "reference_missing_celltype", "level": f"cell_{idx}"} for idx in range(10))
    robustness_rows.extend({"scenario": "prototype_perturbation", "level": f"noise_{idx}"} for idx in range(4))
    _write_csv(run / "robustness" / "robustness_summary.csv", robustness_rows)
    _write_json(run / "robustness" / "robustness_manifest.json", {"n": len(robustness_rows)})
    _write_csv(run / "patch_size_robustness" / "patch_size_summary.csv", [{"patch_size": size} for size in [32, 64, 128, 256]])
    for path in [
        run / "split_sensitivity" / "original_split_gt_summary.csv",
        run / "split_sensitivity" / "split_sensitivity_aggregate.csv",
        run / "split_sensitivity" / "split_sensitivity_manifest.json",
        run / "benchmark_sensitivity" / "radius_cell_count_sensitivity.csv",
        run / "benchmark_sensitivity" / "radius_coverage_summary.csv",
        run / "benchmark_sensitivity" / "radius_cell_count_sensitivity_manifest.json",
    ]:
        _touch(path)

    payload = build_completion_audit(
        benchmark_dir=benchmark,
        run_dir=run,
        external_dir=external,
        external_matched_gt_dir=matched,
        external_pathology_dir=pathology,
        multisample_dir=multisample,
        minimal_multisample_dir=minimal_multi,
        datasheet_dir=datasheet,
        release_dir=release,
        docs_dir=docs,
        output_dir=tmp_path / "audit",
    )

    assert payload["overall_status"] == "complete_except_project_doi"
    assert payload["num_missing"] == 0
    assert payload["num_partial"] == 1
    pending = [record for record in payload["requirements"] if record["status"] == "partial"]
    assert pending[0]["requirement"] == "project Zenodo DOI and deposition"
    assert Path(payload["outputs"]["json"]).exists()
    assert Path(payload["outputs"]["markdown"]).exists()
