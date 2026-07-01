import json
from pathlib import Path

import pandas as pd

from wavestgate.evaluation.manuscript_figure_legends import build_manuscript_figure_legends


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("artifact", encoding="utf-8")


def test_build_manuscript_figure_legends_outputs_claim_evidence(tmp_path: Path) -> None:
    figures_dir = tmp_path / "figures"
    _write_json(figures_dir / "figure_manifest.json", {"num_figures": 6, "num_pass": 6, "num_fail": 0})
    for name in [
        "figure_1_workflow_schematic.png",
        "figure_2_spatial_cell_composition.png",
        "figure_3_baseline_performance.png",
        "figure_4_reliability_calibration.png",
        "figure_5_boundary_niche_pathology.png",
        "supplementary_figure_s1_robustness.png",
    ]:
        _touch(figures_dir / name)

    benchmark_dir = tmp_path / "benchmark"
    _write_json(
        benchmark_dir / "xenium_visium_benchmark_manifest.json",
        {"num_spots": 10, "num_spots_with_ground_truth": 6, "num_cells": 100, "num_cell_types": 3, "spot_radius": 55},
    )
    tables_dir = tmp_path / "tables"
    _write_csv(
        tables_dir / "table_2_main_model_performance.csv",
        [
            {"metric": "JSD", "value": 0.1},
            {"metric": "spotwise_cosine", "value": 0.9},
            {"metric": "mean_celltype_pearson", "value": 0.8},
            {"metric": "proportion_map_cell_types", "value": 3},
            {"metric": "priority_cell_types_mapped", "value": 2},
        ],
    )
    _write_csv(
        tables_dir / "table_3_baseline_comparison.csv",
        [
            {"method": "WaveST-Gate", "rank_by_jsd": 1, "jsd": 0.1, "paired_permutation_p": ""},
            {"method": "RCTD", "rank_by_jsd": 2, "jsd": 0.2, "paired_permutation_p": 0.01},
        ],
    )
    _write_csv(
        tables_dir / "table_5_reliability_boundary_niche.csv",
        [
            {"metric": "uncertainty_error_pearson", "value": 0.5},
            {"metric": "calibration_bin_pearson", "value": 0.9},
            {"metric": "boundary_to_interior_jump_ratio", "value": 2.0},
            {"metric": "num_niches", "value": 5},
            {"metric": "spot_weighted_pathology_agreement_rate", "value": 0.8},
        ],
    )
    _write_csv(tables_dir / "table_7_robustness_summary.csv", [{"scenario": "patch_size"}, {"scenario": "gene_dropout"}])

    run_dir = tmp_path / "run"
    for path in [
        run_dir / "predicted_proportions.csv",
        run_dir / "metrics.csv",
        run_dir / "nature_analysis" / "proportion_maps" / "proportion_map_manifest.json",
        run_dir / "baseline_comparison" / "baseline_comparison.csv",
        run_dir / "baseline_comparison" / "baseline_split_bootstrap_summary.csv",
        run_dir / "baseline_comparison" / "baseline_bootstrap_paired_improvement.csv",
        run_dir / "gate_weights.csv",
        run_dir / "spot_uncertainty.csv",
        run_dir / "nature_analysis" / "reliability_summary.json",
        run_dir / "nature_analysis" / "risk_coverage_curve.csv",
        run_dir / "nature_analysis" / "uncertainty_calibration_bins.csv",
        run_dir / "nature_analysis" / "boundary_summary.json",
        run_dir / "nature_analysis" / "niche_summary.json",
        run_dir / "nature_analysis" / "niche_biological_summary.csv",
        run_dir / "nature_analysis" / "niche_xenium_neighborhood_summary.csv",
        run_dir / "robustness" / "robustness_summary.csv",
        run_dir / "robustness" / "robustness_manifest.json",
        run_dir / "patch_size_robustness" / "patch_size_summary.csv",
    ]:
        _touch(path)
    methods_dir = tmp_path / "methods"
    _write_json(methods_dir / "manuscript_methods.json", {"status": "complete"})

    payload = build_manuscript_figure_legends(
        output_dir=tmp_path / "legends",
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        run_dir=run_dir,
        benchmark_dir=benchmark_dir,
        methods_dir=methods_dir,
    )

    assert payload["status"] == "complete"
    assert len(payload["legends"]) == 6
    assert payload["missing_evidence"] == []
    assert Path(payload["outputs"]["json"]).exists()
    assert Path(payload["outputs"]["markdown"]).exists()
    markdown = Path(payload["outputs"]["markdown"]).read_text(encoding="utf-8")
    assert "Formal baseline comparison" in markdown
    assert "Supplementary Figure S1" in markdown
