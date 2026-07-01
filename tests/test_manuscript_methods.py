import json
from pathlib import Path

import pandas as pd

from wavestgate.evaluation.manuscript_methods import build_manuscript_methods


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_manuscript_methods_outputs_required_sections(tmp_path: Path) -> None:
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
    run_dir = tmp_path / "run"
    _write_csv(
        run_dir / "metrics.csv",
        [
            {
                "step": 4,
                "jsd": 0.1,
                "spotwise_cosine": 0.9,
                "mean_celltype_pearson": 0.8,
                "expression_log1p_rmse": 0.5,
            }
        ],
    )
    _write_csv(
        run_dir / "baseline_comparison" / "baseline_comparison.csv",
        [
            {"method": "WaveST-Gate", "rank_by_jsd": 1, "jsd": 0.1, "paired_permutation_p": ""},
            {"method": "RCTD", "rank_by_jsd": 2, "jsd": 0.2, "paired_permutation_p": 0.01},
        ],
    )
    _write_csv(run_dir / "ablations250" / "ablation_summary.csv", [{"ablation": "full", "jsd": 0.1}, {"ablation": "no_gate_mean_fusion", "jsd": 0.2}])
    tables_dir = tmp_path / "tables"
    _write_csv(tables_dir / "table_1_benchmark_summary.csv", [{"item": "spots", "value": 10}])
    _write_csv(tables_dir / "table_2_main_model_performance.csv", [{"metric": "JSD", "value": 0.1}])
    _write_csv(tables_dir / "table_3_baseline_comparison.csv", [{"method": "WaveST-Gate", "rank_by_jsd": 1, "jsd": 0.1}, {"method": "RCTD", "rank_by_jsd": 2, "jsd": 0.2}])
    _write_csv(tables_dir / "table_4_ablation_delta.csv", [{"ablation": "full", "jsd": 0.1}, {"ablation": "no_gate_mean_fusion", "jsd": 0.2}])
    _write_csv(tables_dir / "table_5_reliability_boundary_niche.csv", [{"claim_area": "reliability", "metric": "uncertainty_error_pearson", "value": 0.5}])
    _write_csv(tables_dir / "table_6_external_generalization.csv", [{"setting": "external", "datasets": 1}])
    _write_csv(tables_dir / "table_7_robustness_summary.csv", [{"scenario": "patch_size", "n_rows": 4}])
    _write_csv(
        tables_dir / "table_8_split_sensitivity.csv",
        [
            {"analysis": "primary_spatial_holdout", "split": "val", "num_supervised_spots": 0},
            {"analysis": "primary_spatial_holdout", "split": "test", "num_supervised_spots": 2},
            {"analysis": "supplementary_gt_stratified_sensitivity", "split": "val", "num_supervised_spots": 2, "mean_jsd": 0.11},
        ],
    )
    _write_csv(tables_dir / "table_9_imagegate_supplement.csv", [{"analysis": "run_level_imagegate_control"}])
    _write_csv(
        tables_dir / "table_10_benchmark_sensitivity.csv",
        [{"analysis": "radius_cell_count_metric", "radius": 55, "min_xenium_cells": 1, "num_spots_passing_threshold": 6, "jsd": 0.1}],
    )
    _write_csv(
        tables_dir / "table_11_rep1_retune_budget_curve.csv",
        [
            {"budget_steps": 0, "best_baseline_method": "RCTD", "best_baseline_jsd": 0.2, "beats_best_baseline": False, "jsd": 0.3},
            {"budget_steps": 25, "best_baseline_method": "RCTD", "best_baseline_jsd": 0.2, "beats_best_baseline": True, "jsd": 0.12},
        ],
    )
    environment = tmp_path / "environment_report.json"
    _write_json(environment, {"torch": {"version": "2.6.0+cu124", "cuda_version": "12.4", "device_count": 1, "devices": [{"name": "NVIDIA GeForce RTX 4090"}]}})

    payload = build_manuscript_methods(
        output_dir=tmp_path / "methods",
        tables_dir=tables_dir,
        run_dir=run_dir,
        benchmark_dir=benchmark_dir,
        environment_report_path=environment,
        statements_dir=tmp_path / "statements",
    )

    assert payload["status"] == "complete"
    assert len(payload["sections"]) == 10
    titles = {section["title"] for section in payload["sections"]}
    assert "Split sensitivity and benchmark confidence controls" in titles
    assert "Image modality contribution control" in titles
    assert "Baseline comparison and fairness" in titles
    assert "Statistical analysis, reproducibility, and compute" in titles
    assert payload["missing_required_paths"] == []
    assert Path(payload["outputs"]["json"]).exists()
    assert Path(payload["outputs"]["markdown"]).exists()
    markdown = Path(payload["outputs"]["markdown"]).read_text(encoding="utf-8")
    assert "WaveST-Gate Methods" in markdown
    assert "RCTD" in markdown
