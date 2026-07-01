import json
from pathlib import Path

import pandas as pd

from wavestgate.evaluation.manuscript_tables import build_arg_parser, build_manuscript_tables


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manuscript_tables_build_expected_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    bench_dir = tmp_path / "benchmark"
    external_dir = tmp_path / "external"
    output_dir = tmp_path / "tables"

    _write_json(
        bench_dir / "xenium_visium_benchmark_manifest.json",
        {"num_spots": 10, "num_spots_with_ground_truth": 4, "num_cells": 100, "num_cell_types": 3, "spot_radius": 55},
    )
    _write_csv(
        bench_dir / "spot_ground_truth_qc.csv",
        [{"spot_id": "s1", "xenium_cell_count": 5, "ground_truth_entropy": 0.2}],
    )
    _write_csv(
        run_dir / "metrics.csv",
        [
            {
                "step": 9,
                "num_supervised_spots": 4,
                "jsd": 0.01,
                "spotwise_cosine": 0.99,
                "mean_celltype_pearson": 0.8,
                "rmse": 0.02,
                "expression_log1p_rmse": 0.5,
                "uncertainty_error_pearson": 0.4,
                "uncertainty_risk_gap": 0.1,
            }
        ],
    )
    _write_json(
        run_dir / "nature_analysis" / "proportion_maps" / "proportion_map_manifest.json",
        {"num_cell_types": 3, "selected_cell_types": ["Tumor", "Stroma"]},
    )
    _write_csv(
        run_dir / "baseline_comparison" / "baseline_comparison.csv",
        [
            {"rank_by_jsd": 1, "method": "WaveST-Gate", "source": "main", "jsd": 0.01, "spotwise_cosine": 0.99},
            {"rank_by_jsd": 2, "method": "Baseline", "source": "baseline", "jsd": 0.2, "spotwise_cosine": 0.7},
        ],
    )
    _write_csv(
        run_dir / "baseline_comparison" / "baseline_split_bootstrap_summary.csv",
        [{"method": "WaveST-Gate", "jsd_mean": 0.011, "jsd_std": 0.001, "spotwise_cosine_mean": 0.99}],
    )
    _write_csv(
        run_dir / "ablations250" / "ablation_summary.csv",
        [
            {"ablation": "full", "jsd": 0.02, "spotwise_cosine": 0.98, "mean_celltype_pearson": 0.8},
            {"ablation": "no_gate", "jsd": 0.05, "spotwise_cosine": 0.9, "mean_celltype_pearson": 0.7},
        ],
    )
    _write_json(run_dir / "nature_analysis" / "reliability_summary.json", {"uncertainty_error_pearson": 0.4, "calibration_bin_pearson": 0.9})
    _write_json(run_dir / "nature_analysis" / "boundary_summary.json", {"boundary_to_interior_jump_ratio": 2.0})
    _write_json(run_dir / "nature_analysis" / "niche_summary.json", {"num_niches": 5})
    _write_csv(external_dir / "pathology_class_summary.csv", [{"agreement_rate": 0.8}])
    _write_csv(
        tmp_path / "external_no_retuning.csv",
        [{"dataset": "d1", "num_spots": 8, "expression_log1p_rmse": 0.6}],
    )
    _write_csv(
        tmp_path / "external_matched_gt.csv",
        [
            {
                "num_spots": 6,
                "no_retuning_model_jsd": 0.3,
                "no_retuning_wavestgate_rank_by_jsd": 2,
                "minimal_retune_test_spots": 2,
                "minimal_retune_test_jsd": 0.04,
                "minimal_retune_wavestgate_rank_by_jsd": 1,
            }
        ],
    )
    _write_csv(tmp_path / "minimal.csv", [{"method": "WaveST-Gate", "num_datasets": 2, "jsd_mean": 0.03, "jsd_std": 0.01}])
    _write_csv(
        run_dir / "robustness" / "robustness_summary.csv",
        [{"scenario": "clean", "level": 0, "jsd": 0.01, "spotwise_cosine": 0.99}],
    )
    _write_csv(run_dir / "patch_size_robustness" / "patch_size_summary.csv", [{"patch_size": 128, "jsd": 0.02, "spotwise_cosine": 0.98}])

    args = build_arg_parser().parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--run-dir",
            str(run_dir),
            "--benchmark-manifest",
            str(bench_dir / "xenium_visium_benchmark_manifest.json"),
            "--spot-qc",
            str(bench_dir / "spot_ground_truth_qc.csv"),
            "--main-metrics",
            str(run_dir / "metrics.csv"),
            "--external-no-retuning",
            str(tmp_path / "external_no_retuning.csv"),
            "--external-matched-gt",
            str(tmp_path / "external_matched_gt.csv"),
            "--minimal-retune-multisample",
            str(tmp_path / "minimal.csv"),
            "--external-pathology-dir",
            str(external_dir),
        ]
    )

    manifest = build_manuscript_tables(args)

    assert Path(manifest["markdown_index"]).exists()
    assert len(manifest["tables"]) == 11
    assert Path(manifest["tables"]["split_sensitivity"]).exists()
    assert Path(manifest["tables"]["benchmark_sensitivity"]).exists()
    assert Path(manifest["tables"]["rep1_retune_budget_curve"]).exists()
    baseline = pd.read_csv(output_dir / "table_3_baseline_comparison.csv")
    assert list(baseline["method"])[:2] == ["WaveST-Gate", "Baseline"]
    ablations = pd.read_csv(output_dir / "table_4_ablation_delta.csv")
    no_gate = ablations.loc[ablations["ablation"] == "no_gate"].iloc[0]
    assert no_gate["delta_vs_full_jsd"] > 0
