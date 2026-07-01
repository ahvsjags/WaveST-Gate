import json
from pathlib import Path

import pandas as pd

from wavestgate.evaluation.leakage_fairness_audit import build_leakage_fairness_audit


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_leakage_fairness_audit_builds_reviewer_outputs(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    run = tmp_path / "run"
    split_guarded = tmp_path / "split_guarded"
    output = tmp_path / "audit"
    primary_config = tmp_path / "primary.json"
    guarded_config = tmp_path / "guarded.json"

    gt_rows = [
        {"spot_id": "s1", "A": 1.0, "B": 0.0},
        {"spot_id": "s2", "A": 0.0, "B": 0.0},
        {"spot_id": "s3", "A": 0.0, "B": 1.0},
        {"spot_id": "s4", "A": 0.0, "B": 0.0},
        {"spot_id": "s5", "A": 0.5, "B": 0.5},
        {"spot_id": "s6", "A": 0.0, "B": 0.0},
    ]
    _write_csv(benchmark / "xenium_cell_proportions.csv", gt_rows)
    _write_csv(
        benchmark / "spot_splits.csv",
        [
            {"spot_id": "s1", "split": "train"},
            {"spot_id": "s2", "split": "train"},
            {"spot_id": "s3", "split": "test"},
            {"spot_id": "s4", "split": "test"},
            {"spot_id": "s5", "split": "val"},
            {"spot_id": "s6", "split": "val"},
        ],
    )
    _write_csv(
        benchmark / "spot_ground_truth_qc.csv",
        [
            {"spot_id": "s1", "has_xenium_ground_truth": True},
            {"spot_id": "s2", "has_xenium_ground_truth": False},
            {"spot_id": "s3", "has_xenium_ground_truth": True},
            {"spot_id": "s4", "has_xenium_ground_truth": False},
            {"spot_id": "s5", "has_xenium_ground_truth": True},
            {"spot_id": "s6", "has_xenium_ground_truth": False},
        ],
    )
    _write_csv(run / "predicted_proportions.csv", gt_rows)
    _write_csv(run / "baseline" / "baseline_predictions.csv", gt_rows)
    _write_csv(
        run / "baseline_comparison" / "baseline_comparison.csv",
        [
            {
                "method": "WaveST-Gate",
                "jsd": 0.0,
                "num_supervised_spots": 3,
                "predictions_path": str(run / "predicted_proportions.csv"),
            },
            {
                "method": "Baseline",
                "jsd": 0.0,
                "num_supervised_spots": 3,
                "predictions_path": str(run / "baseline" / "baseline_predictions.csv"),
            },
        ],
    )
    _write_json(primary_config, {"data": {"prepared_path": "prepared.pt"}, "training": {}})
    _write_json(
        guarded_config,
        {
            "data": {"prepared_path": "prepared.pt", "split_path": str(benchmark / "spot_splits.csv"), "train_splits": ["train"]},
            "training": {},
            "evaluation": {"eval_splits": ["test"]},
        },
    )
    _write_csv(
        split_guarded / "metrics.csv",
        [{"jsd": 0.1, "num_train_spots": 2, "num_eval_spots": 2, "num_supervised_spots": 1}],
    )

    summary = build_leakage_fairness_audit(
        benchmark_dir=benchmark,
        run_dir=run,
        output_dir=output,
        primary_config=primary_config,
        split_guarded_config=guarded_config,
        split_guarded_run_dir=split_guarded,
        num_permutations=5,
        seed=7,
    )

    assert summary["overall_status"] == "complete_with_primary_benchmark_caveat"
    assert summary["baseline_audit"]["num_methods_fail"] == 0
    assert summary["permutation_control"]["observed_jsd"] == 0.0
    assert Path(summary["outputs"]["summary_markdown"]).exists()
    checklist = pd.read_csv(summary["outputs"]["checklist_csv"])
    assert "primary_config_split_guard" in set(checklist["check"])
    assert checklist.loc[checklist["check"].eq("primary_config_split_guard"), "status"].item() == "warn"
