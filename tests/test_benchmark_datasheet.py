import json
from pathlib import Path

import pandas as pd

from wavestgate.evaluation.benchmark_datasheet import build_benchmark_datasheet


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_benchmark_datasheet_integrity_and_gt_only_dominant_counts(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    counts_path = benchmark_dir / "xenium_cell_counts.csv"
    proportions_path = benchmark_dir / "xenium_cell_proportions.csv"
    qc_path = benchmark_dir / "spot_ground_truth_qc.csv"
    splits_path = benchmark_dir / "spot_splits.csv"
    manifest_path = benchmark_dir / "xenium_visium_benchmark_manifest.json"
    protocol_path = tmp_path / "docs" / "xenium_to_visium_benchmark_protocol.md"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text("protocol", encoding="utf-8")

    _write_csv(
        counts_path,
        [
            {"spot_id": "s1", "A": 2, "B": 0},
            {"spot_id": "s2", "A": 1, "B": 3},
            {"spot_id": "s3", "A": 0, "B": 0},
        ],
    )
    _write_csv(
        proportions_path,
        [
            {"spot_id": "s1", "A": 1.0, "B": 0.0},
            {"spot_id": "s2", "A": 0.25, "B": 0.75},
            {"spot_id": "s3", "A": 0.0, "B": 0.0},
        ],
    )
    _write_csv(
        qc_path,
        [
            {"spot_id": "s1", "xenium_cell_count": 2, "ground_truth_entropy": 0.0, "dominant_cell_type": "A", "has_xenium_ground_truth": True},
            {"spot_id": "s2", "xenium_cell_count": 4, "ground_truth_entropy": 0.81, "dominant_cell_type": "B", "has_xenium_ground_truth": True},
            {"spot_id": "s3", "xenium_cell_count": 0, "ground_truth_entropy": 0.0, "dominant_cell_type": "A", "has_xenium_ground_truth": False},
        ],
    )
    _write_csv(splits_path, [{"spot_id": "s1", "split": "train"}, {"spot_id": "s2", "split": "val"}, {"spot_id": "s3", "split": "test"}])
    _write_json(
        manifest_path,
        {
            "protocol": "xenium_to_visium_spot_aggregation",
            "num_spots": 3,
            "num_spots_with_ground_truth": 2,
            "num_cells": 6,
            "num_cell_types": 2,
            "spot_radius": 55,
            "artifacts": {
                "counts": str(counts_path),
                "proportions": str(proportions_path),
                "qc": str(qc_path),
                "splits": str(splits_path),
                "manifest": str(manifest_path),
            },
        },
    )

    payload = build_benchmark_datasheet(benchmark_dir=benchmark_dir, output_dir=tmp_path / "datasheet", protocol_path=protocol_path)

    assert payload["status"] == "complete"
    assert payload["failed_integrity_checks"] == []
    assert {check["status"] for check in payload["integrity_checks"]} == {"pass"}
    assert payload["qc_summary"]["num_spots_with_ground_truth"] == 2
    assert payload["qc_summary"]["dominant_cell_type_counts_on_gt_spots"] == {"A": 1, "B": 1}
    assert len(payload["column_dictionary"]) >= 8
    assert len(payload["cell_type_summary"]) == 2
    assert Path(payload["outputs"]["json"]).exists()
    assert Path(payload["outputs"]["markdown"]).exists()
