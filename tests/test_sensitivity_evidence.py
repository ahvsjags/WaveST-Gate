import json
from pathlib import Path

import pandas as pd
import torch

from wavestgate.data.assemble import BatchMetadata
from wavestgate.data.prepare_dataset import save_prepared_dataset
from wavestgate.evaluation.benchmark_sensitivity import run_benchmark_sensitivity
from wavestgate.evaluation.split_sensitivity import run_split_sensitivity
from wavestgate.models.types import WaveSTGateBatch


def _write_demo_prepared(tmp_path: Path) -> tuple[Path, list[str]]:
    spot_ids = ["s1", "s2", "s3", "s4", "s5"]
    batch = WaveSTGateBatch(
        image_patches=torch.zeros(5, 3, 8, 8),
        st_expression=torch.ones(5, 3),
        reference_prototypes=torch.ones(2, 3),
        proportion_gt=torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9], [0.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    metadata = BatchMetadata(spot_ids=spot_ids, gene_names=["G1", "G2", "G3"], cell_types=["A", "B"])
    prepared = tmp_path / "prepared.pt"
    save_prepared_dataset(prepared, batch, metadata)
    return prepared, spot_ids


def test_split_sensitivity_writes_gt_stratified_outputs(tmp_path):
    prepared, spot_ids = _write_demo_prepared(tmp_path)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        [[0.9, 0.1], [0.7, 0.3], [0.3, 0.7], [0.1, 0.9], [0.5, 0.5]],
        index=spot_ids,
        columns=["A", "B"],
    ).to_csv(predictions, index_label="spot_id")
    splits = tmp_path / "splits.csv"
    pd.DataFrame({"spot_id": spot_ids, "split": ["train", "train", "test", "test", "val"]}).to_csv(splits, index=False)
    qc = tmp_path / "qc.csv"
    pd.DataFrame(
        {
            "spot_id": spot_ids,
            "has_xenium_ground_truth": [True, True, True, True, False],
            "xenium_cell_count": [10, 9, 8, 7, 0],
            "dominant_cell_type": ["A", "A", "B", "B", ""],
        }
    ).to_csv(qc, index=False)

    outputs = run_split_sensitivity(
        prepared,
        predictions,
        splits,
        qc,
        tmp_path / "split_analysis",
        seeds=[1, 2],
        val_fraction=0.25,
        test_fraction=0.25,
    )

    assert Path(outputs["manifest_path"]).exists()
    metrics = pd.read_csv(outputs["metrics_path"])
    assert {"train", "val", "test"}.issubset(set(metrics["split"]))
    assert metrics["num_supervised_spots"].max() > 0


def test_benchmark_sensitivity_writes_radius_cell_count_outputs(tmp_path):
    prepared, spot_ids = _write_demo_prepared(tmp_path)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        [[0.9, 0.1], [0.7, 0.3], [0.3, 0.7], [0.1, 0.9], [0.5, 0.5]],
        index=spot_ids,
        columns=["A", "B"],
    ).to_csv(predictions, index_label="spot_id")
    cells = tmp_path / "cells.csv"
    pd.DataFrame(
        {
            "cell_type": ["A", "A", "B", "B"],
            "x": [0.0, 1.0, 10.0, 11.0],
            "y": [0.0, 1.0, 10.0, 11.0],
        }
    ).to_csv(cells, index=False)
    spots = tmp_path / "spots.csv"
    pd.DataFrame({"spot_id": spot_ids, "x": [0.0, 1.0, 10.0, 11.0, 30.0], "y": [0.0, 1.0, 10.0, 11.0, 30.0]}).to_csv(
        spots,
        index=False,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cells_path": str(cells),
                "spots_path": str(spots),
                "columns": {
                    "cell_type_col": "cell_type",
                    "cell_x_col": "x",
                    "cell_y_col": "y",
                    "spot_id_col": "spot_id",
                    "spot_x_col": "x",
                    "spot_y_col": "y",
                },
            }
        ),
        encoding="utf-8",
    )

    outputs = run_benchmark_sensitivity(
        prepared,
        predictions,
        manifest,
        tmp_path / "benchmark_analysis",
        radii=[2.0, 15.0],
        min_cell_counts=[1, 2],
    )

    assert Path(outputs["manifest_path"]).exists()
    metrics = pd.read_csv(outputs["metrics_path"])
    assert set(metrics["radius"]) == {2.0, 15.0}
    assert metrics["num_spots_passing_threshold"].max() > 0
