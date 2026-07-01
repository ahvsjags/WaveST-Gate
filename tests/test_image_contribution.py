import pandas as pd
import torch

from wavestgate.data.assemble import BatchMetadata
from wavestgate.data.prepare_dataset import save_prepared_dataset
from wavestgate.evaluation.image_contribution import run_image_contribution_analysis
from wavestgate.models.types import WaveSTGateBatch


def test_image_contribution_analysis_writes_summary(tmp_path):
    spot_ids = ["s1", "s2", "s3", "s4"]
    image_patches = torch.zeros(4, 3, 6, 6)
    image_patches[2] = torch.rand(3, 6, 6)
    image_patches[3] = torch.rand(3, 6, 6) * 2.0
    batch = WaveSTGateBatch(
        image_patches=image_patches,
        st_expression=torch.ones(4, 3),
        reference_prototypes=torch.ones(2, 3),
        proportion_gt=torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
            dtype=torch.float32,
        ),
    )
    metadata = BatchMetadata(spot_ids=spot_ids, gene_names=["G1", "G2", "G3"], cell_types=["A", "B"])
    prepared_path = tmp_path / "prepared.pt"
    save_prepared_dataset(prepared_path, batch, metadata)

    image_run = tmp_path / "image"
    no_image_run = tmp_path / "no_image"
    image_run.mkdir()
    no_image_run.mkdir()
    image_pred = pd.DataFrame(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
        index=spot_ids,
        columns=["A", "B"],
    )
    no_image_pred = pd.DataFrame(
        [[0.7, 0.3], [0.6, 0.4], [0.4, 0.6], [0.3, 0.7]],
        index=spot_ids,
        columns=["A", "B"],
    )
    gates = pd.DataFrame(
        [[0.05, 0.94, 0.01], [0.08, 0.91, 0.01], [0.15, 0.84, 0.01], [0.20, 0.79, 0.01]],
        index=spot_ids,
        columns=["image", "expression", "reference"],
    )
    image_pred.to_csv(image_run / "predicted_proportions.csv", index_label="spot_id")
    no_image_pred.to_csv(no_image_run / "predicted_proportions.csv", index_label="spot_id")
    gates.to_csv(image_run / "gate_weights.csv", index_label="spot_id")
    gates.to_csv(image_run / "raw_gate_weights.csv", index_label="spot_id")

    summary = run_image_contribution_analysis(
        prepared_path=prepared_path,
        image_run_dir=image_run,
        no_image_run_dir=no_image_run,
        output_dir=tmp_path / "analysis",
    )

    assert summary["num_supervised_spots"] == 4
    assert summary["mean_paired_jsd_improvement_noimage_minus_imagegate"] > 0
    assert (tmp_path / "analysis" / "image_contribution_summary.json").exists()
    assert (tmp_path / "analysis" / "image_contribution_per_spot.csv").exists()
    assert (tmp_path / "analysis" / "image_contribution_texture_groups.csv").exists()
    assert (tmp_path / "analysis" / "imagegate_run_comparison.csv").exists()
