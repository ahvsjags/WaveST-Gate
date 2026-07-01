import pandas as pd
import torch
from PIL import Image

from wavestgate.data import (
    assemble_wavestgate_batch,
    attach_coordinates,
    build_knn_graph,
    build_reference_prototypes,
    count_cells_in_spots,
    extract_spot_patches,
    load_spot_coordinates,
    load_spot_expression,
    proportions_from_counts,
)
from wavestgate.models.types import WaveSTGateConfig
from wavestgate.models.wavestgate import WaveSTGate


def test_spot_and_scrna_preprocessing_assemble_forward(tmp_path):
    expression_path = tmp_path / "spots.csv"
    coords_path = tmp_path / "coords.csv"
    scrna_path = tmp_path / "scrna.csv"

    pd.DataFrame(
        {
            "spot_id": ["s1", "s2", "s3"],
            "G1": [1.0, 0.5, 2.0],
            "G2": [0.0, 1.0, 2.0],
            "G3": [3.0, 1.0, 0.0],
        }
    ).to_csv(expression_path, index=False)
    pd.DataFrame(
        {
            "spot_id": ["s1", "s2", "s3"],
            "x": [4.0, 10.0, 18.0],
            "y": [4.0, 10.0, 18.0],
        }
    ).to_csv(coords_path, index=False)
    pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4"],
            "cell_type": ["T", "T", "B", "B"],
            "G1": [2.0, 4.0, 0.0, 0.0],
            "G2": [0.0, 0.0, 3.0, 5.0],
            "G3": [1.0, 1.0, 2.0, 2.0],
        }
    ).to_csv(scrna_path, index=False)

    spots = load_spot_expression(expression_path)
    coords = load_spot_coordinates(coords_path)
    spots = attach_coordinates(spots, coords)
    reference = build_reference_prototypes(scrna_path, target_genes=spots.gene_names)
    patches = torch.rand(3, 3, 24, 24)
    batch, metadata = assemble_wavestgate_batch(spots, reference, patches, graph_k=1)

    assert metadata.spot_ids == ["s1", "s2", "s3"]
    assert metadata.gene_names == ["G1", "G2", "G3"]
    assert metadata.cell_types == ["B", "T"]
    assert batch.reference_prototypes.shape == (2, 3)
    assert batch.edge_index.shape[0] == 2
    assert batch.edge_weight is not None

    config = WaveSTGateConfig(num_genes=3, num_cell_types=2, latent_dim=16, hidden_dim=32, patch_size=24)
    output = WaveSTGate(config)(batch)
    assert output.proportions.shape == (3, 2)


def test_extract_spot_patches_handles_border_padding(tmp_path):
    image_path = tmp_path / "he.png"
    image = Image.new("RGB", (12, 12), (10, 20, 30))
    image.save(image_path)
    coords = pd.DataFrame({"x": [1.0, 6.0], "y": [1.0, 6.0]}, index=["edge", "center"])

    patches = extract_spot_patches(image_path, coords, patch_size=8)
    assert patches.shape == (2, 3, 8, 8)
    assert float(patches[0, :, 0, 0].mean()) > 0.9

    out_paths = extract_spot_patches(image_path, coords, patch_size=8, output_dir=tmp_path / "patches")
    assert len(out_paths) == 2
    assert out_paths[0].exists()


def test_xenium_to_visium_counts_and_proportions():
    cells = pd.DataFrame(
        {
            "cell_type": ["T", "B", "T", "M"],
            "x": [0.0, 1.0, 10.0, 11.0],
            "y": [0.0, 0.0, 10.0, 10.0],
        }
    )
    spots = pd.DataFrame(
        {
            "spot_id": ["s1", "s2", "s3"],
            "x": [0.0, 10.0, 30.0],
            "y": [0.0, 10.0, 30.0],
        }
    )
    counts = count_cells_in_spots(cells, spots, spot_radius=2.0)
    proportions = proportions_from_counts(counts)

    assert counts.loc["s1", "T"] == 1
    assert counts.loc["s1", "B"] == 1
    assert counts.loc["s2", "T"] == 1
    assert counts.loc["s2", "M"] == 1
    assert proportions.loc["s1"].sum() == 1.0
    assert proportions.loc["s3"].sum() == 0.0


def test_knn_graph_returns_weights():
    edge_index, edge_weight = build_knn_graph(torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]), k=1)
    assert edge_index.shape == (2, 3)
    assert edge_weight.shape == (3,)
    assert torch.all(edge_weight > 0)
