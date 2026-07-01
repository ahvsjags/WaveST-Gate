import torch

from wavestgate import WaveSTGate, WaveSTGateConfig
from wavestgate.data.synthetic import make_synthetic_batch
from wavestgate.interpretation.niche_analysis import summarize_gate_reliability_by_niche, summarize_niche_composition
from wavestgate.interpretation.visualize_gates import reliability_uncertainty_table
from wavestgate.training.losses import compute_losses, image_gate_participation_loss


def test_wavestgate_forward_contract():
    config = WaveSTGateConfig(
        num_genes=24,
        num_cell_types=5,
        latent_dim=16,
        hidden_dim=32,
        patch_size=33,
        niche_classes=2,
        loss_expr_weight=1.0,
        loss_prop_weight=0.5,
        loss_sparse_weight=0.01,
        loss_uncertainty_weight=0.01,
        loss_boundary_weight=0.01,
        loss_contrast_weight=0.01,
    )
    model = WaveSTGate(config)
    batch = make_synthetic_batch(config, batch_size=4, seed=11)
    output = model(batch)

    assert output.proportions.shape == (4, 5)
    assert output.reconstructed_expression.shape == (4, 24)
    assert output.gate_weights.shape == (4, 3)
    assert output.raw_gate_weights.shape == (4, 3)
    assert output.spot_uncertainty.shape == (4, 1)
    assert output.modality_uncertainty.shape == (4, 3)
    assert output.modality_reliability.shape == (4, 3)
    assert output.agent_attention.shape == (4, 5)
    assert output.niche_logits.shape == (4, 2)
    assert torch.all(output.proportions >= 0)
    assert torch.all(output.spot_uncertainty > 0)
    assert torch.all((output.modality_uncertainty >= 0) & (output.modality_uncertainty <= 1))
    assert torch.allclose(output.proportions.sum(dim=1), torch.ones(4), atol=1e-5)
    assert torch.allclose(output.gate_weights.sum(dim=1), torch.ones(4), atol=1e-5)

    loss, loss_dict = compute_losses(output, batch, config)
    assert torch.isfinite(loss)
    assert set(loss_dict) == {
        "loss_total",
        "loss_expr",
        "loss_prop",
        "loss_sparse",
        "loss_spatial",
        "loss_boundary",
        "loss_contrast",
        "loss_uncertainty",
        "loss_niche",
        "loss_image_gate",
    }
    loss.backward()


def test_image_gate_participation_loss_hinges_on_target():
    config = WaveSTGateConfig(
        num_genes=12,
        num_cell_types=3,
        latent_dim=8,
        hidden_dim=16,
        patch_size=17,
        target_image_gate=0.05,
        target_boundary_image_gate=0.05,
        target_raw_image_gate=0.08,
    )
    model = WaveSTGate(config)
    batch = make_synthetic_batch(config, batch_size=4, seed=19)
    output = model(batch)

    output.gate_weights = torch.tensor(
        [[0.01, 0.98, 0.01], [0.02, 0.97, 0.01], [0.10, 0.89, 0.01], [0.15, 0.84, 0.01]],
        dtype=output.gate_weights.dtype,
    )
    output.raw_gate_weights = torch.tensor(
        [[0.03, 0.95, 0.02], [0.05, 0.92, 0.03], [0.12, 0.85, 0.03], [0.14, 0.83, 0.03]],
        dtype=output.raw_gate_weights.dtype,
    )
    active = image_gate_participation_loss(output, batch, config)
    assert active > 0

    output.gate_weights = torch.tensor(
        [[0.95, 0.04, 0.01], [0.95, 0.04, 0.01], [0.95, 0.04, 0.01], [0.95, 0.04, 0.01]],
        dtype=output.gate_weights.dtype,
    )
    output.raw_gate_weights = output.gate_weights
    inactive = image_gate_participation_loss(output, batch, config)
    assert torch.isclose(inactive, torch.tensor(0.0, dtype=inactive.dtype))


def test_reliability_and_niche_interpretation_tables():
    proportions = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    niche_logits = torch.tensor([[3.0, 0.0], [0.0, 2.0], [0.2, 1.2]])
    gate_weights = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.2, 0.2, 0.6]])
    spot_uncertainty = torch.tensor([[0.1], [0.4], [0.2]])
    modality_uncertainty = 1.0 - gate_weights

    niche_rows = summarize_niche_composition(proportions, niche_logits, cell_types=["tumor", "immune"])
    gate_rows = summarize_gate_reliability_by_niche(gate_weights, niche_logits)
    uncertainty_rows = reliability_uncertainty_table(spot_uncertainty, modality_uncertainty)

    assert niche_rows[0]["dominant_cell_type"] == "tumor"
    assert niche_rows[1]["dominant_cell_type"] == "immune"
    assert gate_rows[1]["num_spots"] == 2
    assert "image_uncertainty" in uncertainty_rows[0]
