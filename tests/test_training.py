import json

from wavestgate.training.train import train_from_config


def test_toy_training_writes_outputs(tmp_path):
    config = {
        "model": {
            "num_genes": 16,
            "num_cell_types": 4,
            "latent_dim": 16,
            "hidden_dim": 32,
            "patch_size": 32,
            "loss_expr_weight": 1.0,
            "loss_prop_weight": 0.5,
            "loss_sparse_weight": 0.01,
            "loss_contrast_weight": 0.01,
            "loss_uncertainty_weight": 0.01,
            "loss_boundary_weight": 0.01,
        },
        "training": {
            "steps": 2,
            "batch_size": 3,
            "learning_rate": 0.001,
            "seed": 3,
            "device": "cpu",
            "checkpoint_path": str(tmp_path / "checkpoint.pt"),
            "metrics_path": str(tmp_path / "metrics.csv"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    metrics = train_from_config(config_path)
    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert "loss_total" in metrics
    assert "spotwise_cosine" in metrics
    assert "uncertainty_error_pearson" in metrics
