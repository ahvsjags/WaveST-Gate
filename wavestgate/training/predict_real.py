"""Run WaveST-Gate inference on a prepared real-data batch."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.models.types import WaveSTGateConfig
from wavestgate.models.wavestgate import WaveSTGate


def _load_model(checkpoint_path: str | Path, device: torch.device | str) -> WaveSTGate:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain config and model_state_dict")
    config = WaveSTGateConfig.from_dict(checkpoint["config"])
    model = WaveSTGate(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_real(
    checkpoint_path: str | Path,
    prepared_path: str | Path,
    proportions_path: str | Path,
    gates_path: str | Path | None = None,
    uncertainty_path: str | Path | None = None,
    modality_reliability_path: str | Path | None = None,
    reconstructed_expression_path: str | Path | None = None,
    niche_logits_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, str]:
    """Write prediction tables for a prepared dataset."""

    run_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    batch, metadata, _ = load_prepared_dataset(prepared_path, device=run_device)
    model = _load_model(checkpoint_path, run_device)
    if model.config.num_genes != batch.st_expression.size(1):
        raise ValueError("Checkpoint gene dimension does not match prepared data")
    if model.config.num_cell_types != batch.reference_prototypes.size(0):
        raise ValueError("Checkpoint cell-type dimension does not match prepared data")

    with torch.no_grad():
        output = model(batch)

    written = {}
    proportions_path = Path(proportions_path)
    proportions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        output.proportions.detach().cpu().numpy(),
        index=metadata.spot_ids,
        columns=metadata.cell_types,
    ).to_csv(proportions_path, index_label="spot_id")
    written["proportions_path"] = str(proportions_path)

    if gates_path is not None:
        gates_path = Path(gates_path)
        gates_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            output.gate_weights.detach().cpu().numpy(),
            index=metadata.spot_ids,
            columns=["image", "expression", "reference"],
        ).to_csv(gates_path, index_label="spot_id")
        written["gates_path"] = str(gates_path)

    if uncertainty_path is not None and output.spot_uncertainty is not None:
        uncertainty_path = Path(uncertainty_path)
        uncertainty_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            output.spot_uncertainty.detach().cpu().reshape(-1).numpy(),
            index=metadata.spot_ids,
            columns=["spot_uncertainty"],
        ).to_csv(uncertainty_path, index_label="spot_id")
        written["uncertainty_path"] = str(uncertainty_path)

    if modality_reliability_path is not None and output.modality_reliability is not None:
        modality_reliability_path = Path(modality_reliability_path)
        modality_reliability_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            output.modality_reliability.detach().cpu().numpy(),
            index=metadata.spot_ids,
            columns=["image", "expression", "reference"],
        ).to_csv(modality_reliability_path, index_label="spot_id")
        written["modality_reliability_path"] = str(modality_reliability_path)

    if reconstructed_expression_path is not None:
        reconstructed_expression_path = Path(reconstructed_expression_path)
        reconstructed_expression_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            output.reconstructed_expression.detach().cpu().numpy(),
            index=metadata.spot_ids,
            columns=metadata.gene_names,
        ).to_csv(reconstructed_expression_path, index_label="spot_id")
        written["reconstructed_expression_path"] = str(reconstructed_expression_path)

    if niche_logits_path is not None and output.niche_logits is not None:
        niche_logits_path = Path(niche_logits_path)
        niche_logits_path.parent.mkdir(parents=True, exist_ok=True)
        columns = [f"niche_{idx}" for idx in range(output.niche_logits.size(1))]
        pd.DataFrame(
            output.niche_logits.detach().cpu().numpy(),
            index=metadata.spot_ids,
            columns=columns,
        ).to_csv(niche_logits_path, index_label="spot_id")
        written["niche_logits_path"] = str(niche_logits_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict cell-type proportions from a prepared real-data batch.")
    parser.add_argument("--checkpoint", required=True, help="Path to WaveST-Gate checkpoint.")
    parser.add_argument("--prepared", required=True, help="Path to prepared.pt dataset.")
    parser.add_argument("--proportions", required=True, help="Output CSV for predicted proportions.")
    parser.add_argument("--gates", default=None, help="Optional output CSV for gate weights.")
    parser.add_argument("--uncertainty", default=None, help="Optional output CSV for spot-level uncertainty.")
    parser.add_argument("--modality-reliability", default=None, help="Optional output CSV for modality reliability.")
    parser.add_argument("--reconstructed-expression", default=None, help="Optional output CSV for reconstructed expression.")
    parser.add_argument("--niche-logits", default=None, help="Optional output CSV for niche logits.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device.")
    args = parser.parse_args()
    written = predict_real(
        checkpoint_path=args.checkpoint,
        prepared_path=args.prepared,
        proportions_path=args.proportions,
        gates_path=args.gates,
        uncertainty_path=args.uncertainty,
        modality_reliability_path=args.modality_reliability,
        reconstructed_expression_path=args.reconstructed_expression,
        niche_logits_path=args.niche_logits,
        device=args.device,
    )
    for key, value in written.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
