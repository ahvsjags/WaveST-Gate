"""Evaluation metrics for deconvolution outputs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def filter_valid_targets(
    predicted: torch.Tensor,
    target: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Keep spots with non-empty ground-truth proportions."""

    mask = target.sum(dim=1) > eps
    if mask.any():
        predicted = predicted[mask]
        target = target[mask]
        uncertainty = uncertainty.reshape(-1)[mask] if uncertainty is not None else None
    else:
        predicted = predicted[:0]
        target = target[:0]
        uncertainty = uncertainty.reshape(-1)[:0] if uncertainty is not None else None
    return predicted, target, uncertainty


def spotwise_cosine(predicted: torch.Tensor, target: torch.Tensor) -> float:
    return F.cosine_similarity(predicted, target, dim=1).mean().item()


def rmse(predicted: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(F.mse_loss(predicted, target)).item()


def mean_celltype_pearson(predicted: torch.Tensor, target: torch.Tensor) -> float:
    pred = predicted.detach().cpu().numpy()
    true = target.detach().cpu().numpy()
    values = []
    for idx in range(pred.shape[1]):
        if np.std(pred[:, idx]) < 1e-8 or np.std(true[:, idx]) < 1e-8:
            continue
        values.append(float(np.corrcoef(pred[:, idx], true[:, idx])[0, 1]))
    return float(np.mean(values)) if values else 0.0


def jensen_shannon_divergence(predicted: torch.Tensor, target: torch.Tensor) -> float:
    eps = 1e-8
    p = predicted.clamp_min(eps)
    p = p / p.sum(dim=1, keepdim=True)
    q = target.clamp_min(eps)
    q = q / q.sum(dim=1, keepdim=True)
    m = 0.5 * (p + q)
    jsd = 0.5 * ((p * (p / m).log()).sum(dim=1) + (q * (q / m).log()).sum(dim=1))
    return jsd.mean().item()


def per_spot_jensen_shannon_divergence(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return one Jensen-Shannon divergence value per spot."""

    eps = 1e-8
    p = predicted.clamp_min(eps)
    p = p / p.sum(dim=1, keepdim=True)
    q = target.clamp_min(eps)
    q = q / q.sum(dim=1, keepdim=True)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p / m).log()).sum(dim=1) + (q * (q / m).log()).sum(dim=1))


def uncertainty_calibration_metrics(
    uncertainty: torch.Tensor | None,
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Summarize whether high uncertainty corresponds to high deconvolution error."""

    if uncertainty is None:
        return {}
    values = uncertainty.detach().reshape(-1).cpu().numpy().astype(float)
    errors = per_spot_jensen_shannon_divergence(predicted, target).detach().cpu().numpy().astype(float)
    if len(values) != len(errors) or len(values) == 0:
        return {}
    if len(values) > 1 and np.std(values) >= 1e-8 and np.std(errors) >= 1e-8:
        corr = float(np.corrcoef(values, errors)[0, 1])
    else:
        corr = 0.0
    quantile = max(int(np.ceil(len(values) * 0.25)), 1)
    order = np.argsort(values)
    low_error = float(np.mean(errors[order[:quantile]]))
    high_error = float(np.mean(errors[order[-quantile:]]))
    return {
        "uncertainty_error_pearson": corr,
        "uncertainty_low_quartile_jsd": low_error,
        "uncertainty_high_quartile_jsd": high_error,
        "uncertainty_risk_gap": high_error - low_error,
        "mean_spot_uncertainty": float(np.mean(values)),
    }


def summarize_proportion_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
) -> dict[str, float]:
    predicted, target, uncertainty = filter_valid_targets(predicted, target, uncertainty)
    if predicted.numel() == 0:
        return {
            "spotwise_cosine": 0.0,
            "mean_celltype_pearson": 0.0,
            "rmse": 0.0,
            "jsd": 0.0,
            "num_supervised_spots": 0.0,
        }
    metrics = {
        "spotwise_cosine": spotwise_cosine(predicted, target),
        "mean_celltype_pearson": mean_celltype_pearson(predicted, target),
        "rmse": rmse(predicted, target),
        "jsd": jensen_shannon_divergence(predicted, target),
        "num_supervised_spots": float(predicted.size(0)),
    }
    metrics.update(uncertainty_calibration_metrics(uncertainty, predicted, target))
    return metrics
