"""Loss functions for WaveST-Gate."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wavestgate.models.types import WaveSTGateBatch, WaveSTGateConfig, WaveSTGateOutput


def expression_reconstruction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.log1p(predicted.clamp_min(0.0)), torch.log1p(target.clamp_min(0.0)))


def proportion_supervised_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    mask = target.sum(dim=1) > eps
    if not mask.any():
        return predicted.new_tensor(0.0)
    target = target[mask]
    predicted = predicted[mask]
    target = target / target.sum(dim=1, keepdim=True).clamp_min(eps)
    predicted = predicted.clamp_min(eps)
    kl = F.kl_div(predicted.log(), target, reduction="batchmean")
    cosine = 1.0 - F.cosine_similarity(predicted, target, dim=1).mean()
    return kl + cosine


def entropy_sparsity_loss(proportions: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    p = proportions.clamp_min(eps)
    return -(p * p.log()).sum(dim=1).mean()


def spatial_smoothness_loss(
    proportions: torch.Tensor,
    edge_index: torch.Tensor | None,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return proportions.new_tensor(0.0)
    src, dst = edge_index.long()
    distances = (proportions[src] - proportions[dst]).pow(2).sum(dim=1)
    if edge_weight is not None:
        weights = edge_weight.to(device=proportions.device, dtype=proportions.dtype)
        distances = distances * weights
        return distances.sum() / weights.sum().clamp_min(1e-8)
    return distances.mean()


def morphology_edge_weights(
    image_feature: torch.Tensor,
    edge_index: torch.Tensor | None,
    edge_weight: torch.Tensor | None = None,
    temperature: float = 0.5,
) -> torch.Tensor | None:
    """Return higher edge weights for morphologically similar neighboring spots."""

    if edge_index is None or edge_index.numel() == 0:
        return None
    src, dst = edge_index.long()
    cosine = F.cosine_similarity(image_feature[src], image_feature[dst], dim=1)
    similarity = ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)
    temperature = max(float(temperature), 1e-6)
    weights = torch.exp(-(1.0 - similarity) / temperature)
    if edge_weight is not None:
        weights = weights * edge_weight.to(device=image_feature.device, dtype=image_feature.dtype)
    return weights


def morphology_boundary_scores(
    image_feature: torch.Tensor,
    edge_index: torch.Tensor | None,
) -> torch.Tensor:
    """Estimate per-spot morphology boundary strength from neighboring image features."""

    scores = image_feature.new_zeros(image_feature.size(0))
    if edge_index is None or edge_index.numel() == 0:
        return scores
    src, dst = edge_index.long()
    cosine = F.cosine_similarity(image_feature[src], image_feature[dst], dim=1)
    dissimilarity = (1.0 - ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)).detach()
    counts = image_feature.new_zeros(image_feature.size(0))
    scores.index_add_(0, src, dissimilarity)
    scores.index_add_(0, dst, dissimilarity)
    counts.index_add_(0, src, torch.ones_like(dissimilarity))
    counts.index_add_(0, dst, torch.ones_like(dissimilarity))
    scores = scores / counts.clamp_min(1.0)
    return scores / scores.max().clamp_min(1e-8)


def boundary_preserving_spatial_loss(
    proportions: torch.Tensor,
    image_feature: torch.Tensor,
    edge_index: torch.Tensor | None,
    edge_weight: torch.Tensor | None = None,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Smooth proportions across similar morphology while preserving tissue boundaries."""

    weights = morphology_edge_weights(image_feature, edge_index, edge_weight, temperature)
    if weights is None:
        return proportions.new_tensor(0.0)
    src, dst = edge_index.long()
    distances = (proportions[src] - proportions[dst]).pow(2).sum(dim=1)
    weighted = distances * weights
    return weighted.sum() / weights.sum().clamp_min(1e-8)


def contrastive_alignment_loss(image_feature: torch.Tensor, expression_feature: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if image_feature.size(0) <= 1:
        return image_feature.new_tensor(0.0)
    image = F.normalize(image_feature, dim=1)
    expression = F.normalize(expression_feature, dim=1)
    logits = image @ expression.transpose(0, 1) / temperature
    labels = torch.arange(image.size(0), device=image.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def uncertainty_calibration_loss(
    spot_uncertainty: torch.Tensor | None,
    predicted: torch.Tensor,
    target: torch.Tensor | None,
) -> torch.Tensor:
    """Train uncertainty to track per-spot deconvolution error when supervision exists."""

    if spot_uncertainty is None or target is None:
        return predicted.new_tensor(0.0)
    eps = 1e-8
    mask = target.sum(dim=1) > eps
    if not mask.any():
        return predicted.new_tensor(0.0)
    target = target[mask]
    predicted = predicted[mask]
    spot_uncertainty = spot_uncertainty.reshape(-1)[mask]
    target = target / target.sum(dim=1, keepdim=True).clamp_min(eps)
    predicted = predicted / predicted.sum(dim=1, keepdim=True).clamp_min(eps)
    per_spot_error = (predicted - target).pow(2).mean(dim=1).detach()
    uncertainty = spot_uncertainty.clamp_min(eps)
    return (per_spot_error / uncertainty + torch.log1p(uncertainty)).mean()


def image_gate_participation_loss(
    output: WaveSTGateOutput,
    batch: WaveSTGateBatch,
    config: WaveSTGateConfig,
) -> torch.Tensor:
    """Prevent the calibrated reliability gate from collapsing the image modality."""

    if output.gate_weights.size(1) < 3:
        return output.gate_weights.new_tensor(0.0)
    image_gate = output.gate_weights[:, 0]
    target = image_gate.new_full(image_gate.shape, max(float(config.target_image_gate), 0.0))
    if config.target_boundary_image_gate > 0:
        boundary_feature = output.morphology_feature if output.morphology_feature is not None else output.image_feature
        boundary_score = morphology_boundary_scores(boundary_feature, batch.edge_index).to(image_gate.device)
        target = target + float(config.target_boundary_image_gate) * boundary_score
    final_loss = F.relu(target.clamp(max=0.95) - image_gate).mean()
    if output.raw_gate_weights is None or config.target_raw_image_gate <= 0:
        return final_loss
    raw_target = image_gate.new_full(image_gate.shape, max(float(config.target_raw_image_gate), 0.0))
    raw_loss = F.relu(raw_target.clamp(max=0.95) - output.raw_gate_weights[:, 0]).mean()
    return final_loss + float(config.raw_image_gate_loss_scale) * raw_loss


def niche_supervised_loss(niche_logits: torch.Tensor | None, target: torch.Tensor | None) -> torch.Tensor:
    """Cross-entropy loss for optional biological niche labels."""

    if niche_logits is None or target is None:
        reference = niche_logits if niche_logits is not None else target
        return reference.new_tensor(0.0)
    if target.ndim == 2:
        target = target.argmax(dim=1)
    return F.cross_entropy(niche_logits, target.long())


def compute_losses(
    output: WaveSTGateOutput,
    batch: WaveSTGateBatch,
    config: WaveSTGateConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    expr = expression_reconstruction_loss(output.reconstructed_expression, batch.st_expression)
    prop = (
        proportion_supervised_loss(output.proportions, batch.proportion_gt)
        if batch.proportion_gt is not None
        else output.proportions.new_tensor(0.0)
    )
    sparse = entropy_sparsity_loss(output.proportions)
    spatial = spatial_smoothness_loss(output.proportions, batch.edge_index, batch.edge_weight)
    boundary_feature = output.morphology_feature if output.morphology_feature is not None else output.image_feature
    boundary = boundary_preserving_spatial_loss(
        output.proportions,
        boundary_feature,
        batch.edge_index,
        batch.edge_weight,
        temperature=config.boundary_temperature,
    )
    contrast = contrastive_alignment_loss(output.image_feature, output.expression_feature)
    uncertainty = uncertainty_calibration_loss(output.spot_uncertainty, output.proportions, batch.proportion_gt)
    image_gate = image_gate_participation_loss(output, batch, config)
    niche = (
        niche_supervised_loss(output.niche_logits, batch.niche_gt)
        if output.niche_logits is not None or batch.niche_gt is not None
        else output.proportions.new_tensor(0.0)
    )

    total = (
        config.loss_expr_weight * expr
        + config.loss_prop_weight * prop
        + config.loss_sparse_weight * sparse
        + config.loss_spatial_weight * spatial
        + config.loss_contrast_weight * contrast
        + config.loss_uncertainty_weight * uncertainty
        + config.loss_boundary_weight * boundary
        + config.loss_niche_weight * niche
        + config.loss_image_gate_weight * image_gate
    )
    return total, {
        "loss_total": total.detach(),
        "loss_expr": expr.detach(),
        "loss_prop": prop.detach(),
        "loss_sparse": sparse.detach(),
        "loss_spatial": spatial.detach(),
        "loss_boundary": boundary.detach(),
        "loss_contrast": contrast.detach(),
        "loss_uncertainty": uncertainty.detach(),
        "loss_niche": niche.detach(),
        "loss_image_gate": image_gate.detach(),
    }
