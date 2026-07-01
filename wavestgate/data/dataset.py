"""Small tensor dataset helpers."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SpotTensorDataset(Dataset):
    """Dataset for preloaded spot tensors.

    The shared reference prototypes are intentionally kept outside individual
    samples so the training loop can batch spots and attach the same reference.
    """

    def __init__(
        self,
        image_patches: torch.Tensor,
        st_expression: torch.Tensor,
        proportions: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> None:
        if image_patches.size(0) != st_expression.size(0):
            raise ValueError("image_patches and st_expression must contain the same number of spots")
        self.image_patches = image_patches
        self.st_expression = st_expression
        self.proportions = proportions
        self.coords = coords
        self.edge_weight = edge_weight

    def __len__(self) -> int:
        return self.image_patches.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "image_patches": self.image_patches[index],
            "st_expression": self.st_expression[index],
        }
        if self.proportions is not None:
            item["proportion_gt"] = self.proportions[index]
        if self.coords is not None:
            item["coords"] = self.coords[index]
        return item
