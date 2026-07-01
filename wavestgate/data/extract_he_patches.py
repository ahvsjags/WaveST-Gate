"""H&E image patch extraction for spot-centered coordinates."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def crop_center_with_padding(
    image: Image.Image,
    center_x: float,
    center_y: float,
    patch_size: int,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Crop a square patch centered at `(center_x, center_y)` with padding."""

    half = patch_size // 2
    left = int(round(center_x)) - half
    top = int(round(center_y)) - half
    right = left + patch_size
    bottom = top + patch_size

    patch = Image.new("RGB", (patch_size, patch_size), fill)
    crop_box = (
        max(left, 0),
        max(top, 0),
        min(right, image.width),
        min(bottom, image.height),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return patch

    cropped = image.crop(crop_box).convert("RGB")
    paste_xy = (crop_box[0] - left, crop_box[1] - top)
    patch.paste(cropped, paste_xy)
    return patch


def patches_to_tensor(patches: Iterable[Image.Image]) -> torch.Tensor:
    """Convert PIL patches to `[B, 3, H, W]` float tensor."""

    tensors = []
    for patch in patches:
        array = np.asarray(patch.convert("RGB"), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    if not tensors:
        raise ValueError("At least one patch is required")
    return torch.stack(tensors, dim=0)


def extract_spot_patches(
    image_path: str | Path,
    coords: pd.DataFrame | torch.Tensor | np.ndarray,
    patch_size: int,
    spot_ids: list[str] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    output_dir: str | Path | None = None,
) -> torch.Tensor | list[Path]:
    """Extract spot-centered H&E patches.

    If `output_dir` is provided, PNG files are written and paths are returned.
    Otherwise a tensor batch is returned.
    """

    image = Image.open(image_path)
    if isinstance(coords, pd.DataFrame):
        if x_col not in coords.columns or y_col not in coords.columns:
            raise ValueError(f"coords DataFrame must contain {x_col!r} and {y_col!r}")
        xy = coords[[x_col, y_col]].values
        ids = spot_ids or [str(idx) for idx in coords.index]
    else:
        xy = np.asarray(coords, dtype=np.float32)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("coords array must have shape [N, 2]")
        ids = spot_ids or [f"spot_{idx}" for idx in range(len(xy))]

    patches = [crop_center_with_padding(image, x, y, patch_size) for x, y in xy]
    if output_dir is None:
        return patches_to_tensor(patches)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for spot_id, patch in zip(ids, patches):
        path = out_dir / f"{spot_id}.png"
        patch.save(path)
        paths.append(path)
    return paths
