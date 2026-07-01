"""Lightweight CSV/NPY/PIL data interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wavestgate.models.types import WaveSTGateBatch


def load_array(path: str | Path) -> torch.Tensor:
    """Load a numeric array from `.npy`, `.csv`, or `.tsv`."""

    path = Path(path)
    if path.suffix == ".npy":
        array = np.load(path)
    elif path.suffix in {".csv", ".txt"}:
        array = pd.read_csv(path, index_col=None).values
    elif path.suffix == ".tsv":
        array = pd.read_csv(path, sep="\t", index_col=None).values
    else:
        raise ValueError(f"Unsupported array format: {path.suffix}")
    return torch.as_tensor(array, dtype=torch.float32)


def load_image_patches(paths: Iterable[str | Path], image_size: Optional[int] = None) -> torch.Tensor:
    """Load RGB image patches into `[B, 3, H, W]` float tensors."""

    tensors = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if image_size is not None:
            image = image.resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    if not tensors:
        raise ValueError("At least one image path is required")
    return torch.stack(tensors, dim=0)


def load_batch_from_files(
    expression_path: str | Path,
    reference_path: str | Path,
    image_paths: Iterable[str | Path],
    proportion_path: str | Path | None = None,
    coords_path: str | Path | None = None,
    image_size: int | None = None,
) -> WaveSTGateBatch:
    """Construct a `WaveSTGateBatch` from simple local files."""

    expression = load_array(expression_path)
    reference = load_array(reference_path)
    patches = load_image_patches(image_paths, image_size=image_size)
    proportions = load_array(proportion_path) if proportion_path is not None else None
    coords = load_array(coords_path) if coords_path is not None else None
    return WaveSTGateBatch(
        image_patches=patches,
        st_expression=expression,
        reference_prototypes=reference,
        coords=coords,
        proportion_gt=proportions,
    )
