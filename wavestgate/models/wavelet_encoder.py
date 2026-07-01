"""Pure-torch wavelet morphology modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def build_haar_kernels(device=None, dtype=torch.float32) -> torch.Tensor:
    """Return Haar analysis/synthesis filters in `[LL, LH, HL, HH]` order."""

    scale = 1.0 / math.sqrt(2.0)
    low = torch.tensor([scale, scale], dtype=dtype, device=device)
    high = torch.tensor([-scale, scale], dtype=dtype, device=device)
    kernels = torch.stack(
        [
            torch.outer(low, low),
            torch.outer(low, high),
            torch.outer(high, low),
            torch.outer(high, high),
        ],
        dim=0,
    )
    return kernels.unsqueeze(1)


@dataclass
class WaveletBands:
    ll: torch.Tensor
    lh: torch.Tensor
    hl: torch.Tensor
    hh: torch.Tensor
    original_hw: tuple[int, int]


class HaarWaveletTransform(nn.Module):
    """Grouped Haar DWT/IDWT that preserves odd input sizes by padding/cropping."""

    def __init__(self) -> None:
        super().__init__()
        kernels = build_haar_kernels()
        self.register_buffer("analysis", kernels)
        self.register_buffer("synthesis", kernels)

    def dwt(self, x: torch.Tensor) -> WaveletBands:
        bsz, channels, height, width = x.shape
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        weight = self.analysis.to(dtype=x.dtype, device=x.device).repeat(channels, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, groups=channels)
        y = y.view(bsz, channels, 4, y.size(-2), y.size(-1)).contiguous()
        return WaveletBands(
            ll=y[:, :, 0],
            lh=y[:, :, 1],
            hl=y[:, :, 2],
            hh=y[:, :, 3],
            original_hw=(height, width),
        )

    def idwt(self, bands: WaveletBands) -> torch.Tensor:
        bsz, channels, _, _ = bands.ll.shape
        y = torch.stack([bands.ll, bands.lh, bands.hl, bands.hh], dim=2)
        y = y.view(bsz, 4 * channels, bands.ll.size(-2), bands.ll.size(-1))
        weight = self.synthesis.to(dtype=y.dtype, device=y.device).repeat(channels, 1, 1, 1)
        x = F.conv_transpose2d(y, weight, stride=2, groups=channels)
        height, width = bands.original_hw
        return x[:, :, :height, :width]


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, ratio: int = 8) -> None:
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class WaveletGatedCrossBand(nn.Module):
    """Wavelet-gated cross-band block adapted from DW-GCA."""

    def __init__(self, channels: int, se_ratio: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.wavelet = HaarWaveletTransform()
        self.theta = nn.Parameter(torch.zeros(3, channels, 1, 1))
        self.direction_gate = nn.Parameter(torch.zeros(3, channels, 1, 1))
        self.subband_redistribution = nn.Parameter(torch.zeros(channels, 3))
        self.channel_proj = nn.Linear(channels, channels)
        self.se = SqueezeExcitation(channels, se_ratio)
        self.gamma = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    @staticmethod
    def _soft_threshold(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * F.relu(torch.abs(x) - threshold)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bands = self.wavelet.dwt(x)
        high = [bands.lh, bands.hl, bands.hh]

        thresholds = torch.sigmoid(self.theta)
        gates = torch.sigmoid(self.direction_gate)
        filtered = []
        for idx, band in enumerate(high):
            scale = band.abs().mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            threshold = thresholds[idx].unsqueeze(0) * scale
            filtered.append(self._soft_threshold(band, threshold) * gates[idx].unsqueeze(0))

        stack = torch.stack(filtered, dim=2)
        band_attention = F.softmax(stack.abs().mean(dim=1, keepdim=True), dim=2)
        high_mix = (stack * band_attention).sum(dim=2)
        redistribution = F.softmax(self.subband_redistribution, dim=1).view(1, self.channels, 3, 1, 1)
        redistributed = high_mix.unsqueeze(2) * redistribution

        rec = self.wavelet.idwt(
            WaveletBands(
                ll=bands.ll,
                lh=redistributed[:, :, 0],
                hl=redistributed[:, :, 1],
                hh=redistributed[:, :, 2],
                original_hw=bands.original_hw,
            )
        )
        pooled = F.adaptive_avg_pool2d(rec, 1).flatten(1)
        channel_weight = F.softmax(self.channel_proj(pooled), dim=1).view(x.size(0), self.channels, 1, 1)
        out = x * channel_weight + self.gamma * self.se(rec)
        diagnostics = {
            "ll": bands.ll,
            "lh": bands.lh,
            "hl": bands.hl,
            "hh": bands.hh,
            "band_attention": band_attention.squeeze(1),
        }
        return out, diagnostics


class DirectionalWaveletAttention(nn.Module):
    """Windowed attention on LL features guided by directional high-frequency bands."""

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.wavelet = HaarWaveletTransform()
        self.spectral_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 4, 4), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(dim // 4, 4), dim, 1),
            nn.Sigmoid(),
        )
        self.direction_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1, groups=_group_count(dim)),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid(),
        )
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Dropout2d(dropout))

    def _partition(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, height, width = x.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        windows = x.view(
            x.size(0),
            x.size(1),
            hp // self.window_size,
            self.window_size,
            wp // self.window_size,
            self.window_size,
        )
        windows = windows.permute(0, 2, 4, 1, 3, 5).contiguous()
        return windows.view(-1, x.size(1), self.window_size, self.window_size), (hp, wp)

    def _reverse(self, windows: torch.Tensor, batch_size: int, padded_hw: tuple[int, int], target_hw: tuple[int, int]) -> torch.Tensor:
        hp, wp = padded_hw
        x = windows.view(
            batch_size,
            hp // self.window_size,
            wp // self.window_size,
            self.dim,
            self.window_size,
            self.window_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(batch_size, self.dim, hp, wp)
        height, width = target_hw
        return x[:, :, :height, :width]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = x * self.spectral_gate(x)
        bands = self.wavelet.dwt(x)
        direction = self.direction_conv(torch.cat([bands.lh, bands.hl], dim=1))

        q, k, v = self.qkv(bands.ll).chunk(3, dim=1)
        v = v * (1.0 + direction)
        q_win, padded_hw = self._partition(q)
        k_win, _ = self._partition(k)
        v_win, _ = self._partition(v)

        tokens = self.window_size * self.window_size
        head_dim = self.dim // self.num_heads
        q_flat = q_win.flatten(2).transpose(1, 2).view(-1, tokens, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k_flat = k_win.flatten(2).transpose(1, 2).view(-1, tokens, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v_flat = v_win.flatten(2).transpose(1, 2).view(-1, tokens, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attention = (q_flat @ k_flat.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        out = attention @ v_flat
        out = out.permute(0, 2, 1, 3).contiguous().view(-1, tokens, self.dim)
        out = out.transpose(1, 2).view(-1, self.dim, self.window_size, self.window_size)
        ll_att = self._reverse(out, x.size(0), padded_hw, bands.ll.shape[-2:])
        ll_att = self.proj(ll_att)
        rec = self.wavelet.idwt(WaveletBands(ll=ll_att, lh=bands.lh, hl=bands.hl, hh=bands.hh, original_hw=bands.original_hw))
        return rec + shortcut


class WaveletMorphologyEncoder(nn.Module):
    """H&E encoder returning a morphology token and feature map."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        groups = _group_count(latent_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, 3, padding=1),
            nn.GroupNorm(groups, latent_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(groups, latent_dim),
            nn.SiLU(inplace=True),
        )
        self.cross_band = WaveletGatedCrossBand(latent_dim)
        self.directional = DirectionalWaveletAttention(latent_dim, num_heads=4, window_size=4, dropout=dropout)
        self.refine = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(groups, latent_dim),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        image_patches: torch.Tensor,
        return_map: bool = False,
        return_diagnostics: bool = False,
    ):
        x = self.stem(image_patches)
        x, diagnostics = self.cross_band(x)
        x = self.directional(x)
        x = self.refine(x)
        token = self.pool(x).flatten(1)
        if return_map and return_diagnostics:
            return token, x, diagnostics
        if return_map:
            return token, x
        return token
