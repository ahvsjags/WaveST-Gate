"""Local cross-modal refinement without external CV dependencies."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPooling2D(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0) -> None:
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_exp = torch.exp(x.clamp(max=10.0))
        return self.avgpool(x_exp * x) / self.avgpool(x_exp).clamp_min(1e-6)


class LocalAttention(nn.Module):
    def __init__(self, channels: int, hidden: int = 16) -> None:
        super().__init__()
        hidden = min(max(hidden, 4), channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            SoftPooling2D(7, stride=3, padding=3),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.interpolate(self.body(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(x[:, :1])
        return x * weights * gate


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, b: int = 1, gamma: int = 2) -> None:
        super().__init__()
        kernel_size = int(abs((math.log(max(channels, 2), 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class CrossModalAttention(nn.Module):
    def __init__(self, channels: int, max_tokens: int = 4096) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        hidden = max(channels // 8, 4)
        self.query = nn.Conv2d(channels, hidden, 1)
        self.key = nn.Conv2d(channels, hidden, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, source: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = source.shape
        original = source
        attn_source = source
        attn_guidance = guidance
        tokens = height * width
        if tokens > self.max_tokens:
            scale = math.sqrt(self.max_tokens / float(tokens))
            pooled_h = max(int(height * scale), 1)
            pooled_w = max(int(width * scale), 1)
            attn_source = F.adaptive_avg_pool2d(source, (pooled_h, pooled_w))
            attn_guidance = F.adaptive_avg_pool2d(guidance, (pooled_h, pooled_w))
            height, width = pooled_h, pooled_w
        q = self.query(attn_source).view(bsz, -1, height * width).transpose(1, 2)
        k = self.key(attn_guidance).view(bsz, -1, height * width)
        attention = torch.bmm(q, k).softmax(dim=-1)
        value = self.value(attn_guidance).view(bsz, channels, height * width)
        out = torch.bmm(value, attention.transpose(1, 2)).view(bsz, channels, height, width)
        if out.shape[-2:] != original.shape[-2:]:
            out = F.interpolate(out, size=original.shape[-2:], mode="bilinear", align_corners=False)
        return original + self.gamma * out


class LocalCrossModalRefinement(nn.Module):
    """Refine image maps with a spatially broadcast expression map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.image_attention = LocalAttention(channels)
        self.expression_attention = LocalAttention(channels)
        self.channel_attention = ChannelAttention(channels)
        self.cross_attention = CrossModalAttention(channels)
        self.output = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, image_map: torch.Tensor, expression_map: torch.Tensor) -> torch.Tensor:
        image_map = self.channel_attention(self.image_attention(image_map))
        expression_map = self.channel_attention(self.expression_attention(expression_map))
        cross = self.cross_attention(image_map, expression_map)
        return self.output(torch.cat([image_map, expression_map, cross], dim=1))
