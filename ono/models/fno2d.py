from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


Tensor = torch.Tensor


@dataclass(frozen=True)
class FNO2dConfig:
    in_channels: int = 3
    hidden_channels: int = 24
    out_channels: int = 1
    num_layers: int = 3
    num_modes_x: int = 8
    num_modes_y: int = 8


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_modes_x: int, num_modes_y: int) -> None:
        super().__init__()
        scale = 1.0 / max(1, in_channels * out_channels)
        self.out_channels = out_channels
        self.num_modes_x = num_modes_x
        self.num_modes_y = num_modes_y
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, num_modes_x, num_modes_y)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, num_modes_x, num_modes_y)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, size_x, size_y = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-2, -1))
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            size_x,
            x_ft.size(-1),
            dtype=torch.cfloat,
            device=x.device,
        )
        modes_x = min(self.num_modes_x, x_ft.size(-2))
        modes_y = min(self.num_modes_y, x_ft.size(-1))
        weight = torch.complex(
            self.weight_real[:, :, :modes_x, :modes_y],
            self.weight_imag[:, :, :modes_x, :modes_y],
        )
        out_ft[:, :, :modes_x, :modes_y] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :modes_x, :modes_y],
            weight,
        )
        return torch.fft.irfftn(out_ft, s=(size_x, size_y), dim=(-2, -1))


class FNOBlock2d(nn.Module):
    def __init__(self, channels: int, num_modes_x: int, num_modes_y: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, num_modes_x, num_modes_y)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.spectral(x) + self.pointwise(x))


class NeuralOperator2D(nn.Module):
    def __init__(self, config: FNO2dConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv2d(config.in_channels, config.hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(config.hidden_channels, config.num_modes_x, config.num_modes_y) for _ in range(config.num_layers)]
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.hidden_channels, config.out_channels, kernel_size=1),
        )

    def _prepare_inputs(self, coeff: Tensor, x_grid: Tensor) -> tuple[Tensor, Tensor]:
        if coeff.ndim != 3:
            raise ValueError("coeff must have shape [B, H, W]")
        if x_grid.ndim == 3:
            x_grid = x_grid.unsqueeze(0).expand(coeff.size(0), -1, -1, -1)
        elif x_grid.ndim != 4:
            raise ValueError("x_grid must have shape [H, W, 2] or [B, H, W, 2]")
        if x_grid.shape[:3] != coeff.shape:
            raise ValueError("coeff and x_grid must agree on shape [B, H, W]")
        stacked = torch.cat([coeff.unsqueeze(1), x_grid.permute(0, 3, 1, 2)], dim=1)
        return stacked, x_grid

    def forward(self, coeff: Tensor, x_grid: Tensor) -> Tensor:
        stacked, _ = self._prepare_inputs(coeff, x_grid)
        hidden = self.input_projection(stacked)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(hidden).squeeze(1)

    def predict_grid(self, coeff: Tensor, x_grid: Tensor) -> Tensor:
        return self.forward(coeff, x_grid)

    def predict_points(self, coeff: Tensor, x_query: Tensor, x_grid: Tensor) -> Tensor:
        grid_values = self.predict_grid(coeff, x_grid)
        if x_query.ndim == 2:
            x_query = x_query.unsqueeze(0).expand(coeff.size(0), -1, -1)
        elif x_query.ndim != 3:
            raise ValueError("x_query must have shape [K, 2] or [B, K, 2]")
        if x_query.size(0) != coeff.size(0):
            raise ValueError("x_query batch size must match coeff batch size")
        if x_grid.ndim == 3:
            x_grid = x_grid.unsqueeze(0).expand(coeff.size(0), -1, -1, -1)
        outputs = []
        for batch_idx in range(coeff.size(0)):
            outputs.append(_bilinear_interp_2d(x_grid[batch_idx], grid_values[batch_idx], x_query[batch_idx]))
        return torch.stack(outputs, dim=0)


def _bilinear_interp_2d(x_grid: Tensor, y_grid: Tensor, x_query: Tensor) -> Tensor:
    if x_grid.ndim != 3 or x_grid.size(-1) != 2:
        raise ValueError("x_grid must have shape [H, W, 2]")
    if y_grid.ndim != 2 or x_grid.shape[:2] != y_grid.shape:
        raise ValueError("y_grid must have shape [H, W]")
    x_axis = x_grid[:, 0, 0].contiguous()
    y_axis = x_grid[0, :, 1].contiguous()
    query_x = torch.clamp(x_query[:, 0], min=float(x_axis[0]), max=float(x_axis[-1]))
    query_y = torch.clamp(x_query[:, 1], min=float(y_axis[0]), max=float(y_axis[-1]))

    right_x = torch.searchsorted(x_axis, query_x, right=False)
    right_y = torch.searchsorted(y_axis, query_y, right=False)
    right_x = torch.clamp(right_x, 1, x_axis.numel() - 1)
    right_y = torch.clamp(right_y, 1, y_axis.numel() - 1)
    left_x = right_x - 1
    left_y = right_y - 1

    x0 = x_axis[left_x]
    x1 = x_axis[right_x]
    y0 = y_axis[left_y]
    y1 = y_axis[right_y]
    tx = (query_x - x0) / torch.clamp(x1 - x0, min=1e-8)
    ty = (query_y - y0) / torch.clamp(y1 - y0, min=1e-8)

    q00 = y_grid[left_x, left_y]
    q01 = y_grid[left_x, right_y]
    q10 = y_grid[right_x, left_y]
    q11 = y_grid[right_x, right_y]
    return (
        (1.0 - tx) * (1.0 - ty) * q00
        + (1.0 - tx) * ty * q01
        + tx * (1.0 - ty) * q10
        + tx * ty * q11
    )
