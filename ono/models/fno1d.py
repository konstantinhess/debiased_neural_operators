from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


Tensor = torch.Tensor


@dataclass(frozen=True)
class FNO1dConfig:
    in_channels: int = 2
    hidden_channels: int = 32
    out_channels: int = 1
    num_layers: int = 3
    num_modes: int = 12


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes = num_modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, num_modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, num_modes)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, grid_size = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.size(-1),
            dtype=torch.cfloat,
            device=x.device,
        )
        modes = min(self.num_modes, x_ft.size(-1))
        weight = torch.complex(
            self.weight_real[:, :, :modes],
            self.weight_imag[:, :, :modes],
        )
        out_ft[:, :, :modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :modes],
            weight,
        )
        return torch.fft.irfft(out_ft, n=grid_size, dim=-1)


class FNOBlock1d(nn.Module):
    def __init__(self, channels: int, num_modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, num_modes)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.spectral(x) + self.pointwise(x))


class NeuralOperator1D(nn.Module):
    def __init__(self, config: FNO1dConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv1d(config.in_channels, config.hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock1d(config.hidden_channels, config.num_modes) for _ in range(config.num_layers)]
        )
        self.output_projection = nn.Sequential(
            nn.Conv1d(config.hidden_channels, config.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(config.hidden_channels, config.out_channels, kernel_size=1),
        )

    def _prepare_inputs(self, coeff: Tensor, x_grid: Tensor) -> tuple[Tensor, Tensor]:
        if coeff.ndim == 2:
            coeff_channels = coeff.unsqueeze(1)
        elif coeff.ndim == 3:
            coeff_channels = coeff
        else:
            raise ValueError("coeff must have shape [B, G] or [B, C, G]")
        if x_grid.ndim == 1:
            x_grid = x_grid.unsqueeze(0).expand(coeff_channels.size(0), -1)
        elif x_grid.ndim != 2:
            raise ValueError("x_grid must have shape [G] or [B, G]")
        if coeff_channels.shape[0] != x_grid.shape[0] or coeff_channels.shape[-1] != x_grid.shape[-1]:
            raise ValueError("coeff and x_grid must agree on batch size and grid length")
        stacked = torch.cat([coeff_channels, x_grid.unsqueeze(1)], dim=1)
        if stacked.size(1) != self.config.in_channels:
            raise ValueError(
                f"expected {self.config.in_channels} input channels but received {stacked.size(1)}"
            )
        return stacked, x_grid

    def forward(self, coeff: Tensor, x_grid: Tensor) -> Tensor:
        stacked, _ = self._prepare_inputs(coeff, x_grid)
        hidden = self.input_projection(stacked)
        for block in self.blocks:
            hidden = block(hidden)
        output = self.output_projection(hidden).squeeze(1)
        return output

    def predict_grid(self, coeff: Tensor, x_grid: Tensor) -> Tensor:
        return self.forward(coeff, x_grid)

    def predict_points(self, coeff: Tensor, x_query: Tensor, x_grid: Tensor) -> Tensor:
        grid_values = self.predict_grid(coeff, x_grid)
        if x_query.ndim == 1:
            x_query = x_query.unsqueeze(0).expand(coeff.size(0), -1)
        elif x_query.ndim != 2:
            raise ValueError("x_query must have shape [K] or [B, K]")
        if x_query.size(0) != coeff.size(0):
            raise ValueError("x_query batch size must match coeff batch size")

        if x_grid.ndim == 1:
            x_grid_batch = x_grid.unsqueeze(0).expand(coeff.size(0), -1)
        else:
            x_grid_batch = x_grid

        outputs = []
        for batch_idx in range(coeff.size(0)):
            outputs.append(_linear_interp_1d(x_grid_batch[batch_idx], grid_values[batch_idx], x_query[batch_idx]))
        return torch.stack(outputs, dim=0)


def _linear_interp_1d(x_grid: Tensor, y_grid: Tensor, x_query: Tensor) -> Tensor:
    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("x_grid and y_grid must be one-dimensional")
    if x_grid.numel() != y_grid.numel():
        raise ValueError("x_grid and y_grid must have the same length")
    clamped = torch.clamp(x_query, min=float(x_grid[0]), max=float(x_grid[-1]))
    right = torch.searchsorted(x_grid, clamped, right=False)
    right = torch.clamp(right, 1, x_grid.numel() - 1)
    left = right - 1
    x_left = x_grid[left]
    x_right = x_grid[right]
    y_left = y_grid[left]
    y_right = y_grid[right]
    denom = torch.clamp(x_right - x_left, min=1e-8)
    alpha = (clamped - x_left) / denom
    return y_left + alpha * (y_right - y_left)

