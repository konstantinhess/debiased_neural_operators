from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


Tensor = torch.Tensor


@dataclass(frozen=True)
class DeepONet1dConfig:
    in_channels: int = 2
    branch_hidden_channels: int = 32
    trunk_hidden_channels: int = 32
    latent_dim: int = 32
    branch_layers: int = 2
    trunk_layers: int = 2
    out_channels: int = 1


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
    for _ in range(max(0, num_hidden_layers - 1)):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class DeepONet1D(nn.Module):
    def __init__(self, config: DeepONet1dConfig) -> None:
        super().__init__()
        self.config = config
        self.branch_input = nn.Conv1d(config.in_channels, config.branch_hidden_channels, kernel_size=1)
        self.branch_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(config.branch_hidden_channels, config.branch_hidden_channels, kernel_size=1),
                nn.GELU(),
            )
            for _ in range(config.branch_layers)
        )
        self.branch_pool = nn.AdaptiveAvgPool1d(1)
        self.branch_head = nn.Linear(
            config.branch_hidden_channels,
            config.out_channels * config.latent_dim,
        )
        self.trunk = _build_mlp(
            input_dim=1,
            hidden_dim=config.trunk_hidden_channels,
            output_dim=config.out_channels * config.latent_dim,
            num_hidden_layers=config.trunk_layers,
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

    def _branch_features(self, stacked_inputs: Tensor) -> Tensor:
        hidden = self.branch_input(stacked_inputs)
        for block in self.branch_blocks:
            hidden = hidden + block(hidden)
        pooled = self.branch_pool(hidden).squeeze(-1)
        branch = self.branch_head(pooled)
        return branch.view(stacked_inputs.size(0), self.config.out_channels, self.config.latent_dim)

    def _trunk_features(self, x_query: Tensor) -> Tensor:
        if x_query.ndim == 1:
            x_query = x_query.unsqueeze(0)
        elif x_query.ndim != 2:
            raise ValueError("x_query must have shape [K] or [B, K]")
        trunk = self.trunk(x_query.unsqueeze(-1))
        return trunk.view(x_query.size(0), x_query.size(1), self.config.out_channels, self.config.latent_dim)

    def _evaluate(self, coeff: Tensor, x_query: Tensor, x_grid: Tensor) -> Tensor:
        stacked, _ = self._prepare_inputs(coeff, x_grid)
        if x_query.ndim == 1:
            x_query = x_query.unsqueeze(0).expand(stacked.size(0), -1)
        elif x_query.ndim != 2:
            raise ValueError("x_query must have shape [K] or [B, K]")
        if x_query.size(0) != stacked.size(0):
            raise ValueError("x_query batch size must match coeff batch size")
        branch = self._branch_features(stacked)
        trunk = self._trunk_features(x_query)
        values = torch.einsum("bol,bkol->bok", branch, trunk)
        if self.config.out_channels == 1:
            return values[:, 0, :]
        return values

    def forward(self, coeff: Tensor, x_grid: Tensor) -> Tensor:
        return self.predict_grid(coeff, x_grid)

    def predict_grid(self, coeff: Tensor, x_grid: Tensor) -> Tensor:
        return self._evaluate(coeff, x_grid, x_grid)

    def predict_points(self, coeff: Tensor, x_query: Tensor, x_grid: Tensor) -> Tensor:
        return self._evaluate(coeff, x_query, x_grid)

