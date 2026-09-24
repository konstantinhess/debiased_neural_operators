from __future__ import annotations

import torch


Tensor = torch.Tensor
DEFAULT_THRESHOLD_KAPPA = 8.0
DEFAULT_THRESHOLD_C = 0.5
DARCY_VISIBLE_KAPPA_MODE = "darcy_visible_unit_interval"


def _validate(u_grid: Tensor, weights: Tensor) -> None:
    if u_grid.ndim not in (1, 2):
        raise ValueError("u_grid must have shape [G] or [B, G]")
    if weights.ndim == 1 and u_grid.shape[-1] != weights.shape[0]:
        raise ValueError("u_grid and weights disagree on grid size")
    if weights.ndim == 2 and (u_grid.ndim != 2 or u_grid.shape != weights.shape):
        raise ValueError("batched weights must match u_grid")
    if weights.ndim not in (1, 2):
        raise ValueError("weights must have shape [G] or [B, G]")


def _resolve_kappa(kappa: float, mode: str | None) -> float:
    if mode is None:
        return kappa
    if mode == DARCY_VISIBLE_KAPPA_MODE:
        return 7.5 + 2.5 * kappa
    raise ValueError(f"unsupported kappa mode: {mode}")


def spatial_average(u_grid: Tensor, quad_weights: Tensor, x_grid: Tensor | None = None) -> Tensor:
    _validate(u_grid, quad_weights)
    return torch.sum(u_grid * quad_weights, dim=-1)


def auc_functional(u_grid: Tensor, quad_weights: Tensor, x_grid: Tensor | None = None) -> Tensor:
    return spatial_average(u_grid, quad_weights, x_grid)


def smooth_time_above_threshold(
    u_grid: Tensor,
    quad_weights: Tensor,
    x_grid: Tensor | None = None,
    kappa: float = DEFAULT_THRESHOLD_KAPPA,
    c_star: float = DEFAULT_THRESHOLD_C,
) -> Tensor:
    _validate(u_grid, quad_weights)
    return torch.sum(torch.sigmoid(kappa * (u_grid - c_star)) * quad_weights, dim=-1)


def smooth_excess_above_threshold(
    u_grid: Tensor,
    quad_weights: Tensor,
    x_grid: Tensor | None = None,
    kappa: float = DEFAULT_THRESHOLD_KAPPA,
    c: float = DEFAULT_THRESHOLD_C,
    kappa_mode: str | None = None,
) -> Tensor:
    _validate(u_grid, quad_weights)
    internal = _resolve_kappa(kappa, kappa_mode)
    return torch.sum(
        torch.nn.functional.softplus(internal * (u_grid - c)) * quad_weights / internal,
        dim=-1,
    )


def soft_cmax_functional(
    u_grid: Tensor,
    quad_weights: Tensor,
    x_grid: Tensor | None = None,
    lam: float = 6.0,
) -> Tensor:
    _validate(u_grid, quad_weights)
    if lam <= 0:
        raise ValueError("soft_cmax lam must be positive")
    shifted = lam * u_grid
    maximum = torch.amax(shifted, dim=-1, keepdim=True)
    log_sum = torch.log(torch.sum(quad_weights * torch.exp(shifted - maximum), dim=-1))
    return (log_sum + maximum.squeeze(-1)) / lam


def get_functional(name: str):
    registry = {
        "average": spatial_average,
        "auc": auc_functional,
        "smooth_tat": smooth_time_above_threshold,
        "smooth_excess_above_threshold": smooth_excess_above_threshold,
        "soft_cmax": soft_cmax_functional,
    }
    if name not in registry:
        raise ValueError(f"unsupported paper functional: {name}")
    return registry[name]


def riesz_density(
    name: str,
    u_grid: Tensor,
    quad_weights: Tensor | None = None,
    x_grid: Tensor | None = None,
    **params,
) -> Tensor:
    if name in {"average", "auc"}:
        return torch.ones_like(u_grid)
    if name == "smooth_tat":
        kappa = float(params.get("kappa", DEFAULT_THRESHOLD_KAPPA))
        threshold = float(params.get("c_star", DEFAULT_THRESHOLD_C))
        sigmoid = torch.sigmoid(kappa * (u_grid - threshold))
        return kappa * sigmoid * (1.0 - sigmoid)
    if name == "smooth_excess_above_threshold":
        kappa = _resolve_kappa(
            float(params.get("kappa", DEFAULT_THRESHOLD_KAPPA)),
            params.get("kappa_mode"),
        )
        threshold = float(params.get("c", DEFAULT_THRESHOLD_C))
        return torch.sigmoid(kappa * (u_grid - threshold))
    if name == "soft_cmax":
        if quad_weights is None:
            raise ValueError("soft_cmax Riesz density requires quadrature weights")
        lam = float(params.get("lam", 6.0))
        if lam <= 0:
            raise ValueError("soft_cmax lam must be positive")
        shifted = lam * u_grid
        maximum = torch.amax(shifted, dim=-1, keepdim=True)
        exponentials = torch.exp(shifted - maximum)
        return exponentials / torch.sum(quad_weights * exponentials, dim=-1, keepdim=True)
    raise ValueError(f"unsupported paper functional: {name}")


def functional_jvp(
    name: str,
    u_grid: Tensor,
    direction: Tensor,
    quad_weights: Tensor,
    x_grid: Tensor | None = None,
    **params,
) -> Tensor:
    if u_grid.ndim != 1 or direction.ndim != 1 or quad_weights.ndim != 1:
        raise ValueError("functional_jvp expects one-dimensional tensors")
    if u_grid.shape != direction.shape or u_grid.shape != quad_weights.shape:
        raise ValueError("u_grid, direction, and weights must have the same shape")
    functional = get_functional(name)

    def wrapped(value: Tensor) -> Tensor:
        return functional(value, quad_weights, x_grid, **params)

    _, jvp = torch.autograd.functional.jvp(
        wrapped,
        (u_grid,),
        (direction,),
        create_graph=True,
    )
    return jvp
