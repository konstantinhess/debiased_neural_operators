from __future__ import annotations

import torch


Tensor = torch.Tensor
DEFAULT_THRESHOLD_KAPPA = 8.0
DEFAULT_THRESHOLD_C = 0.5
DARCY_VISIBLE_KAPPA_MODE = "darcy_visible_unit_interval"


def _validate_inputs(u_grid: Tensor, quad_weights: Tensor) -> None:
    if u_grid.ndim not in (1, 2):
        raise ValueError("u_grid must have shape [G] or [B, G]")
    if quad_weights.ndim == 1:
        if u_grid.shape[-1] != quad_weights.shape[0]:
            raise ValueError("u_grid and quad_weights must agree on the grid dimension")
    elif quad_weights.ndim == 2:
        if u_grid.ndim != 2 or u_grid.shape != quad_weights.shape:
            raise ValueError("batched quad_weights must have shape [B, G] matching u_grid")
    else:
        raise ValueError("quad_weights must have shape [G] or [B, G]")


def _validate_coordinate_inputs(u_grid: Tensor, x_grid: Tensor | None) -> None:
    if x_grid is None:
        return
    if x_grid.ndim == u_grid.ndim:
        if x_grid.shape != u_grid.shape:
            raise ValueError("1D coordinate grid must match u_grid shape")
        return
    if x_grid.ndim == u_grid.ndim + 1 and x_grid.shape[:-1] == u_grid.shape and x_grid.shape[-1] == 2:
        return
    raise ValueError("x_grid must have shape matching u_grid or u_grid plus a trailing coordinate dimension")


def _sigmoid_threshold(u_grid: Tensor, kappa: float, c: float) -> Tensor:
    return torch.sigmoid(kappa * (u_grid - c))


def _resolve_threshold_kappa(kappa: float, kappa_mode: str | None = None) -> float:
    if kappa_mode is None:
        return kappa
    if kappa_mode == DARCY_VISIBLE_KAPPA_MODE:
        return 7.5 + 2.5 * kappa
    raise ValueError(f"unsupported kappa_mode: {kappa_mode}")


def _regional_weight_star(x_grid: Tensor) -> Tensor:
    if x_grid.ndim in {1, 2} and (x_grid.ndim == 1 or x_grid.shape[-1] != 2):
        coord = x_grid
        coord_min = torch.amin(coord, dim=-1, keepdim=True)
        coord_max = torch.amax(coord, dim=-1, keepdim=True)
        coord_norm = (coord - coord_min) / torch.clamp(coord_max - coord_min, min=1e-8)
        return torch.exp(-0.5 * ((coord_norm - 0.7) / 0.15) ** 2)

    x_coord = x_grid[..., 0]
    y_coord = x_grid[..., 1]
    x_min = torch.amin(x_coord, dim=-1, keepdim=True)
    x_max = torch.amax(x_coord, dim=-1, keepdim=True)
    y_min = torch.amin(y_coord, dim=-1, keepdim=True)
    y_max = torch.amax(y_coord, dim=-1, keepdim=True)
    x_norm = (x_coord - x_min) / torch.clamp(x_max - x_min, min=1e-8)
    y_norm = (y_coord - y_min) / torch.clamp(y_max - y_min, min=1e-8)
    return torch.exp(-0.5 * (((x_norm - 0.7) / 0.18) ** 2 + ((y_norm - 0.35) / 0.18) ** 2))


def spatial_average(u_grid: Tensor, quad_weights: Tensor, x_grid: Tensor | None = None) -> Tensor:
    _validate_inputs(u_grid, quad_weights)
    return torch.sum(u_grid * quad_weights, dim=-1)


def auc_functional(u_grid: Tensor, quad_weights: Tensor, x_grid: Tensor | None = None) -> Tensor:
    return spatial_average(u_grid, quad_weights, x_grid=x_grid)


def l2_energy(u_grid: Tensor, quad_weights: Tensor, x_grid: Tensor | None = None) -> Tensor:
    _validate_inputs(u_grid, quad_weights)
    return 0.5 * torch.sum((u_grid ** 2) * quad_weights, dim=-1)


def smooth_time_above_threshold(
    u_grid: Tensor,
    quad_weights: Tensor,
    x_grid: Tensor | None = None,
    kappa: float = DEFAULT_THRESHOLD_KAPPA,
    c_star: float = DEFAULT_THRESHOLD_C,
) -> Tensor:
    _validate_inputs(u_grid, quad_weights)
    return torch.sum(_sigmoid_threshold(u_grid, kappa=kappa, c=c_star) * quad_weights, dim=-1)


def smooth_threshold_exceedance(
    u_grid: Tensor,
    quad_weights: Tensor,
    x_grid: Tensor | None = None,
    kappa: float = DEFAULT_THRESHOLD_KAPPA,
    c: float = DEFAULT_THRESHOLD_C,
    kappa_mode: str | None = None,
) -> Tensor:
    _validate_inputs(u_grid, quad_weights)
    kappa_internal = _resolve_threshold_kappa(kappa, kappa_mode=kappa_mode)
    return torch.sum(_sigmoid_threshold(u_grid, kappa=kappa_internal, c=c) * quad_weights, dim=-1)


def smooth_excess_above_threshold(
    u_grid: Tensor,
    quad_weights: Tensor,
    x_grid: Tensor | None = None,
    kappa: float = DEFAULT_THRESHOLD_KAPPA,
    c: float = DEFAULT_THRESHOLD_C,
    kappa_mode: str | None = None,
) -> Tensor:
    _validate_inputs(u_grid, quad_weights)
    kappa_internal = _resolve_threshold_kappa(kappa, kappa_mode=kappa_mode)
    shifted = kappa_internal * (u_grid - c)
    return torch.sum(torch.nn.functional.softplus(shifted) * quad_weights / kappa_internal, dim=-1)


def high_vorticity_exceedance(
    u_grid: Tensor,
    quad_weights: Tensor,
    x_grid: Tensor | None = None,
    kappa: float = 8.0,
    c: float = 1.0,
) -> Tensor:
    _validate_inputs(u_grid, quad_weights)
    shifted = kappa * (torch.abs(u_grid) - c)
    return torch.sum(torch.sigmoid(shifted) * quad_weights, dim=-1)


def weighted_regional_average(
    u_grid: Tensor,
    quad_weights: Tensor,
    x_grid: Tensor | None = None,
) -> Tensor:
    _validate_inputs(u_grid, quad_weights)
    _validate_coordinate_inputs(u_grid, x_grid)
    if x_grid is None:
        raise ValueError("weighted_regional_average requires x_grid")
    return torch.sum(_regional_weight_star(x_grid) * u_grid * quad_weights, dim=-1)


def logsumexp_functional(u_grid: Tensor, quad_weights: Tensor, x_grid: Tensor | None = None) -> Tensor:
    _validate_inputs(u_grid, quad_weights)
    shifted = u_grid - torch.amax(u_grid, dim=-1, keepdim=True)
    return torch.log(torch.sum(quad_weights * torch.exp(shifted), dim=-1)) + torch.amax(
        u_grid, dim=-1
    )


def soft_cmax_functional(
    u_grid: Tensor,
    quad_weights: Tensor,
    x_grid: Tensor | None = None,
    lam: float = 6.0,
) -> Tensor:
    _validate_inputs(u_grid, quad_weights)
    shifted = lam * u_grid
    max_shift = torch.amax(shifted, dim=-1, keepdim=True)
    log_weighted_sum = torch.log(torch.sum(quad_weights * torch.exp(shifted - max_shift), dim=-1))
    return (log_weighted_sum + max_shift.squeeze(-1)) / lam


def get_functional(name: str):
    functionals = {
        "average": spatial_average,
        "auc": auc_functional,
        "pk_auc": auc_functional,
        "l2_energy": l2_energy,
        "enstrophy": l2_energy,
        "smooth_tat": smooth_time_above_threshold,
        "tat": smooth_time_above_threshold,
        "smooth_threshold_exceedance": smooth_threshold_exceedance,
        "threshold_exceedance": smooth_threshold_exceedance,
        "g_thr": smooth_threshold_exceedance,
        "smooth_excess_above_threshold": smooth_excess_above_threshold,
        "excess_above_threshold": smooth_excess_above_threshold,
        "g_exc": smooth_excess_above_threshold,
        "high_vorticity_exceedance": high_vorticity_exceedance,
        "vorticity_exceedance": high_vorticity_exceedance,
        "weighted_regional_average": weighted_regional_average,
        "regional_average": weighted_regional_average,
        "g_reg": weighted_regional_average,
        "soft_cmax": soft_cmax_functional,
        "softmax": soft_cmax_functional,
        "logsumexp": logsumexp_functional,
        "log_sum_exp": logsumexp_functional,
    }
    if name not in functionals:
        raise ValueError(f"unsupported functional: {name}")
    return functionals[name]


def riesz_density(
    name: str,
    u_grid: Tensor,
    quad_weights: Tensor | None = None,
    x_grid: Tensor | None = None,
    **functional_params,
) -> Tensor:
    kappa = float(functional_params.get("kappa", DEFAULT_THRESHOLD_KAPPA))
    kappa_mode = functional_params.get("kappa_mode")
    kappa_internal = _resolve_threshold_kappa(kappa, kappa_mode=kappa_mode)
    threshold = float(functional_params.get("c", functional_params.get("c_star", DEFAULT_THRESHOLD_C)))
    if name in {"average", "auc", "pk_auc"}:
        return torch.ones_like(u_grid)
    if name in {"l2_energy", "enstrophy"}:
        return u_grid
    if name in {"smooth_tat", "tat"}:
        sigmoid = _sigmoid_threshold(u_grid, kappa=kappa_internal, c=threshold)
        return kappa_internal * sigmoid * (1.0 - sigmoid)
    if name in {"smooth_threshold_exceedance", "threshold_exceedance", "g_thr"}:
        sigmoid = _sigmoid_threshold(u_grid, kappa=kappa_internal, c=threshold)
        return kappa_internal * sigmoid * (1.0 - sigmoid)
    if name in {"smooth_excess_above_threshold", "excess_above_threshold", "g_exc"}:
        return _sigmoid_threshold(u_grid, kappa=kappa_internal, c=threshold)
    if name in {"high_vorticity_exceedance", "vorticity_exceedance"}:
        sigmoid = torch.sigmoid(kappa_internal * (torch.abs(u_grid) - threshold))
        return kappa_internal * sigmoid * (1.0 - sigmoid) * torch.sign(u_grid)
    if name in {"weighted_regional_average", "regional_average", "g_reg"}:
        _validate_coordinate_inputs(u_grid, x_grid)
        if x_grid is None:
            raise ValueError("x_grid is required for weighted_regional_average riesz_density")
        return _regional_weight_star(x_grid)
    if name in {"soft_cmax", "softmax"}:
        if quad_weights is None:
            raise ValueError("quad_weights are required for soft_cmax riesz_density")
        shifted = 6.0 * u_grid
        max_shift = torch.amax(shifted, dim=-1, keepdim=True)
        exp_shifted = torch.exp(shifted - max_shift)
        weighted = quad_weights * exp_shifted
        return exp_shifted / torch.sum(weighted, dim=-1, keepdim=True)
    if name in {"logsumexp", "log_sum_exp"}:
        if quad_weights is None:
            raise ValueError("quad_weights are required for logsumexp riesz_density")
        shifted = u_grid - torch.amax(u_grid, dim=-1, keepdim=True)
        exp_shifted = torch.exp(shifted)
        weighted = quad_weights * exp_shifted
        return exp_shifted / torch.sum(weighted, dim=-1, keepdim=True)
    raise ValueError(f"unsupported functional: {name}")


def functional_jvp(
    name: str,
    u_grid: Tensor,
    direction: Tensor,
    quad_weights: Tensor,
    x_grid: Tensor | None = None,
    **functional_params,
) -> Tensor:
    functional = get_functional(name)
    if u_grid.ndim != 1 or direction.ndim != 1 or quad_weights.ndim != 1:
        raise ValueError("functional_jvp expects one-dimensional tensors")
    if u_grid.shape != direction.shape or u_grid.shape != quad_weights.shape:
        raise ValueError("u_grid, direction, and quad_weights must have the same shape")
    if x_grid is not None and x_grid.ndim not in {1, 2}:
        raise ValueError("functional_jvp expects x_grid with shape [G] or [G, 2]")

    def wrapped(u: Tensor) -> Tensor:
        return functional(u, quad_weights, x_grid, **functional_params)

    _, jvp = torch.autograd.functional.jvp(wrapped, (u_grid,), (direction,), create_graph=True)
    return jvp
