from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Darcy2DConfig:
    dataset_name: str
    output_root: str
    seed: int
    pde_class: str
    num_points_x: int
    num_points_y: int
    coarse_cells_x: int
    coarse_cells_y: int
    num_basis: int
    log_coeff_scale: float
    coeff_min: float
    sampling_family: str
    max_num_patches: int
    amplitude_min: float
    amplitude_max: float
    patch_width_min: float
    patch_width_max: float
    num_observations: int
    observation_design: str
    noise_std: float
    design_strength: float
    epsilon_design: float
    saliency_scale: float
    splits: dict[str, int]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Darcy2DConfig":
        equation = payload["equation"]
        grid = payload["grid"]
        input_function = payload["input_function"]
        observations = payload["observations"]
        return cls(
            dataset_name=str(payload["dataset_name"]),
            output_root=str(payload.get("output_root", "generated/datasets")),
            seed=int(payload["seed"]),
            pde_class=str(equation["pde_class"]),
            num_points_x=int(grid["num_points_x"]),
            num_points_y=int(grid["num_points_y"]),
            coarse_cells_x=int(input_function["coarse_cells_x"]),
            coarse_cells_y=int(input_function["coarse_cells_y"]),
            num_basis=int(input_function["num_basis"]),
            log_coeff_scale=float(input_function["log_coeff_scale"]),
            coeff_min=float(input_function["coeff_min"]),
            sampling_family=str(input_function.get("sampling_family", "piecewise_constant_transformed_gaussian_field")),
            max_num_patches=int(input_function.get("max_num_patches", 2)),
            amplitude_min=float(input_function.get("amplitude_min", 0.4)),
            amplitude_max=float(input_function.get("amplitude_max", 1.2)),
            patch_width_min=float(input_function.get("patch_width_min", 0.12)),
            patch_width_max=float(input_function.get("patch_width_max", 0.35)),
            num_observations=int(observations["num_observations"]),
            observation_design=str(observations["design"]),
            noise_std=float(observations["noise_std"]),
            design_strength=float(observations.get("design_strength", 1.0)),
            epsilon_design=float(observations.get("epsilon_design", 1e-3)),
            saliency_scale=float(observations.get("saliency_scale", 4.0)),
            splits={str(k): int(v) for k, v in payload["splits"].items()},
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "Darcy2DConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def dataset_root(self) -> Path:
        return Path(self.output_root) / self.dataset_name


def build_grid(num_points_x: int, num_points_y: int) -> tuple[Array, Array, Array]:
    x_axis = np.linspace(0.0, 1.0, num_points_x, dtype=np.float64)
    y_axis = np.linspace(0.0, 1.0, num_points_y, dtype=np.float64)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
    x_grid = np.stack([xx, yy], axis=-1)
    return x_axis, y_axis, x_grid


def build_quadrature_weights(num_points_x: int, num_points_y: int) -> Array:
    wx = np.ones(num_points_x, dtype=np.float64)
    wy = np.ones(num_points_y, dtype=np.float64)
    wx[0] = 0.5
    wx[-1] = 0.5
    wy[0] = 0.5
    wy[-1] = 0.5
    weights = np.outer(wx, wy)
    weights /= weights.sum()
    return weights


def _basis_field(coarse_x: Array, coarse_y: Array, mode_x: int, mode_y: int) -> Array:
    return (
        np.sqrt(2.0) * np.cos(np.pi * mode_x * coarse_x)[:, None]
        * np.sqrt(2.0) * np.cos(np.pi * mode_y * coarse_y)[None, :]
    )


def sample_coefficient_field(
    rng: np.random.Generator,
    config: Darcy2DConfig,
) -> Array:
    if config.sampling_family == "pk_pulse_patch_2d":
        x_axis = np.linspace(0.0, 1.0, config.num_points_x, dtype=np.float64)
        y_axis = np.linspace(0.0, 1.0, config.num_points_y, dtype=np.float64)
        xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
        field = np.ones((config.num_points_x, config.num_points_y), dtype=np.float64)
        num_patches = int(rng.integers(1, config.max_num_patches + 1))
        for _ in range(num_patches):
            width_x = float(rng.uniform(config.patch_width_min, config.patch_width_max))
            width_y = float(rng.uniform(config.patch_width_min, config.patch_width_max))
            start_x = float(rng.uniform(0.0, max(0.0, 1.0 - width_x)))
            start_y = float(rng.uniform(0.0, max(0.0, 1.0 - width_y)))
            end_x = min(1.0, start_x + width_x)
            end_y = min(1.0, start_y + width_y)
            amplitude = float(rng.uniform(config.amplitude_min, config.amplitude_max))
            mask = (xx >= start_x) & (xx <= end_x) & (yy >= start_y) & (yy <= end_y)
            field[mask] += amplitude
        return np.clip(field, config.coeff_min, None)
    if config.sampling_family != "piecewise_constant_transformed_gaussian_field":
        raise ValueError(f"unsupported sampling_family: {config.sampling_family}")
    coarse_x = np.linspace(0.0, 1.0, config.coarse_cells_x, dtype=np.float64)
    coarse_y = np.linspace(0.0, 1.0, config.coarse_cells_y, dtype=np.float64)
    latent = np.zeros((config.coarse_cells_x, config.coarse_cells_y), dtype=np.float64)
    for mode_x in range(1, config.num_basis + 1):
        for mode_y in range(1, config.num_basis + 1):
            latent += rng.normal(loc=0.0, scale=1.0) * _basis_field(coarse_x, coarse_y, mode_x, mode_y)
    latent *= config.log_coeff_scale / max(config.num_basis, 1)
    coarse_coeff = np.exp(latent)
    coarse_coeff = np.clip(coarse_coeff, config.coeff_min, None)
    repeat_x = int(np.ceil(config.num_points_x / config.coarse_cells_x))
    repeat_y = int(np.ceil(config.num_points_y / config.coarse_cells_y))
    tiled = np.repeat(np.repeat(coarse_coeff, repeat_x, axis=0), repeat_y, axis=1)
    return tiled[: config.num_points_x, : config.num_points_y]


def _harmonic_mean(a: Array, b: Array) -> Array:
    return 2.0 * a * b / np.clip(a + b, 1e-8, None)


def solve_darcy_2d(coeff_field: Array) -> Array:
    if coeff_field.ndim != 2:
        raise ValueError("coeff_field must have shape [H, W]")
    num_points_x, num_points_y = coeff_field.shape
    if num_points_x < 3 or num_points_y < 3:
        raise ValueError("need at least a 3x3 grid")

    hx = 1.0 / (num_points_x - 1)
    hy = 1.0 / (num_points_y - 1)
    interior_x = num_points_x - 2
    interior_y = num_points_y - 2
    size = interior_x * interior_y
    matrix = np.zeros((size, size), dtype=np.float64)
    rhs = np.ones(size, dtype=np.float64)

    def flat(i: int, j: int) -> int:
        return i * interior_y + j

    for i in range(interior_x):
        for j in range(interior_y):
            gi = i + 1
            gj = j + 1
            a_center = coeff_field[gi, gj]
            a_w = _harmonic_mean(a_center, coeff_field[gi - 1, gj]) / (hx * hx)
            a_e = _harmonic_mean(a_center, coeff_field[gi + 1, gj]) / (hx * hx)
            a_s = _harmonic_mean(a_center, coeff_field[gi, gj - 1]) / (hy * hy)
            a_n = _harmonic_mean(a_center, coeff_field[gi, gj + 1]) / (hy * hy)
            row = flat(i, j)
            matrix[row, row] = a_w + a_e + a_s + a_n
            if i > 0:
                matrix[row, flat(i - 1, j)] = -a_w
            if i < interior_x - 1:
                matrix[row, flat(i + 1, j)] = -a_e
            if j > 0:
                matrix[row, flat(i, j - 1)] = -a_s
            if j < interior_y - 1:
                matrix[row, flat(i, j + 1)] = -a_n

    solution = np.zeros_like(coeff_field, dtype=np.float64)
    solution[1:-1, 1:-1] = np.linalg.solve(matrix, rhs).reshape(interior_x, interior_y)
    return solution


def normalize_for_design(values: Array) -> Array:
    shifted = values - values.mean()
    scale = np.max(np.abs(shifted))
    if scale <= 0.0:
        return np.zeros_like(values)
    return shifted / scale


def _adaptive_probabilities(saliency: Array, config: Darcy2DConfig) -> Array:
    logits = np.exp(config.saliency_scale * np.abs(normalize_for_design(saliency))) + config.epsilon_design
    return logits / logits.sum()


def _interface_saliency(coeff_field: Array) -> Array:
    log_coeff = np.log(np.clip(coeff_field, 1e-8, None))
    grad_x = np.roll(log_coeff, -1, axis=0) - np.roll(log_coeff, 1, axis=0)
    grad_y = np.roll(log_coeff, -1, axis=1) - np.roll(log_coeff, 1, axis=1)
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def _solution_gradient(solution: Array) -> Array:
    grad_x = np.roll(solution, -1, axis=0) - np.roll(solution, 1, axis=0)
    grad_y = np.roll(solution, -1, axis=1) - np.roll(solution, 1, axis=1)
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def select_equidistant_indices(num_points_x: int, num_points_y: int, num_observations: int) -> Array:
    side = int(round(np.sqrt(num_observations)))
    if side * side != num_observations:
        raise ValueError("2D equidistant design expects num_observations to be a perfect square")
    rows = np.rint(np.linspace(0, num_points_x - 1, num=side)).astype(np.int64)
    cols = np.rint(np.linspace(0, num_points_y - 1, num=side)).astype(np.int64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([rr.reshape(-1), cc.reshape(-1)], axis=-1)


def sample_observations(
    rng: np.random.Generator,
    coeff_field: Array,
    solution: Array,
    x_grid: Array,
    config: Darcy2DConfig,
) -> dict[str, Array]:
    if config.observation_design == "equidistant":
        obs_index = select_equidistant_indices(coeff_field.shape[0], coeff_field.shape[1], config.num_observations)
        design_prob = np.zeros_like(solution, dtype=np.float64)
        design_prob[obs_index[:, 0], obs_index[:, 1]] = 1.0 / config.num_observations
    elif config.observation_design == "input_mixture":
        adaptive_prob = _adaptive_probabilities(_interface_saliency(coeff_field), config)
        uniform_prob = np.full_like(solution, 1.0 / solution.size, dtype=np.float64)
        rho = float(np.clip(config.design_strength, 0.0, 1.0))
        design_prob = (1.0 - rho) * uniform_prob + rho * adaptive_prob
        flat_index = rng.choice(coeff_field.size, size=config.num_observations, replace=False, p=design_prob.reshape(-1)).astype(np.int64)
        flat_index.sort()
        obs_index = np.stack(np.unravel_index(flat_index, coeff_field.shape), axis=-1)
    elif config.observation_design == "pilot_mixture":
        adaptive_prob = _adaptive_probabilities(_solution_gradient(solution), config)
        uniform_prob = np.full_like(solution, 1.0 / solution.size, dtype=np.float64)
        rho = float(np.clip(config.design_strength, 0.0, 1.0))
        design_prob = (1.0 - rho) * uniform_prob + rho * adaptive_prob
        flat_index = rng.choice(coeff_field.size, size=config.num_observations, replace=False, p=design_prob.reshape(-1)).astype(np.int64)
        flat_index.sort()
        obs_index = np.stack(np.unravel_index(flat_index, coeff_field.shape), axis=-1)
    elif config.observation_design == "inverse_mixture":
        inverse_coeff = 1.0 / np.clip(coeff_field, 1e-8, None)
        saliency = inverse_coeff * (1.0 + _interface_saliency(coeff_field))
        adaptive_prob = _adaptive_probabilities(saliency, config)
        uniform_prob = np.full_like(solution, 1.0 / solution.size, dtype=np.float64)
        rho = float(np.clip(config.design_strength, 0.0, 1.0))
        design_prob = (1.0 - rho) * uniform_prob + rho * adaptive_prob
        flat_index = rng.choice(coeff_field.size, size=config.num_observations, replace=False, p=design_prob.reshape(-1)).astype(np.int64)
        flat_index.sort()
        obs_index = np.stack(np.unravel_index(flat_index, coeff_field.shape), axis=-1)
    elif config.observation_design == "inverse_energy_mixture":
        inverse_coeff = 1.0 / np.clip(coeff_field, 1e-8, None)
        saliency = inverse_coeff * (0.5 + np.abs(solution) + _solution_gradient(solution))
        adaptive_prob = _adaptive_probabilities(saliency, config)
        uniform_prob = np.full_like(solution, 1.0 / solution.size, dtype=np.float64)
        rho = float(np.clip(config.design_strength, 0.0, 1.0))
        design_prob = (1.0 - rho) * uniform_prob + rho * adaptive_prob
        flat_index = rng.choice(coeff_field.size, size=config.num_observations, replace=False, p=design_prob.reshape(-1)).astype(np.int64)
        flat_index.sort()
        obs_index = np.stack(np.unravel_index(flat_index, coeff_field.shape), axis=-1)
    elif config.observation_design == "input_dependent":
        design_prob = _adaptive_probabilities(np.log(np.clip(coeff_field, 1e-8, None)), config)
        flat_index = rng.choice(
            coeff_field.size,
            size=config.num_observations,
            replace=False,
            p=design_prob.reshape(-1),
        ).astype(np.int64)
        flat_index.sort()
        obs_index = np.stack(np.unravel_index(flat_index, coeff_field.shape), axis=-1)
    else:
        raise ValueError(f"unsupported observation design: {config.observation_design}")

    obs_x = x_grid[obs_index[:, 0], obs_index[:, 1]]
    obs_signal = solution[obs_index[:, 0], obs_index[:, 1]]
    obs_y = obs_signal + rng.normal(loc=0.0, scale=config.noise_std, size=config.num_observations)
    xi = np.zeros_like(design_prob, dtype=np.float64)
    positive = design_prob > 0.0
    quad_weights = build_quadrature_weights(coeff_field.shape[0], coeff_field.shape[1])
    xi[positive] = quad_weights[positive] / design_prob[positive]
    return {
        "obs_index": obs_index,
        "obs_x": obs_x,
        "obs_signal": obs_signal,
        "obs_y": obs_y,
        "design_prob": design_prob,
        "xi": xi,
    }


def generate_split(
    config: Darcy2DConfig,
    split_name: str,
    num_samples: int,
    rng: np.random.Generator,
) -> dict[str, Array]:
    _, _, x_grid = build_grid(config.num_points_x, config.num_points_y)
    quad_weights = build_quadrature_weights(config.num_points_x, config.num_points_y)
    coeff = np.zeros((num_samples, config.num_points_x, config.num_points_y), dtype=np.float32)
    solution = np.zeros_like(coeff)
    obs_index = np.zeros((num_samples, config.num_observations, 2), dtype=np.int64)
    obs_x = np.zeros((num_samples, config.num_observations, 2), dtype=np.float32)
    obs_signal = np.zeros((num_samples, config.num_observations), dtype=np.float32)
    obs_y = np.zeros((num_samples, config.num_observations), dtype=np.float32)
    design_prob = np.zeros_like(coeff)
    xi = np.zeros_like(coeff)

    for row in range(num_samples):
        coeff_field = sample_coefficient_field(rng, config)
        solved = solve_darcy_2d(coeff_field)
        obs = sample_observations(rng, coeff_field, solved, x_grid, config)
        coeff[row] = coeff_field.astype(np.float32)
        solution[row] = solved.astype(np.float32)
        obs_index[row] = obs["obs_index"]
        obs_x[row] = obs["obs_x"].astype(np.float32)
        obs_signal[row] = obs["obs_signal"].astype(np.float32)
        obs_y[row] = obs["obs_y"].astype(np.float32)
        design_prob[row] = obs["design_prob"].astype(np.float32)
        xi[row] = obs["xi"].astype(np.float32)

    return {
        "split_name": np.array(split_name),
        "coeff": coeff,
        "solution": solution,
        "obs_index": obs_index,
        "obs_x": obs_x,
        "obs_signal": obs_signal,
        "obs_y": obs_y,
        "design_prob": design_prob,
        "xi": xi,
        "x_grid": x_grid.astype(np.float32),
        "quad_weights": quad_weights.astype(np.float32),
    }


def dataset_metadata(config: Darcy2DConfig) -> dict[str, Any]:
    return {
        "benchmark_name": "darcy_2d",
        "dataset_name": config.dataset_name,
        "schema_version": 1,
        "pde_class": config.pde_class,
        "observation_design": config.observation_design,
        "seed": config.seed,
        "grid": {
            "num_points_x": config.num_points_x,
            "num_points_y": config.num_points_y,
            "domain": [[0.0, 1.0], [0.0, 1.0]],
            "quadrature_normalization": "sum_to_one",
        },
        "equation": {
            "boundary": "zero_dirichlet",
            "forcing_value": 1.0,
        },
        "input_function": {
            "representation": "grid_values",
            "sampling_family": config.sampling_family,
            "coarse_cells_x": config.coarse_cells_x,
            "coarse_cells_y": config.coarse_cells_y,
            "num_basis": config.num_basis,
            "log_coeff_scale": config.log_coeff_scale,
            "coeff_min": config.coeff_min,
            "max_num_patches": config.max_num_patches,
            "amplitude_min": config.amplitude_min,
            "amplitude_max": config.amplitude_max,
            "patch_width_min": config.patch_width_min,
            "patch_width_max": config.patch_width_max,
        },
        "observations": {
            "num_observations": config.num_observations,
            "noise_std": config.noise_std,
            "design_strength": config.design_strength,
            "epsilon_design": config.epsilon_design,
            "saliency_scale": config.saliency_scale,
        },
        "splits": config.splits,
        "arrays": {
            "coeff": {"dtype": "float32", "shape": "[N, H, W]"},
            "solution": {"dtype": "float32", "shape": "[N, H, W]"},
            "obs_index": {"dtype": "int64", "shape": "[N, K, 2]"},
            "obs_x": {"dtype": "float32", "shape": "[N, K, 2]"},
            "obs_signal": {"dtype": "float32", "shape": "[N, K]"},
            "obs_y": {"dtype": "float32", "shape": "[N, K]"},
            "design_prob": {"dtype": "float32", "shape": "[N, H, W]"},
            "xi": {"dtype": "float32", "shape": "[N, H, W]"},
            "x_grid": {"dtype": "float32", "shape": "[H, W, 2]"},
            "quad_weights": {"dtype": "float32", "shape": "[H, W]"},
        },
    }
