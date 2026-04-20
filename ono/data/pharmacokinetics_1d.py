from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Pharmacokinetics1DConfig:
    dataset_name: str
    output_root: str
    seed: int
    pde_class: str
    time_horizon: float
    num_grid_points: int
    mean_log_clearance: float
    mean_log_volume: float
    std_log_clearance: float
    std_log_volume: float
    corr_log_clearance_volume: float
    pulse_family: str
    max_num_pulses: int
    amplitude_min: float
    amplitude_max: float
    duration_min: float
    duration_max: float
    noise_std: float
    num_observations: int
    observation_design: str
    design_strength: float
    peak_bandwidth: float
    overlap_floor: float
    onset_bandwidth: float
    onset_decay_timescale: float
    splits: dict[str, int]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Pharmacokinetics1DConfig":
        equation = payload["equation"]
        grid = payload["grid"]
        population = payload["population"]
        dosing = payload["dosing"]
        observations = payload["observations"]
        return cls(
            dataset_name=payload["dataset_name"],
            output_root=payload.get("output_root", "generated/datasets"),
            seed=int(payload["seed"]),
            pde_class=str(equation["pde_class"]),
            time_horizon=float(equation.get("time_horizon", 24.0)),
            num_grid_points=int(grid["num_points"]),
            mean_log_clearance=float(population["mean_log_clearance"]),
            mean_log_volume=float(population["mean_log_volume"]),
            std_log_clearance=float(population["std_log_clearance"]),
            std_log_volume=float(population["std_log_volume"]),
            corr_log_clearance_volume=float(population["corr_log_clearance_volume"]),
            pulse_family=str(dosing.get("pulse_family", "one_or_two_pulse")),
            max_num_pulses=int(dosing.get("max_num_pulses", 2)),
            amplitude_min=float(dosing["amplitude_min"]),
            amplitude_max=float(dosing["amplitude_max"]),
            duration_min=float(dosing["duration_min"]),
            duration_max=float(dosing["duration_max"]),
            noise_std=float(observations["noise_std"]),
            num_observations=int(observations["num_observations"]),
            observation_design=str(observations["design"]),
            design_strength=float(observations.get("design_strength", 0.0)),
            peak_bandwidth=float(observations.get("peak_bandwidth", 1.5)),
            overlap_floor=float(observations.get("overlap_floor", 0.0)),
            onset_bandwidth=float(observations.get("onset_bandwidth", observations.get("peak_bandwidth", 1.5))),
            onset_decay_timescale=float(
                observations.get("onset_decay_timescale", observations.get("onset_bandwidth", observations.get("peak_bandwidth", 1.5)))
            ),
            splits={str(k): int(v) for k, v in payload["splits"].items()},
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "Pharmacokinetics1DConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def dataset_root(self) -> Path:
        return Path(self.output_root) / self.dataset_name


def build_time_grid(num_points: int, time_horizon: float) -> Array:
    return np.linspace(0.0, time_horizon, num_points, dtype=np.float64)


def build_quadrature_weights(num_points: int) -> Array:
    weights = np.ones(num_points, dtype=np.float64)
    weights[0] = 0.5
    weights[-1] = 0.5
    weights /= weights.sum()
    return weights


def sample_pk_parameters(
    rng: np.random.Generator,
    config: Pharmacokinetics1DConfig,
) -> tuple[float, float, float, float]:
    cov = np.array(
        [
            [config.std_log_clearance**2, config.corr_log_clearance_volume * config.std_log_clearance * config.std_log_volume],
            [config.corr_log_clearance_volume * config.std_log_clearance * config.std_log_volume, config.std_log_volume**2],
        ],
        dtype=np.float64,
    )
    mean = np.array(
        [config.mean_log_clearance, config.mean_log_volume],
        dtype=np.float64,
    )
    log_clearance, log_volume = rng.multivariate_normal(mean=mean, cov=cov)
    return (
        float(log_clearance),
        float(log_volume),
        float(np.exp(log_clearance)),
        float(np.exp(log_volume)),
    )


def sample_dosing_profile(
    rng: np.random.Generator,
    t_grid: Array,
    config: Pharmacokinetics1DConfig,
) -> Array:
    if config.pulse_family != "one_or_two_pulse":
        raise ValueError(f"unsupported pulse_family: {config.pulse_family}")
    num_pulses = int(rng.integers(1, config.max_num_pulses + 1))
    profile = np.zeros_like(t_grid, dtype=np.float64)
    for _ in range(num_pulses):
        duration = float(rng.uniform(config.duration_min, config.duration_max))
        start_max = max(0.0, config.time_horizon - duration)
        start = float(rng.uniform(0.0, start_max))
        end = min(config.time_horizon, start + duration)
        amplitude = float(rng.uniform(config.amplitude_min, config.amplitude_max))
        mask = (t_grid >= start) & (t_grid <= end)
        profile[mask] += amplitude
    return profile


def solve_pk_trajectory(
    dosing_rate: Array,
    t_grid: Array,
    clearance: float,
    volume: float,
) -> Array:
    if t_grid.ndim != 1 or dosing_rate.shape != t_grid.shape:
        raise ValueError("dosing_rate and t_grid must have shape [G]")
    if t_grid.size < 2:
        raise ValueError("need at least two grid points")
    elimination = clearance / volume
    solution = np.zeros_like(t_grid, dtype=np.float64)
    for idx in range(1, t_grid.size):
        dt = float(t_grid[idx] - t_grid[idx - 1])
        decay = np.exp(-elimination * dt)
        drive = dosing_rate[idx - 1] / max(volume, 1e-8)
        if elimination > 1e-8:
            solution[idx] = decay * solution[idx - 1] + drive * (1.0 - decay) / elimination
        else:
            solution[idx] = solution[idx - 1] + dt * drive
    return solution


def _gaussian_peak_density(t_grid: Array, peak_time: float, bandwidth: float) -> Array:
    scaled = (t_grid - peak_time) / max(bandwidth, 1e-6)
    density = np.exp(-0.5 * (scaled**2))
    density_sum = float(density.sum())
    if density_sum <= 0.0:
        return np.full_like(t_grid, fill_value=1.0 / t_grid.size, dtype=np.float64)
    return density / density_sum


def _window_peak_density(t_grid: Array, peak_time: float, halfwidth: float) -> Array:
    window = np.abs(t_grid - peak_time) <= max(halfwidth, 1e-6)
    density = window.astype(np.float64)
    density_sum = float(density.sum())
    if density_sum <= 0.0:
        return np.full_like(t_grid, fill_value=1.0 / t_grid.size, dtype=np.float64)
    return density / density_sum


def effective_design_strength(config: Pharmacokinetics1DConfig) -> float:
    # Public rho is reported on [0, 1]; the accepted PK design keeps the
    # original behavior by using rho / 5 internally.
    return float(np.clip(config.design_strength / 5.0, 0.0, 0.999))


def sample_observations(
    rng: np.random.Generator,
    t_grid: Array,
    solution: Array,
    dosing_rate: Array,
    quad_weights: Array,
    config: Pharmacokinetics1DConfig,
) -> dict[str, Array]:
    if config.observation_design == "uniform":
        design_prob = np.full_like(t_grid, fill_value=1.0 / t_grid.size, dtype=np.float64)
    elif config.observation_design in {"peak_mixture", "peak_window_mixture"}:
        peak_index = int(np.argmax(solution))
        peak_time = float(t_grid[peak_index])
        if config.observation_design == "peak_window_mixture":
            peak_density = _window_peak_density(t_grid, peak_time, config.peak_bandwidth)
        else:
            peak_density = _gaussian_peak_density(t_grid, peak_time, config.peak_bandwidth)
        uniform = np.full_like(t_grid, fill_value=1.0 / t_grid.size, dtype=np.float64)
        gamma = effective_design_strength(config)
        design_prob = (1.0 - gamma) * uniform + gamma * peak_density
        design_prob /= design_prob.sum()
    else:
        raise ValueError(f"unsupported observation design: {config.observation_design}")

    obs_index = rng.choice(
        t_grid.size,
        size=config.num_observations,
        replace=False,
        p=design_prob,
    ).astype(np.int64)
    obs_index.sort()
    obs_t = t_grid[obs_index]
    obs_signal = solution[obs_index]
    noise = rng.normal(loc=0.0, scale=config.noise_std, size=config.num_observations)
    obs_y = obs_signal + noise

    xi = np.zeros_like(design_prob, dtype=np.float64)
    positive = design_prob > 0.0
    xi[positive] = quad_weights[positive] / design_prob[positive]
    return {
        "obs_index": obs_index,
        "obs_x": obs_t,
        "obs_signal": obs_signal,
        "obs_y": obs_y,
        "design_prob": design_prob,
        "xi": xi,
    }


def generate_split(
    config: Pharmacokinetics1DConfig,
    split_name: str,
    num_samples: int,
    rng: np.random.Generator,
) -> dict[str, Array]:
    t_grid = build_time_grid(config.num_grid_points, config.time_horizon)
    quad_weights = build_quadrature_weights(config.num_grid_points)

    coeff = np.zeros((num_samples, 3, config.num_grid_points), dtype=np.float32)
    solution = np.zeros((num_samples, config.num_grid_points), dtype=np.float32)
    obs_index = np.zeros((num_samples, config.num_observations), dtype=np.int64)
    obs_x = np.zeros((num_samples, config.num_observations), dtype=np.float32)
    obs_signal = np.zeros((num_samples, config.num_observations), dtype=np.float32)
    obs_y = np.zeros((num_samples, config.num_observations), dtype=np.float32)
    design_prob = np.zeros((num_samples, config.num_grid_points), dtype=np.float32)
    xi = np.zeros((num_samples, config.num_grid_points), dtype=np.float32)
    log_clearance = np.zeros(num_samples, dtype=np.float32)
    log_volume = np.zeros(num_samples, dtype=np.float32)
    clearance = np.zeros(num_samples, dtype=np.float32)
    volume = np.zeros(num_samples, dtype=np.float32)
    peak_time = np.zeros(num_samples, dtype=np.float32)

    for row in range(num_samples):
        log_cl, log_v, cl, v = sample_pk_parameters(rng, config)
        dosing = sample_dosing_profile(rng, t_grid, config)
        solved = solve_pk_trajectory(dosing, t_grid, cl, v)
        obs = sample_observations(rng, t_grid, solved, dosing, quad_weights, config)

        coeff[row, 0] = dosing.astype(np.float32)
        coeff[row, 1] = np.full(config.num_grid_points, log_cl, dtype=np.float32)
        coeff[row, 2] = np.full(config.num_grid_points, log_v, dtype=np.float32)
        solution[row] = solved.astype(np.float32)
        obs_index[row] = obs["obs_index"]
        obs_x[row] = obs["obs_x"].astype(np.float32)
        obs_signal[row] = obs["obs_signal"].astype(np.float32)
        obs_y[row] = obs["obs_y"].astype(np.float32)
        design_prob[row] = obs["design_prob"].astype(np.float32)
        xi[row] = obs["xi"].astype(np.float32)
        log_clearance[row] = log_cl
        log_volume[row] = log_v
        clearance[row] = cl
        volume[row] = v
        peak_time[row] = float(t_grid[int(np.argmax(solved))])

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
        "log_clearance": log_clearance,
        "log_volume": log_volume,
        "clearance": clearance,
        "volume": volume,
        "peak_time": peak_time,
        "x_grid": t_grid.astype(np.float32),
        "quad_weights": quad_weights.astype(np.float32),
    }


def dataset_metadata(config: Pharmacokinetics1DConfig) -> dict[str, Any]:
    return {
        "benchmark_name": "pharmacokinetics_1d",
        "dataset_name": config.dataset_name,
        "schema_version": 1,
        "pde_class": config.pde_class,
        "observation_design": config.observation_design,
        "seed": config.seed,
        "grid": {
            "num_points": config.num_grid_points,
            "domain": [0.0, config.time_horizon],
            "quadrature_normalization": "sum_to_one",
        },
        "equation": {
            "time_horizon": config.time_horizon,
            "dynamics": "du_dt = -(CL/V) u + (1/V) r(t)",
            "initial_condition": 0.0,
        },
        "population": {
            "distribution": "correlated_log_normal",
            "mean_log_clearance": config.mean_log_clearance,
            "mean_log_volume": config.mean_log_volume,
            "std_log_clearance": config.std_log_clearance,
            "std_log_volume": config.std_log_volume,
            "corr_log_clearance_volume": config.corr_log_clearance_volume,
        },
        "input_function": {
            "representation": "grid_values_multichannel",
            "channels": ["dosing_rate", "log_clearance", "log_volume"],
            "pulse_family": config.pulse_family,
            "max_num_pulses": config.max_num_pulses,
            "amplitude_min": config.amplitude_min,
            "amplitude_max": config.amplitude_max,
            "duration_min": config.duration_min,
            "duration_max": config.duration_max,
        },
        "observations": {
            "num_observations": config.num_observations,
            "noise_std": config.noise_std,
            "design_strength": config.design_strength,
            "design_strength_effective": effective_design_strength(config),
            "peak_bandwidth": config.peak_bandwidth,
        },
        "splits": config.splits,
        "arrays": {
            "coeff": {"dtype": "float32", "shape": "[N, 3, G]"},
            "solution": {"dtype": "float32", "shape": "[N, G]"},
            "obs_index": {"dtype": "int64", "shape": "[N, K]"},
            "obs_x": {"dtype": "float32", "shape": "[N, K]"},
            "obs_signal": {"dtype": "float32", "shape": "[N, K]"},
            "obs_y": {"dtype": "float32", "shape": "[N, K]"},
            "design_prob": {"dtype": "float32", "shape": "[N, G]"},
            "xi": {"dtype": "float32", "shape": "[N, G]"},
            "log_clearance": {"dtype": "float32", "shape": "[N]"},
            "log_volume": {"dtype": "float32", "shape": "[N]"},
            "clearance": {"dtype": "float32", "shape": "[N]"},
            "volume": {"dtype": "float32", "shape": "[N]"},
            "peak_time": {"dtype": "float32", "shape": "[N]"},
            "x_grid": {"dtype": "float32", "shape": "[G]"},
            "quad_weights": {"dtype": "float32", "shape": "[G]"},
        },
    }
