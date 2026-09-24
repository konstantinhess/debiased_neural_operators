from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from ono.data.generate import load_dataset_config
from ono.data.pharmacokinetics_1d import generate_split
from ono.functionals import get_functional, riesz_density


PATHS = ("solution_only", "joint")


def _functional(name: str, solution: torch.Tensor, weights: torch.Tensor, x_grid: torch.Tensor, params: dict) -> torch.Tensor:
    return get_functional(name)(solution, weights, x_grid, **params)


def _directions(x_grid: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    time = x_grid / x_grid[-1]
    h_raw = torch.ones_like(time)
    b_raw = -(1.0 + 0.25 * torch.cos(2.0 * torch.pi * time))
    h = h_raw / torch.sqrt(torch.sum(weights * h_raw.square()))
    b = b_raw / torch.sqrt(torch.sum(weights * b_raw.square()))
    return h, b


def _draw(config, seed: int, size: int) -> dict[str, np.ndarray]:
    return generate_split(config, "test", size, np.random.default_rng(seed))


def _reference(config, functional: str, params: dict, seed: int, size: int) -> tuple[float, float]:
    samples: list[np.ndarray] = []
    for offset in range(0, size, 512):
        count = min(512, size - offset)
        batch = _draw(config, seed + offset, count)
        samples.append(
            _functional(
                functional,
                torch.from_numpy(batch["solution"]),
                torch.from_numpy(batch["quad_weights"]).expand(count, -1),
                torch.from_numpy(batch["x_grid"]).expand(count, -1),
                params,
            ).numpy()
        )
    values = np.concatenate(samples)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def _repeat(config, functional: str, params: dict, deltas: list[float], seed: int, size: int) -> list[dict]:
    batch = _draw(config, seed, size)
    s0 = torch.from_numpy(batch["solution"])
    weights = torch.from_numpy(batch["quad_weights"]).expand(size, -1)
    x_grid = torch.from_numpy(batch["x_grid"]).expand(size, -1)
    obs_index = torch.from_numpy(batch["obs_index"]).long()
    obs_y = torch.from_numpy(batch["obs_y"])
    xi = torch.from_numpy(batch["xi"])
    h, b = _directions(x_grid[0], weights[0])
    h_batch, b_batch = h.expand_as(s0), b.expand_as(s0)
    beta0 = xi * riesz_density(functional, s0, weights, x_grid, **params)
    rows: list[dict] = []
    for path in PATHS:
        for delta in deltas:
            s_delta = s0 + delta * h_batch
            beta_delta = beta0 + (delta * b_batch if path == "joint" else 0.0)
            plugin = _functional(functional, s_delta, weights, x_grid, params)
            s_points = torch.gather(s_delta, 1, obs_index)
            beta_points = torch.gather(beta_delta, 1, obs_index)
            dope = plugin + torch.mean(beta_points * (obs_y - s_points), dim=1)
            row = {
                "path": path,
                "delta": float(delta),
                "dope_estimate": float(dope.mean()),
                "h_empirical_l2": float(torch.sqrt(torch.mean(torch.sum(h_batch.square() * weights, dim=1)))),
                "b_empirical_l2": float(torch.sqrt(torch.mean(torch.sum(b_batch.square() * weights, dim=1)))),
            }
            if path == "solution_only":
                row["plugin_estimate"] = float(plugin.mean())
            rows.append(row)
    return rows


def _bootstrap_abs_se(values: np.ndarray, seed: int, draws: int = 2000) -> float:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    return float(np.std(np.abs(values[indices].mean(axis=1)), ddof=1))


def _aggregate(raw: list[dict], theta: float, theta_se: float, seed: int) -> list[dict]:
    output: list[dict] = []
    pairs = (("solution_only", "plugin"), ("solution_only", "dope"), ("joint", "dope"))
    for path, estimator in pairs:
        for delta in sorted({float(row["delta"]) for row in raw}):
            selected = [row for row in raw if row["path"] == path and float(row["delta"]) == delta]
            errors = np.asarray([float(row[f"{estimator}_estimate"]) - theta for row in selected])
            estimator_se = float(errors.std(ddof=1) / np.sqrt(errors.size))
            bias_se = float(np.sqrt(estimator_se**2 + theta_se**2))
            output.append({
                "path": path,
                "estimator": estimator,
                "delta": delta,
                "num_repeats": int(errors.size),
                "signed_bias": float(errors.mean()),
                "absolute_bias": float(abs(errors.mean())),
                "estimator_mc_se": estimator_se,
                "target_mc_se": theta_se,
                "bias_mc_se": bias_se,
                "absolute_bias_mc_se": _bootstrap_abs_se(errors, seed + len(output)),
                "noise_dominated": bool(abs(errors.mean()) <= 2.0 * bias_se),
            })
    return output


def _slope_rows(summary: list[dict], raw: list[dict], theta: float, theta_se: float) -> list[dict]:
    output: list[dict] = []
    for path, estimator in (("solution_only", "plugin"), ("solution_only", "dope"), ("joint", "dope")):
        candidates = sorted(
            [
                row for row in summary
                if row["path"] == path and row["estimator"] == estimator
                and row["delta"] > 0 and not row["noise_dominated"] and row["absolute_bias"] > 0
            ],
            key=lambda row: row["delta"],
        )
        result: dict[str, Any] = {
            "path": path,
            "estimator": estimator,
            "points_used": [row["delta"] for row in candidates],
            "slope": None,
            "slope_delta_se": None,
            "slope_ci95_lower": None,
            "slope_ci95_upper": None,
        }
        if len(candidates) >= 2:
            deltas = np.asarray(result["points_used"], dtype=float)
            biases = np.asarray([row["absolute_bias"] for row in candidates])
            slope = float(np.polyfit(np.log(deltas), np.log(biases), 1)[0])
            repeat_ids = sorted({int(row["repeat"]) for row in raw})
            matrix = np.asarray([
                [
                    next(
                        float(row[f"{estimator}_estimate"]) - theta
                        for row in raw
                        if int(row["repeat"]) == repeat and row["path"] == path and float(row["delta"]) == delta
                    )
                    for delta in deltas
                ]
                for repeat in repeat_ids
            ])
            signed_bias = matrix.mean(axis=0)
            x = np.log(deltas)
            weights = (x - x.mean()) / np.sum((x - x.mean()) ** 2)
            gradient = weights / signed_bias
            covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1) / matrix.shape[0])
            covariance += theta_se**2 * np.ones((len(deltas), len(deltas)))
            slope_se = float(np.sqrt(max(float(gradient @ covariance @ gradient), 0.0)))
            result.update({
                "slope": slope,
                "slope_delta_se": slope_se,
                "slope_ci95_lower": slope - 1.96 * slope_se,
                "slope_ci95_upper": slope + 1.96 * slope_se,
            })
        output.append(result)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _series(summary: list[dict], path: str, estimator: str) -> tuple[np.ndarray, ...]:
    rows = sorted(
        [row for row in summary if row["path"] == path and row["estimator"] == estimator],
        key=lambda row: row["delta"],
    )
    return tuple(np.asarray([float(row[key]) for row in rows]) for key in ("delta", "absolute_bias", "absolute_bias_mc_se"))


def _plot(summary: list[dict], output: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    specs = [
        ("solution_only", "plugin", "Plug-in, $S$ only", "#404040", "o"),
        ("solution_only", "dope", "DOPE, $S$ only", "#0072B2", "s"),
        ("joint", "dope", "DOPE, joint", "#C62828", "^"),
    ]
    fig, axis = plt.subplots(figsize=(5.15, 3.85), constrained_layout=True)
    for path, estimator, label, color, marker in specs:
        delta, bias, se = _series(summary, path, estimator)
        axis.errorbar(delta, bias, yerr=1.96 * se, color=color, marker=marker, markersize=6.5, linestyle="none", capsize=2.5)
        axis.plot(delta, bias, color=color, linewidth=1.25, linestyle=":", alpha=0.7, label=label)
        reliable = np.asarray([
            not row["noise_dominated"]
            for row in sorted(
                [r for r in summary if r["path"] == path and r["estimator"] == estimator],
                key=lambda r: r["delta"],
            )
        ]) & (delta > 0)
        if reliable.sum() >= 2:
            slope, intercept = np.polyfit(np.log(delta[reliable]), np.log(bias[reliable]), 1)
            xfit = np.linspace(0, delta[reliable].max(), 200)
            axis.plot(xfit, np.exp(intercept) * xfit**slope, color=color, linewidth=1.6)
    axis.set_xlabel("Perturbation magnitude $\\delta$")
    axis.set_ylabel("Absolute bias")
    axis.grid(True, color="#d9d9d9", linewidth=0.6)
    axis.legend(loc="upper left", frameon=True, handlelength=2.5)
    fig.savefig(output / "smooth_tat_oracle_centered_bias.png", dpi=400, bbox_inches="tight")
    fig.savefig(output / "smooth_tat_oracle_centered_bias.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.15, 3.85), constrained_layout=True)
    for path, estimator, label, color, marker in specs:
        delta, bias, se = _series(summary, path, estimator)
        keep = delta > 0
        axis.errorbar(delta[keep], bias[keep], yerr=1.96 * se[keep], color=color, marker=marker, linestyle=":", capsize=2.5, label=label)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Perturbation magnitude $\\delta$")
    axis.set_ylabel("Absolute bias")
    axis.grid(True, color="#d9d9d9", linewidth=0.6)
    axis.legend(frameon=True)
    fig.savefig(output / "smooth_tat_oracle_centered_loglog.png", dpi=400, bbox_inches="tight")
    fig.savefig(output / "smooth_tat_oracle_centered_loglog.pdf", bbox_inches="tight")
    plt.close(fig)


def run_oracle_study(
    config_path: str | Path,
    *,
    repeats: int | None = None,
    test_size: int | None = None,
    truth_size: int | None = None,
    output_root: str | Path | None = None,
) -> Path:
    config_path = Path(config_path).resolve()
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = load_dataset_config(config_payload["dataset"])
    functional = str(config_payload["functional"])
    params = dict(config_payload["functional_params"])
    deltas = sorted({float(value) for value in config_payload["deltas"]})
    if len(deltas) > 6 or 0.0 not in deltas or any(value < 0 for value in deltas):
        raise ValueError("oracle perturbation config must contain 0 and at most six nonnegative deltas")
    num_repeats = int(repeats or config_payload["repeats"])
    n_test = int(test_size or config_payload["test_size"])
    n_truth = int(truth_size or config_payload["truth_size"])
    seed = int(config_payload["seed"])
    base_output = Path(output_root) if output_root is not None else config_path.parents[1] / config_payload["output_root"]
    output = base_output / config_payload["name"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    theta, theta_se = _reference(dataset, functional, params, seed, n_truth)
    raw: list[dict] = []
    for repeat in range(num_repeats):
        rows = _repeat(dataset, functional, params, deltas, seed + repeat, n_test)
        for row in rows:
            row["repeat"] = repeat
        raw.extend(rows)
    summary = _aggregate(raw, theta, theta_se, seed)
    slopes = _slope_rows(summary, raw, theta, theta_se)
    payload = {
        "experiment": config_payload["name"],
        "reference": {"theta": theta, "mc_se": theta_se, "size": n_truth},
        "test_size": n_test,
        "num_repeats": num_repeats,
        "summary": summary,
        "slopes": slopes,
    }
    (output / "raw_results.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output / "raw_results.csv", raw)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "local_slopes.csv", slopes)
    _plot(summary, output)
    return output
