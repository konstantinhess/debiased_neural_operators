from __future__ import annotations

from pathlib import Path
import csv
import json
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ono.functionals import get_functional
from ono.models import NuisanceModelBundleConfig
from ono.training import SparseNuisanceDataset, set_global_seed, train_solution_and_riesz


HISTORY = 96
HORIZON = 24
STRIDE = 24


def load_etth1(path: str | Path, splits: dict[str, int]) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"ETTh1 was not found at {source}. Download ETTh1.csv and place it at paper/data/ETTh1.csv."
        )
    frame = pd.read_csv(source)
    if "date" not in frame.columns:
        raise ValueError("ETTh1.csv must contain a date column")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)
    channels = [column for column in frame.columns if column != "date"]
    if len(channels) != 7 or "OT" not in channels:
        raise ValueError(f"expected seven numerical channels including OT, got {channels}")
    values = frame[channels].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("ETTh1 contains missing or non-finite values")
    num_samples = sum(splits.values())
    required = HISTORY + HORIZON + (num_samples - 1) * STRIDE
    if len(values) < required:
        raise ValueError(f"ETTh1.csv needs at least {required} chronologically ordered rows")

    histories, futures = [], []
    for sample in range(num_samples):
        start = sample * STRIDE
        histories.append(values[start : start + HISTORY])
        futures.append(values[start + HISTORY : start + HISTORY + HORIZON, channels.index("OT")])
    history = np.stack(histories)
    future = np.stack(futures)
    train_history = history[: splits["train"]]
    train_future = future[: splits["train"]]
    input_mean = train_history.reshape(-1, 7).mean(axis=0)
    input_std = np.maximum(train_history.reshape(-1, 7).std(axis=0), 1e-6)
    output_mean = float(train_future.mean())
    output_std = float(max(train_future.std(), 1e-6))
    history_z = (history - input_mean[None, None, :]) / input_std[None, None, :]
    future_z = (future - output_mean) / output_std
    coeff = history_z.reshape(num_samples, 4, HORIZON, 7).transpose(0, 1, 3, 2).reshape(num_samples, 28, HORIZON)
    indices: dict[str, slice] = {}
    cursor = 0
    for name, size in splits.items():
        indices[name] = slice(cursor, cursor + size)
        cursor += size
    return {
        "coeff": coeff.astype(np.float32),
        "future_standardized": future_z.astype(np.float32),
        "previous_day_ot_standardized": history_z[:, -HORIZON:, channels.index("OT")].astype(np.float32),
        "split_indices": indices,
        "output_mean": output_mean,
        "output_std": output_std,
        "channels": channels,
    }


def early_block_observations(
    future: np.ndarray,
    k: int,
    rho: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 1 <= k <= HORIZON:
        raise ValueError("K must be in [1, 24]")
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1]")
    q = np.full((future.shape[0], HORIZON), 0.02 / HORIZON, dtype=np.float64)
    q[:, :8] += 0.98 / 8.0
    probabilities = (1.0 - rho) / HORIZON + rho * q
    rng = np.random.default_rng(seed)
    indices = np.stack(
        [rng.choice(HORIZON, size=k, replace=False, p=probability) for probability in probabilities]
    ).astype(np.int64)
    indices.sort(axis=1)
    observations = np.take_along_axis(future, indices, axis=1)
    return indices, observations.astype(np.float32), probabilities.astype(np.float32)


class SparseETTh1Dataset(Dataset):
    """Only sparse future observations are returned; full futures remain inaccessible."""

    def __init__(self, coeff: np.ndarray, future: np.ndarray, k: int, rho: float, seed: int) -> None:
        indices, observations, probabilities = early_block_observations(future, k, rho, seed)
        self.coeff = torch.from_numpy(coeff)
        self.obs_index = torch.from_numpy(indices)
        self.obs_y = torch.from_numpy(observations)
        self.design_prob = torch.from_numpy(probabilities)
        self.x_grid = torch.linspace(0.0, 1.0, HORIZON)
        self.weights = torch.full((HORIZON,), 1.0 / HORIZON)

    def __len__(self) -> int:
        return int(self.coeff.size(0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        locations = self.obs_index[index]
        return {
            "coeff": self.coeff[index],
            "x_grid": self.x_grid,
            "obs_x": self.x_grid[locations],
            "obs_y": self.obs_y[index],
            "obs_index": locations,
            "quad_weights": self.weights,
            "design_prob": self.design_prob[index],
        }


def _functional(name: str, values: torch.Tensor, lam: float) -> torch.Tensor:
    weights = torch.full_like(values, 1.0 / HORIZON)
    if name == "mean":
        return get_functional("average")(values, weights)
    if name == "soft_cmax":
        return get_functional("soft_cmax")(values, weights, lam=lam)
    raise ValueError(name)


def _repeat(data: dict, config: dict, rho: float, repeat: int, epochs: int | None) -> dict:
    seed = int(config["seed"])
    set_global_seed(seed + repeat)
    split = data["split_indices"]
    datasets: dict[str, SparseETTh1Dataset] = {}
    for offset, name in enumerate(("train", "val", "test")):
        selection = split[name]
        datasets[name] = SparseETTh1Dataset(
            data["coeff"][selection],
            data["future_standardized"][selection],
            int(config["k"]),
            rho,
            seed + 10000 * repeat + offset,
        )

    model_payload = json.loads(json.dumps(config["model"]))
    if epochs is not None:
        model_payload["solution_module"]["max_epochs"] = epochs
        model_payload["debiasing_module"]["max_epochs"] = epochs
    bundle = NuisanceModelBundleConfig.from_dict(model_payload)
    functional = "average" if config["functional"] == "mean" else "soft_cmax"
    lam = 1.0 / float(config.get("soft_cmax_temperature", 0.5))
    params = {} if functional == "average" else {"lam": lam}
    solution, riesz, selections = train_solution_and_riesz(
        SparseNuisanceDataset(datasets["train"]),
        SparseNuisanceDataset(datasets["val"]),
        bundle,
        functional,
        params,
        int(config["batch_size"]),
        "full",
    )

    plugin_terms: list[float] = []
    dope_terms: list[float] = []
    with torch.no_grad():
        for batch in DataLoader(datasets["test"], batch_size=int(config["batch_size"]), shuffle=False):
            prediction = solution.predict_grid(batch["coeff"], batch["x_grid"])
            _, beta_points = riesz.compose_beta(
                batch["coeff"], batch["x_grid"], batch["obs_x"], {"u_hat": prediction, **batch}
            )
            plugin = _functional(config["functional"], prediction, lam)
            prediction_points = solution.predict_points(batch["coeff"], batch["obs_x"], batch["x_grid"])
            dope = plugin + torch.mean(beta_points * (batch["obs_y"] - prediction_points), dim=1)
            plugin_terms.extend(plugin.cpu().tolist())
            dope_terms.extend(dope.cpu().tolist())

    future_test = torch.from_numpy(data["future_standardized"][split["test"]])
    target_z = float(_functional(config["functional"], future_test, lam).mean())
    plugin_z, dope_z = float(np.mean(plugin_terms)), float(np.mean(dope_terms))
    scale, location = float(data["output_std"]), float(data["output_mean"])
    target = scale * target_z + location
    plugin = scale * plugin_z + location
    dope = scale * dope_z + location
    return {
        "repeat": repeat,
        "training_seed": seed + repeat,
        "observation_seeds": {
            "train": seed + 10000 * repeat,
            "val": seed + 10000 * repeat + 1,
            "test": seed + 10000 * repeat + 2,
        },
        "target": target,
        "plugin_estimate": plugin,
        "dope_estimate": dope,
        "plugin_bias": plugin - target,
        "dope_bias": dope - target,
        "selection": selections,
    }


def _mc_se(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def _rmse_se(errors: np.ndarray) -> float:
    if errors.size <= 1:
        return 0.0
    rmse = float(np.sqrt(np.mean(errors**2)))
    return 0.0 if rmse == 0 else float(np.std(errors**2, ddof=1) / (2 * rmse * np.sqrt(errors.size)))


def _summarize(rows: list[dict]) -> dict:
    target = np.asarray([row["target"] for row in rows])
    summary: dict[str, Any] = {
        "num_repeats": len(rows),
        "target_mean": float(target.mean()),
        "target_mc_se": _mc_se(target),
        "methods": {},
    }
    for method in ("plugin", "dope"):
        estimates = np.asarray([row[f"{method}_estimate"] for row in rows])
        errors = np.asarray([row[f"{method}_bias"] for row in rows])
        summary["methods"][method] = {
            "estimate_mean": float(estimates.mean()),
            "estimate_mc_se": _mc_se(estimates),
            "bias_mean": float(errors.mean()),
            "bias_mc_se": _mc_se(errors),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "rmse_delta_se": _rmse_se(errors),
        }
    plugin_rmse = summary["methods"]["plugin"]["rmse"]
    summary["relative_rmse_improvement_dope_pct"] = (
        100 * (1 - summary["methods"]["dope"]["rmse"] / plugin_rmse) if plugin_rmse else 0.0
    )
    return summary


def _write_rows(path: Path, rows: list[dict]) -> None:
    scalar_rows = [
        {key: value for key, value in row.items() if not isinstance(value, dict)}
        for row in rows
    ]
    fields = sorted({key for row in scalar_rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)


def run_etth1_study(
    config_path: str | Path,
    *,
    data_path: str | Path | None = None,
    repeats: int | None = None,
    epochs: int | None = None,
    rhos: list[float] | None = None,
    output_root: str | Path | None = None,
) -> Path:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = {key: int(value) for key, value in config["splits"].items()}
    configured_data = config_path.parents[1] / config["data_path"]
    data = load_etth1(data_path or configured_data, splits)
    selected_rhos = rhos or [float(value) for value in config["rho_grid"]]
    num_repeats = int(repeats or config["repeats"])
    base_output = Path(output_root) if output_root is not None else config_path.parents[1] / config["output_root"]
    output = base_output / config["name"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summaries: dict[str, Any] = {}
    table_rows: list[dict] = []
    for rho in selected_rhos:
        rho_root = output / f"rho_{rho:g}"
        rho_root.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for repeat in range(num_repeats):
            row = _repeat(data, config, rho, repeat, epochs)
            rows.append(row)
            (rho_root / "partial_repeats.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            _write_rows(rho_root / "raw_repeats.csv", rows)
        test_slice = data["split_indices"]["test"]
        for repeat in range(num_repeats):
            locations, _, probabilities = early_block_observations(
                data["future_standardized"][test_slice],
                int(config["k"]),
                rho,
                int(config["seed"]) + 10000 * repeat + 2,
            )
            np.savez_compressed(
                rho_root / f"test_sparse_design_repeat_{repeat}.npz",
                obs_index=locations,
                design_prob=probabilities,
            )
        summary = _summarize(rows)
        summaries[f"{rho:g}"] = summary
        (rho_root / "summary.json").write_text(
            json.dumps({"rho": rho, "summary": summary, "repeats": rows}, indent=2),
            encoding="utf-8",
        )
        for method, metrics in summary["methods"].items():
            table_rows.append({"rho": rho, "method": method, **metrics})
    payload = {
        "experiment": config["name"],
        "sampling_design": "early_block_without_replacement",
        "splits": splits,
        "num_repeats": num_repeats,
        "rho": summaries,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = sorted({key for row in table_rows for key in row})
    with (output / "table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)
    return output
