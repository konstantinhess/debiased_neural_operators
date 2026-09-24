from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import csv
import json
import math
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ono.data.dataset import GeneratedSplitDataset
from ono.data.generate import DatasetConfig, load_dataset_config, with_dataset_overrides, write_dataset
from ono.functionals import get_functional, riesz_density
from ono.models import NuisanceModelBundleConfig
from ono.models.nuisance import interpolate_from_grid
from ono.training import (
    SparseNuisanceDataset,
    set_global_seed,
    train_additional_riesz,
    train_solution_and_riesz,
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value))


def _flatten_grid(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.size(0), -1)


def _flatten_coordinates(x_grid: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x_grid.ndim == reference.ndim:
        return x_grid.reshape(x_grid.size(0), -1)
    return x_grid.reshape(x_grid.size(0), -1, x_grid.size(-1))


def functional_values(
    name: str,
    solution: torch.Tensor,
    weights: torch.Tensor,
    x_grid: torch.Tensor,
    params: dict[str, float],
) -> torch.Tensor:
    return get_functional(name)(
        _flatten_grid(solution),
        _flatten_grid(weights),
        _flatten_coordinates(x_grid, solution),
        **params,
    )


def normal_ci(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    center = float(values.mean())
    variance = (
        float(np.sum((values - center) ** 2) / (values.size * (values.size - 1)))
        if values.size > 1
        else 0.0
    )
    se = math.sqrt(max(variance, 0.0))
    half = 1.96 * se
    return {"low": center - half, "high": center + half, "length": 2.0 * half, "se": se}


def normal_ci_from_variance(center: float, variance: float) -> dict[str, float]:
    se = math.sqrt(max(variance, 0.0))
    half = 1.96 * se
    return {"low": center - half, "high": center + half, "length": 2.0 * half, "se": se}


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _rmse_delta_se(errors: list[float]) -> float:
    if len(errors) <= 1:
        return 0.0
    squares = np.square(np.asarray(errors, dtype=np.float64))
    rmse = float(np.sqrt(squares.mean()))
    if rmse == 0.0:
        return 0.0
    return float(np.std(squares, ddof=1) / (2.0 * rmse * np.sqrt(len(errors))))


def aggregate_method(rows: list[dict[str, Any]], method: str) -> dict[str, float]:
    estimates = [float(row["methods"][method]["estimate"]) for row in rows]
    errors = [float(row["methods"][method]["bias"]) for row in rows]
    lengths = [float(row["methods"][method]["ci_length"]) for row in rows]
    coverage = [float(row["methods"][method]["ci_covers"]) for row in rows]
    return {
        "estimate_mean": float(np.mean(estimates)),
        "estimate_std": _sample_std(estimates),
        "bias_mean": float(np.mean(errors)),
        "bias_std": _sample_std(errors),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "rmse_delta_se": _rmse_delta_se(errors),
        "coverage": float(np.mean(coverage)),
        "mean_interval_length": float(np.mean(lengths)),
        "interval_length_std": _sample_std(lengths),
    }


def _bundle(model_payload: dict[str, Any], epochs_override: int | None) -> NuisanceModelBundleConfig:
    bundle = NuisanceModelBundleConfig.from_dict(model_payload)
    if epochs_override is None:
        return bundle
    return replace(
        bundle,
        solution_module=replace(bundle.solution_module, max_epochs=epochs_override),
        debiasing_module=replace(bundle.debiasing_module, max_epochs=epochs_override),
    )


def _dataset_payload(
    config: dict[str, Any],
    condition: float,
    seed: int,
    output_root: Path,
    dataset_name: str,
    split_override: dict[str, int] | None,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(config["dataset"]))
    payload["dataset_name"] = dataset_name
    payload["output_root"] = str(output_root)
    payload["seed"] = int(seed)
    if split_override is not None:
        payload["splits"] = split_override
    if payload["equation"]["pde_class"] == "pharmacokinetics_1d":
        payload["observations"]["design_strength"] = float(condition)
    return payload


def _condition_seed(config: dict[str, Any], condition: float) -> int:
    mapping = config.get("seed_by_rho")
    if mapping is None:
        return int(config["dataset"]["seed"])
    key = f"{condition:g}"
    if key not in mapping:
        raise KeyError(f"seed_by_rho has no entry for rho={key}")
    return int(mapping[key])


def _truth_theta(
    dataset_config: DatasetConfig,
    functional: str,
    params: dict[str, float],
    batch_size: int,
    truth_size: int,
    seed_offset: int,
    cache_root: Path,
    name: str,
) -> float:
    truth_config = with_dataset_overrides(
        dataset_config,
        seed=int(dataset_config.seed) + seed_offset,
        output_root=str(cache_root),
        dataset_name=name,
        splits={"test": truth_size},
    )
    root = write_dataset(truth_config)
    loader = DataLoader(GeneratedSplitDataset(root, "test"), batch_size=batch_size, shuffle=False)
    terms: list[float] = []
    with torch.no_grad():
        for batch in loader:
            terms.extend(
                functional_values(
                    functional,
                    batch["solution"],
                    batch["quad_weights"],
                    batch["x_grid"],
                    params,
                ).cpu().tolist()
            )
    return float(np.mean(terms))


def _unlabeled_plugin_terms(
    dataset_config: DatasetConfig,
    solution,
    functional: str,
    params: dict[str, float],
    batch_size: int,
    size: int,
    cache_root: Path,
    name: str,
) -> list[float]:
    if size <= 0:
        return []
    pool_config = with_dataset_overrides(
        dataset_config,
        seed=int(dataset_config.seed) + 100000,
        output_root=str(cache_root),
        dataset_name=name,
        splits={"test": size},
    )
    root = write_dataset(pool_config)
    loader = DataLoader(GeneratedSplitDataset(root, "test"), batch_size=batch_size, shuffle=False)
    terms: list[float] = []
    with torch.no_grad():
        for batch in loader:
            prediction = solution.predict_grid(batch["coeff"], batch["x_grid"])
            terms.extend(
                functional_values(
                    functional, prediction, batch["quad_weights"], batch["x_grid"], params
                ).cpu().tolist()
            )
    return terms


def _method_payload(samples: list[float], truth: float) -> dict[str, float]:
    estimate = float(np.mean(samples))
    ci = normal_ci(samples)
    return {
        "estimate": estimate,
        "bias": estimate - truth,
        "ci_low": ci["low"],
        "ci_high": ci["high"],
        "ci_length": ci["length"],
        "standard_error": ci["se"],
        "ci_covers": float(ci["low"] <= truth <= ci["high"]),
    }


def _evaluate_labeled(
    test: GeneratedSplitDataset,
    solution,
    riesz,
    structured,
    functional: str,
    params: dict[str, float],
    batch_size: int,
    include_oracles: bool,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    terms: dict[str, list[float]] = {"plugin": [], "dope": []}
    if structured is not None:
        terms["structured"] = []
    if include_oracles:
        terms["oracle_beta"] = []
    correction_terms: list[float] = []
    solution_errors: list[float] = []
    beta_errors: list[float] = []

    for batch in DataLoader(test, batch_size=batch_size, shuffle=False):
        with torch.no_grad():
            s_grid = solution.predict_grid(batch["coeff"], batch["x_grid"])
            s_points = solution.predict_points(batch["coeff"], batch["obs_x"], batch["x_grid"])
            beta_grid, beta_points = riesz.compose_beta(
                batch["coeff"], batch["x_grid"], batch["obs_x"], {"u_hat": s_grid, **batch}
            )
            plugin = functional_values(
                functional, s_grid, batch["quad_weights"], batch["x_grid"], params
            )
            correction = torch.mean(beta_points * (batch["obs_y"] - s_points), dim=-1)
            terms["plugin"].extend(plugin.cpu().tolist())
            terms["dope"].extend((plugin + correction).cpu().tolist())
            correction_terms.extend(correction.cpu().tolist())

            true_beta = batch["xi"] * riesz_density(
                functional,
                _flatten_grid(batch["solution"]),
                _flatten_grid(batch["quad_weights"]),
                _flatten_coordinates(batch["x_grid"], batch["solution"]),
                **params,
            ).reshape_as(batch["xi"])
            solution_errors.extend(
                torch.sum(
                    (_flatten_grid(s_grid - batch["solution"]) ** 2)
                    * _flatten_grid(batch["quad_weights"]),
                    dim=-1,
                ).cpu().tolist()
            )
            beta_errors.extend(
                torch.sum(
                    (_flatten_grid(beta_grid - true_beta) ** 2)
                    * _flatten_grid(batch["quad_weights"]),
                    dim=-1,
                ).cpu().tolist()
            )

            if structured is not None:
                _, points = structured.compose_beta(
                    batch["coeff"], batch["x_grid"], batch["obs_x"], {"u_hat": s_grid, **batch}
                )
                terms["structured"].extend(
                    (plugin + torch.mean(points * (batch["obs_y"] - s_points), dim=-1)).cpu().tolist()
                )
            if include_oracles:
                points = interpolate_from_grid(batch["x_grid"], true_beta, batch["obs_x"])
                terms["oracle_beta"].extend(
                    (plugin + torch.mean(points * (batch["obs_y"] - s_points), dim=-1)).cpu().tolist()
                )

    return terms, {
        "correction_terms": correction_terms,
        "solution_rmse": float(np.sqrt(np.mean(solution_errors))),
        "beta_rmse": float(np.sqrt(np.mean(beta_errors))),
    }


def _ppi_payload(
    counts: list[int],
    labeled_plugin: list[float],
    labeled_correction: list[float],
    unlabeled_plugin: list[float],
    truth: float,
) -> dict[str, dict[str, float]]:
    n1 = len(labeled_plugin)
    correction_mean = float(np.mean(labeled_correction))
    result: dict[str, dict[str, float]] = {}
    for n2 in counts:
        plugin_samples = labeled_plugin + unlabeled_plugin[:n2]
        total = n1 + n2
        plugin_estimate = float(np.mean(plugin_samples))
        dope_estimate = plugin_estimate + correction_mean
        plugin_variance = float(np.var(plugin_samples, ddof=1)) if total > 1 else 0.0
        correction_variance = float(np.var(labeled_correction, ddof=1)) if n1 > 1 else 0.0
        covariance = (
            float(np.cov(labeled_plugin, labeled_correction, ddof=1)[0, 1]) if n1 > 1 else 0.0
        )
        dope_variance = plugin_variance / total + correction_variance / n1 + 2.0 * covariance / total
        plugin_ci = normal_ci(plugin_samples)
        dope_ci = normal_ci_from_variance(dope_estimate, dope_variance)
        result[str(n2)] = {
            "n1": n1,
            "n2": n2,
            "truth": truth,
            "plugin_estimate": plugin_estimate,
            "plugin_bias": plugin_estimate - truth,
            "plugin_standard_error": math.sqrt(plugin_variance / total),
            "plugin_ci_length": plugin_ci["length"],
            "plugin_ci_covers": float(plugin_ci["low"] <= truth <= plugin_ci["high"]),
            "dope_estimate": dope_estimate,
            "dope_bias": dope_estimate - truth,
            "dope_standard_error": math.sqrt(max(dope_variance, 0.0)),
            "dope_ci_length": dope_ci["length"],
            "dope_ci_covers": float(dope_ci["low"] <= truth <= dope_ci["high"]),
            "plugin_correction_covariance": covariance,
        }
    return result


def run_repeat(
    config: dict[str, Any],
    condition: float,
    repeat: int,
    output_root: Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    base_seed = _condition_seed(config, condition)
    dataset_seed = base_seed + int(config["repeat_seed_offset"]) + repeat
    training_seed = int(config["training_seed"])
    set_global_seed(training_seed)

    split_override = None
    if any(overrides.get(key) is not None for key in ("train_size", "val_size", "test_size")):
        original = config["dataset"]["splits"]
        split_override = {
            "train": int(overrides.get("train_size") or original["train"]),
            "val": int(overrides.get("val_size") or original["val"]),
            "test": int(overrides.get("test_size") or original["test"]),
        }
    name = f"{config['name']}_condition_{condition:g}_repeat_{repeat}"
    cache_root = output_root / "_generated"
    payload = _dataset_payload(config, condition, dataset_seed, cache_root, name, split_override)
    dataset_config = load_dataset_config(payload)
    root = write_dataset(dataset_config)
    train_raw = GeneratedSplitDataset(root, "train")
    val_raw = GeneratedSplitDataset(root, "val")
    test = GeneratedSplitDataset(root, "test")
    train = SparseNuisanceDataset(train_raw)
    val = SparseNuisanceDataset(val_raw)

    bundle = _bundle(config["model"], overrides.get("epochs"))
    functional = str(config["functional"])
    params = dict(config.get("functional_params", {}))
    if config["experiment"] == "darcy":
        params["kappa"] = float(condition)
    solution, riesz, selections = train_solution_and_riesz(
        train,
        val,
        bundle,
        functional,
        params,
        int(config["batch_size"]),
        str(config.get("debiasing_mode", "full")),
    )
    structured = None
    if "structured" in config["methods"]:
        structured, selections["structured"] = train_additional_riesz(
            "structured",
            train,
            val,
            solution,
            bundle,
            functional,
            params,
            int(config["batch_size"]),
        )
    terms, diagnostics = _evaluate_labeled(
        test,
        solution,
        riesz,
        structured,
        functional,
        params,
        int(config["batch_size"]),
        include_oracles="oracle_beta" in config["methods"],
    )

    truth_size = int(overrides.get("truth_size") or config["truth_pool_size"])
    truth_seed_offset = 200000 if config["experiment"] == "pk_ppi" else 300000
    truth = _truth_theta(
        dataset_config,
        functional,
        params,
        int(config["batch_size"]),
        truth_size,
        truth_seed_offset,
        cache_root,
        f"{name}_truth",
    )
    methods = {method: _method_payload(samples, truth) for method, samples in terms.items()}
    row: dict[str, Any] = {
        "condition": condition,
        "repeat": repeat,
        "dataset_seed": dataset_seed,
        "training_seed": training_seed,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "truth": truth,
        "truth_pool_size": truth_size,
        "methods": methods,
        "nuisance": {
            "solution_rmse": diagnostics["solution_rmse"],
            "beta_rmse": diagnostics["beta_rmse"],
        },
        "selection": selections,
    }
    if config["experiment"] == "pk_ppi":
        counts = [int(value) for value in config["unlabeled_counts"]]
        unlabeled = _unlabeled_plugin_terms(
            dataset_config,
            solution,
            functional,
            params,
            int(config["batch_size"]),
            max(counts),
            cache_root,
            f"{name}_unlabeled",
        )
        row["ppi"] = _ppi_payload(
            counts,
            terms["plugin"],
            diagnostics["correction_terms"],
            unlabeled,
            truth,
        )
    return row


def _aggregate_ppi(rows: list[dict[str, Any]], counts: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for n2 in counts:
        key = str(n2)
        entry: dict[str, Any] = {"n1": rows[0]["ppi"][key]["n1"], "n2": n2}
        for method in ("plugin", "dope"):
            estimates = [float(row["ppi"][key][f"{method}_estimate"]) for row in rows]
            errors = [float(row["ppi"][key][f"{method}_bias"]) for row in rows]
            ses = [float(row["ppi"][key][f"{method}_standard_error"]) for row in rows]
            lengths = [float(row["ppi"][key][f"{method}_ci_length"]) for row in rows]
            covers = [float(row["ppi"][key][f"{method}_ci_covers"]) for row in rows]
            entry[method] = {
                "estimate_mean": float(np.mean(estimates)),
                "estimate_std": _sample_std(estimates),
                "bias_mean": float(np.mean(errors)),
                "bias_std": _sample_std(errors),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "rmse_delta_se": _rmse_delta_se(errors),
                "standard_error_mean": float(np.mean(ses)),
                "coverage": float(np.mean(covers)),
                "mean_interval_length": float(np.mean(lengths)),
            }
        summary[key] = entry
    return summary


def _write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat: list[dict[str, Any]] = []
    for row in rows:
        if "ppi" in row:
            for n2, values in row["ppi"].items():
                flat.append({
                    "condition": row["condition"],
                    "repeat": row["repeat"],
                    "dataset_seed": row["dataset_seed"],
                    "n2": int(n2),
                    **{key: value for key, value in values.items() if isinstance(value, (int, float))},
                })
        else:
            for method, values in row["methods"].items():
                flat.append({
                    "condition": row["condition"],
                    "repeat": row["repeat"],
                    "dataset_seed": row["dataset_seed"],
                    "method": method,
                    **values,
                })
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in flat for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def _write_table_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    if "ppi" in summary:
        for n2, entry in summary["ppi"].items():
            for method in ("plugin", "dope"):
                rows.append({"n1": entry["n1"], "n2": int(n2), "method": method, **entry[method]})
    else:
        for condition, entry in summary["conditions"].items():
            for method, metrics in entry["methods"].items():
                rows.append({"condition": float(condition), "method": method, **metrics})
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_study(
    config_path: str | Path,
    *,
    repeats: int | None = None,
    epochs: int | None = None,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    truth_size: int | None = None,
    conditions: list[float] | None = None,
    unlabeled_counts: list[int] | None = None,
    output_root: str | Path | None = None,
) -> Path:
    config_path = Path(config_path).resolve()
    config = load_json(config_path)
    if unlabeled_counts is not None:
        config["unlabeled_counts"] = [int(value) for value in unlabeled_counts]
    selected_conditions = conditions or [float(value) for value in config["conditions"]]
    num_repeats = int(repeats or config["repeats"])
    base_output = Path(output_root) if output_root is not None else config_path.parents[1] / config["output_root"]
    destination = base_output / config["name"]
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / "resolved_config.json", config)
    overrides = {
        "epochs": epochs,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "truth_size": truth_size,
    }
    rows: list[dict[str, Any]] = []
    for condition in selected_conditions:
        for repeat in range(num_repeats):
            row = run_repeat(config, condition, repeat, destination, overrides)
            rows.append(row)
            save_json(destination / "repeats" / f"condition_{condition:g}_repeat_{repeat}.json", row)
            save_json(destination / "raw_repeats.json", rows)
            _write_flat_csv(destination / "raw_repeats.csv", rows)

    if config["experiment"] == "pk_ppi":
        summary = {
            "experiment": config["name"],
            "num_repeats": num_repeats,
            "training_protocol": "honest_train_val_test",
            "ppi": _aggregate_ppi(rows, [int(value) for value in config["unlabeled_counts"]]),
        }
    else:
        condition_summary: dict[str, Any] = {}
        for condition in selected_conditions:
            selected = [row for row in rows if float(row["condition"]) == condition]
            condition_summary[f"{condition:g}"] = {
                "num_repeats": len(selected),
                "target_mean": float(np.mean([row["truth"] for row in selected])),
                "target_std": _sample_std([row["truth"] for row in selected]),
                "methods": {
                    method: aggregate_method(selected, method)
                    for method in selected[0]["methods"]
                },
            }
        summary = {
            "experiment": config["name"],
            "num_repeats": num_repeats,
            "training_protocol": "honest_train_val_test",
            "conditions": condition_summary,
        }
    save_json(destination / "summary.json", summary)
    _write_table_csv(destination / "table.csv", summary)
    return destination
