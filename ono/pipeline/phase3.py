from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import random
from tempfile import TemporaryDirectory

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from ono.data.generate import load_dataset_config, write_dataset
from ono.data.dataset import GeneratedSplitDataset
from ono.functionals import get_functional, riesz_density
from ono.models import (
    DebiasingOperatorModule,
    NuisanceModelBundleConfig,
    SolutionOperatorModule,
)
from ono.models.nuisance import interpolate_from_grid
from ono.pipeline.config import Phase3Config
from ono.tracking import MlflowTracker


class DebiasingTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, base_subset: Subset, solution_module: SolutionOperatorModule) -> None:
        self.base_subset = base_subset
        self.solution_module = solution_module
        self.solution_module.freeze_model()

    def __len__(self) -> int:
        return len(self.base_subset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base_subset[index]
        coeff = sample["coeff"].unsqueeze(0)
        x_grid = sample["x_grid"].unsqueeze(0)
        with torch.no_grad():
            u_hat = self.solution_module.predict_grid(coeff, x_grid).squeeze(0)
        return {
            **sample,
            "u_hat": u_hat,
        }


def _to_python(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_to_python),
        encoding="utf-8",
    )


def _resolve_generated_dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    if root.exists():
        return root
    # PK dataset roots were renamed in configs without renaming the existing generated-data directories.
    root_str = str(root)
    if "pharmacokinetics_1d_pk_" in root_str and "pharmacokinetics_1d_pk_irreg_unit_" not in root_str:
        legacy = Path(root_str.replace("pharmacokinetics_1d_pk_", "pharmacokinetics_1d_pk_irreg_unit_"))
        if legacy.exists():
            return legacy
    return root


def _flatten_batch_coordinates(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if values.ndim == reference.ndim:
        return values.reshape(values.size(0), -1)
    if values.ndim == reference.ndim + 1:
        return values.reshape(values.size(0), -1, values.size(-1))
    raise ValueError("x_grid shape is incompatible with the reference grid")


def _functional_values(
    functional_name: str,
    u_grid: torch.Tensor,
    quad_weights: torch.Tensor,
    x_grid: torch.Tensor,
    functional_params: dict[str, float] | None = None,
) -> torch.Tensor:
    functional = get_functional(functional_name)
    return functional(
        _flatten_batch_grid(u_grid),
        _flatten_batch_grid(quad_weights),
        _flatten_batch_coordinates(x_grid, u_grid),
        **(functional_params or {}),
    )


def _weighted_l2_error(pred: torch.Tensor, truth: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    pred_flat = _flatten_batch_grid(pred)
    truth_flat = _flatten_batch_grid(truth)
    weights_flat = _flatten_batch_grid(weights)
    return torch.sum(((pred_flat - truth_flat) ** 2) * weights_flat, dim=-1)


def _flatten_batch_grid(values: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("values must have batch dimension")
    return values.reshape(values.size(0), -1)


def _normal_ci(samples: list[float]) -> tuple[float, float, float]:
    array = np.array(samples, dtype=np.float64)
    center = float(np.mean(array))
    if array.size <= 1:
        se = 0.0
    else:
        centered = array - center
        variance_hat = float(np.sum(centered ** 2) / (array.size ** 2))
        se = float(np.sqrt(variance_hat))
    half_width = 1.96 * se
    return center - half_width, center + half_width, 2.0 * half_width


def _apply_solution_bias(
    u_grid: torch.Tensor,
    true_solution: torch.Tensor,
    level: float,
) -> torch.Tensor:
    if level == 0.0:
        return u_grid
    return u_grid + level * (u_grid - true_solution)


def _apply_beta_bias(
    beta_grid: torch.Tensor,
    true_beta: torch.Tensor,
    level: float,
) -> torch.Tensor:
    if level == 0.0:
        return beta_grid
    return beta_grid + level * (beta_grid - true_beta)


def _scalar_error_metrics(estimate: float, truth: float) -> dict[str, float]:
    error = float(estimate - truth)
    return {
        "estimate": float(estimate),
        "bias": error,
        "rmse": abs(error),
    }


def _single_run_estimator_summary(
    estimate: float,
    bias: float,
    rmse: float,
    ci_length: float,
    ci_covers_true_theta: bool,
) -> dict[str, float | None]:
    return {
        "estimate_mean": float(estimate),
        "estimate_std": None,
        "bias_mean": float(bias),
        "bias_std": None,
        "rmse_mean": float(rmse),
        "rmse_std": None,
        "ci_coverage": float(ci_covers_true_theta),
        "ci_length_mean": float(ci_length),
        "ci_length_std": None,
    }


def _single_run_summary(
    config: Phase3Config,
    resolved_spec: dict,
    num_samples: int,
    metrics: dict,
) -> dict:
    summary = {
        "pipeline_config": resolved_spec["pipeline_config"],
        "model_config": resolved_spec["model_config"],
        "dataset_metadata": resolved_spec["dataset_metadata"],
        "num_repeats": 1,
        "num_samples_per_run": int(num_samples),
        "num_folds": int(config.num_folds),
        "metrics": {
            "target": {
                "true_theta_mean": float(metrics["true_theta"]),
                "true_theta_std": None,
                "target_source": str(metrics["target_source"]),
                "target_pool_size": (
                    int(metrics["target_pool_size"])
                    if metrics.get("target_pool_size") is not None
                    else None
                ),
            },
            "estimators": {
                "plugin": _single_run_estimator_summary(
                    estimate=metrics["plugin_estimate"],
                    bias=metrics["plugin_bias"],
                    rmse=metrics["plugin_rmse"],
                    ci_length=metrics["plugin_ci_length"],
                    ci_covers_true_theta=metrics["plugin_ci_covers_true_theta"],
                ),
                "onestep": _single_run_estimator_summary(
                    estimate=metrics["onestep_estimate"],
                    bias=metrics["onestep_bias"],
                    rmse=metrics["onestep_rmse"],
                    ci_length=metrics["onestep_ci_length"],
                    ci_covers_true_theta=metrics["onestep_ci_covers_true_theta"],
                ),
            },
            "nuisance": {
                "solution_rmse_mean": float(metrics["solution_rmse"]),
                "solution_rmse_std": None,
                "beta_rmse_mean": float(metrics["beta_rmse"]),
                "beta_rmse_std": None,
            },
            "ablations": {},
        },
    }
    if metrics.get("nuisance_study") is not None:
        summary["nuisance_study"] = metrics["nuisance_study"]
    if metrics.get("ppi_study") is not None:
        summary["ppi_study"] = metrics["ppi_study"]
    for variant, variant_metrics in metrics.get("ablation_metrics", {}).items():
        summary["metrics"]["ablations"][variant] = _single_run_estimator_summary(
            estimate=variant_metrics["estimate"],
            bias=variant_metrics["bias"],
            rmse=variant_metrics["rmse"],
            ci_length=variant_metrics["ci_length"],
            ci_covers_true_theta=variant_metrics["ci_covers_true_theta"],
        )
    return summary


def _build_corruption_payload(
    levels: list[float],
    values_by_level: dict[float, dict[str, list[float]]],
    include_plugin: bool,
) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for level in levels:
        level_values = values_by_level[level]
        entry = {
            "level": float(level),
            "solution_rmse": float(np.sqrt(np.mean(level_values["solution_sq_errors"]))) if level_values.get("solution_sq_errors") else 0.0,
            "beta_rmse": float(np.sqrt(np.mean(level_values["beta_sq_errors"]))) if level_values.get("beta_sq_errors") else 0.0,
            "onestep_estimate": float(np.mean(level_values["onestep_terms"])),
            "onestep_bias": _scalar_error_metrics(float(np.mean(level_values["onestep_terms"])), float(np.mean(level_values["truth_terms"])))["bias"],
            "onestep_rmse": _scalar_error_metrics(float(np.mean(level_values["onestep_terms"])), float(np.mean(level_values["truth_terms"])))["rmse"],
        }
        if include_plugin:
            entry.update(
                {
                    "plugin_estimate": float(np.mean(level_values["plugin_terms"])),
                    "plugin_bias": _scalar_error_metrics(float(np.mean(level_values["plugin_terms"])), float(np.mean(level_values["truth_terms"])))["bias"],
                    "plugin_rmse": _scalar_error_metrics(float(np.mean(level_values["plugin_terms"])), float(np.mean(level_values["truth_terms"])))["rmse"],
                }
            )
        payload[f"{level:.4f}"] = entry
    return payload


def _build_ppi_payload(
    unlabeled_counts: list[int],
    labeled_plugin_terms: list[float],
    labeled_correction_terms: list[float],
    fixed_true_theta: float,
    unlabeled_plugin_terms: list[float],
) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    n1 = len(labeled_plugin_terms)
    if n1 == 0:
        raise ValueError("PPI study requires at least one labeled sample")
    correction_mean = float(np.mean(labeled_correction_terms))
    for n2 in unlabeled_counts:
        extra_plugin = unlabeled_plugin_terms[:n2]
        total_n = n1 + n2
        plugin_terms = labeled_plugin_terms + extra_plugin
        plugin_estimate = float(np.mean(plugin_terms))
        onestep_estimate = plugin_estimate + correction_mean
        scaled_correction = (total_n / n1) if total_n > 0 else 1.0
        adjusted_terms = [term + scaled_correction * corr for term, corr in zip(labeled_plugin_terms, labeled_correction_terms)] + extra_plugin
        true_theta = fixed_true_theta
        plugin_metrics = _scalar_error_metrics(plugin_estimate, true_theta)
        onestep_metrics = _scalar_error_metrics(onestep_estimate, true_theta)
        plugin_ci_low, plugin_ci_high, plugin_ci_length = _normal_ci(plugin_terms)
        onestep_ci_low, onestep_ci_high, onestep_ci_length = _normal_ci(adjusted_terms)
        payload[str(n2)] = {
            "n1": float(n1),
            "n2": float(n2),
            "total_plugin_samples": float(total_n),
            "true_theta": true_theta,
            "plugin_estimate": plugin_metrics["estimate"],
            "plugin_bias": plugin_metrics["bias"],
            "plugin_rmse": plugin_metrics["rmse"],
            "plugin_ci_low": plugin_ci_low,
            "plugin_ci_high": plugin_ci_high,
            "plugin_ci_length": plugin_ci_length,
            "plugin_ci_covers_true_theta": float(plugin_ci_low <= true_theta <= plugin_ci_high),
            "onestep_estimate": onestep_metrics["estimate"],
            "onestep_bias": onestep_metrics["bias"],
            "onestep_rmse": onestep_metrics["rmse"],
            "onestep_ci_low": onestep_ci_low,
            "onestep_ci_high": onestep_ci_high,
            "onestep_ci_length": onestep_ci_length,
            "onestep_ci_covers_true_theta": float(onestep_ci_low <= true_theta <= onestep_ci_high),
        }
    return payload


def _train_debiasing_variant(
    mode: str,
    train_subset: Subset,
    solution_module: SolutionOperatorModule,
    bundle: NuisanceModelBundleConfig,
    config: Phase3Config,
    fold_root: Path,
) -> DebiasingOperatorModule:
    debiasing_dataset = DebiasingTrainingDataset(train_subset, solution_module)
    debiasing_module = DebiasingOperatorModule(
        bundle.model,
        bundle.debiasing_module,
        functional_name=config.functional,
        functional_params=config.functional_params,
        frozen_solution_model=solution_module.model,
        mode=mode,
    )
    debiasing_trainer = _make_trainer(bundle.debiasing_module.max_epochs)
    debiasing_loader = DataLoader(debiasing_dataset, batch_size=config.batch_size, shuffle=True)
    debiasing_trainer.fit(debiasing_module, train_dataloaders=debiasing_loader)
    return debiasing_module


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


def build_folds(num_samples: int, num_folds: int, seed: int) -> list[list[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator).tolist()
    fold_sizes = [num_samples // num_folds for _ in range(num_folds)]
    for idx in range(num_samples % num_folds):
        fold_sizes[idx] += 1
    folds: list[list[int]] = []
    cursor = 0
    for fold_size in fold_sizes:
        folds.append(permutation[cursor: cursor + fold_size])
        cursor += fold_size
    return folds


def _make_trainer(max_epochs: int) -> pl.Trainer:
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
    )


def _compute_unlabeled_plugin_terms(
    config: Phase3Config,
    dataset_metadata: dict,
    solution_modules: list[SolutionOperatorModule],
) -> list[float]:
    if config.ppi_study is None:
        return []
    max_n2 = max(config.ppi_study.unlabeled_counts, default=0)
    if max_n2 <= 0:
        return []
    if config.dataset_config_path is None:
        raise ValueError("PPI study requires dataset_config_path in the pipeline config.")

    dataset_config = load_dataset_config(config.dataset_config_path)
    with TemporaryDirectory() as tmp_dir:
        unlabeled_config = dataset_config.__class__(
            **{
                **dataset_config.__dict__,
                "seed": int(dataset_metadata["seed"]) + 100000,
                "output_root": tmp_dir,
                "dataset_name": f"pk_ppi_unlabeled_{int(dataset_metadata['seed'])}_{int(config.seed)}",
                "splits": {"test": max_n2},
            }
        )
        unlabeled_root = write_dataset(unlabeled_config)
        unlabeled_dataset = GeneratedSplitDataset(unlabeled_root, "test")
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config.batch_size, shuffle=False)
        plugin_terms: list[float] = []
        for batch in unlabeled_loader:
            coeff = batch["coeff"]
            x_grid = batch["x_grid"][:, :]
            quad_weights = batch["quad_weights"][:, :]
            with torch.no_grad():
                predictions = [module.predict_grid(coeff, x_grid) for module in solution_modules]
                mean_prediction = torch.mean(torch.stack(predictions, dim=0), dim=0)
                plugin = _functional_values(
                    config.functional,
                    mean_prediction,
                    quad_weights,
                    x_grid,
                    functional_params=config.functional_params,
                )
            plugin_terms.extend(plugin.cpu().tolist())
    return plugin_terms


def _compute_ppi_truth_pool_theta(
    config: Phase3Config,
    dataset_metadata: dict,
) -> float:
    if config.ppi_study is None or config.ppi_study.truth_pool_size is None:
        raise ValueError("Accepted PPI runs require ppi_study.truth_pool_size.")
    if config.dataset_config_path is None:
        raise ValueError("PPI study requires dataset_config_path in the pipeline config.")

    truth_pool_size = int(config.ppi_study.truth_pool_size)
    if truth_pool_size <= 0:
        raise ValueError("PPI truth_pool_size must be positive.")

    dataset_config = load_dataset_config(config.dataset_config_path)
    with TemporaryDirectory() as tmp_dir:
        truth_config = dataset_config.__class__(
            **{
                **dataset_config.__dict__,
                "seed": int(dataset_metadata["seed"]) + 200000,
                "output_root": tmp_dir,
                "dataset_name": f"pk_ppi_truth_{int(dataset_metadata['seed'])}_{int(config.seed)}",
                "splits": {"test": truth_pool_size},
            }
        )
        truth_root = write_dataset(truth_config)
        truth_dataset = GeneratedSplitDataset(truth_root, "test")
        truth_loader = DataLoader(truth_dataset, batch_size=config.batch_size, shuffle=False)
        truth_terms: list[float] = []
        for batch in truth_loader:
            quad_weights = batch["quad_weights"][:, :]
            true_solution = batch["solution"]
            with torch.no_grad():
                truth = _functional_values(
                    config.functional,
                    true_solution,
                    quad_weights,
                    batch["x_grid"],
                    functional_params=config.functional_params,
                )
            truth_terms.extend(truth.cpu().tolist())
    return float(np.mean(truth_terms))


def _compute_standard_truth_pool_theta(
    config: Phase3Config,
    dataset_metadata: dict,
) -> float:
    if config.dataset_config_path is None:
        raise ValueError("Standard fixed-truth-pool evaluation requires dataset_config_path in the pipeline config.")

    dataset_config = load_dataset_config(config.dataset_config_path)
    with TemporaryDirectory() as tmp_dir:
        truth_config = dataset_config.__class__(
            **{
                **dataset_config.__dict__,
                "seed": int(dataset_metadata["seed"]) + 300000,
                "output_root": tmp_dir,
                "dataset_name": f"pk_std_truth_{int(dataset_metadata['seed'])}_{int(config.seed)}",
                "splits": {"test": 2000},
            }
        )
        truth_root = write_dataset(truth_config)
        truth_dataset = GeneratedSplitDataset(truth_root, "test")
        truth_loader = DataLoader(truth_dataset, batch_size=config.batch_size, shuffle=False)
        truth_terms: list[float] = []
        for batch in truth_loader:
            quad_weights = batch["quad_weights"][:, :]
            true_solution = batch["solution"]
            with torch.no_grad():
                truth = _functional_values(
                    config.functional,
                    true_solution,
                    quad_weights,
                    batch["x_grid"],
                    functional_params=config.functional_params,
                )
            truth_terms.extend(truth.cpu().tolist())
    return float(np.mean(truth_terms))


def run_phase3(config: Phase3Config, enable_mlflow: bool = True) -> Path:
    _set_global_seed(config.seed)
    bundle = NuisanceModelBundleConfig.from_path(config.model_config_path)
    if config.trainer_max_epochs is not None:
        bundle = NuisanceModelBundleConfig(
            model=bundle.model,
            solution_module=bundle.solution_module.__class__(
                learning_rate=bundle.solution_module.learning_rate,
                weight_decay=bundle.solution_module.weight_decay,
                max_epochs=config.trainer_max_epochs,
                riesz_penalty_lambda=bundle.solution_module.riesz_penalty_lambda,
            ),
            debiasing_module=bundle.debiasing_module.__class__(
                learning_rate=bundle.debiasing_module.learning_rate,
                weight_decay=bundle.debiasing_module.weight_decay,
                max_epochs=config.trainer_max_epochs,
                riesz_penalty_lambda=bundle.debiasing_module.riesz_penalty_lambda,
            ),
        )

    dataset = GeneratedSplitDataset(_resolve_generated_dataset_root(config.dataset_root), config.split)
    folds = build_folds(len(dataset), config.num_folds, config.seed)
    benchmark_name = str(dataset.metadata.get("benchmark_name", dataset.metadata.get("pde_class", "benchmark")))
    run_root = Path(config.output_root) / benchmark_name / config.functional / config.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    tracker = MlflowTracker(
        experiment_name=config.run_name,
        run_name=config.run_name,
        enabled=enable_mlflow,
    )
    tracker.start(
        tags={
            "entrypoint": "run_phase3",
            "benchmark": benchmark_name,
            "functional": config.functional,
            "split": config.split,
        }
    )
    try:
        resolved_spec = {
            "pipeline_config": asdict(config),
            "model_config": json.loads(Path(config.model_config_path).read_text(encoding="utf-8")),
            "dataset_metadata": dataset.metadata,
        }
        _save_json(run_root / "config.json", resolved_spec)
        tracker.log_params_from_dict("pipeline", resolved_spec["pipeline_config"])
        tracker.log_params_from_dict("model", resolved_spec["model_config"])
        tracker.log_params_from_dict("dataset", resolved_spec["dataset_metadata"])
        tracker.log_json_artifact(resolved_spec, "resolved_spec.json")
        tracker.log_artifact(run_root / "config.json", artifact_path="run_outputs")

        all_plugin_terms: list[float] = []
        all_correction_terms: list[float] = []
        all_pseudo_outcomes: list[float] = []
        all_true_terms: list[float] = []
        all_solution_sq_errors: list[float] = []
        all_beta_sq_errors: list[float] = []
        variant_names = list(dict.fromkeys(config.ablation_variants))
        all_variant_terms: dict[str, list[float]] = {name: [] for name in variant_names}
        trained_solution_modules: list[SolutionOperatorModule] = []
        solution_bias_levels = config.nuisance_study.solution_bias_levels if config.nuisance_study is not None else []
        beta_bias_levels = config.nuisance_study.beta_bias_levels if config.nuisance_study is not None else []
        joint_bias_levels = config.nuisance_study.joint_bias_levels if config.nuisance_study is not None else []
        solution_bias_payloads = {
            float(level): {
                "plugin_terms": [],
                "onestep_terms": [],
                "truth_terms": [],
                "solution_sq_errors": [],
                "beta_sq_errors": [],
            }
            for level in solution_bias_levels
        }
        beta_bias_payloads = {
            float(level): {
                "onestep_terms": [],
                "truth_terms": [],
                "solution_sq_errors": [],
                "beta_sq_errors": [],
            }
            for level in beta_bias_levels
        }
        joint_bias_payloads = {
            float(level): {
                "onestep_terms": [],
                "truth_terms": [],
                "solution_sq_errors": [],
                "beta_sq_errors": [],
            }
            for level in joint_bias_levels
        }

        for fold_idx, eval_indices in enumerate(folds):
            eval_set = set(eval_indices)
            train_indices = [idx for idx in range(len(dataset)) if idx not in eval_set]
            fold_root = run_root / f"fold_{fold_idx}"
            fold_root.mkdir(parents=True, exist_ok=True)

            train_subset = Subset(dataset, train_indices)
            eval_subset = Subset(dataset, eval_indices)

            solution_module = SolutionOperatorModule(bundle.model, bundle.solution_module)
            solution_trainer = _make_trainer(bundle.solution_module.max_epochs)
            solution_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
            solution_trainer.fit(solution_module, train_dataloaders=solution_loader)
            solution_module.freeze_model()
            trained_solution_modules.append(solution_module)

            debiasing_module = _train_debiasing_variant(
                mode="full",
                train_subset=train_subset,
                solution_module=solution_module,
                bundle=bundle,
                config=config,
                fold_root=fold_root,
            )

            trained_variant_modules: dict[str, DebiasingOperatorModule] = {}
            for variant in variant_names:
                if variant == "structured":
                    trained_variant_modules[variant] = _train_debiasing_variant(
                        mode="structured",
                        train_subset=train_subset,
                        solution_module=solution_module,
                        bundle=bundle,
                        config=config,
                        fold_root=fold_root,
                    )
                elif variant == "oracle_wg":
                    trained_variant_modules[variant] = _train_debiasing_variant(
                        mode="oracle_wg",
                        train_subset=train_subset,
                        solution_module=solution_module,
                        bundle=bundle,
                        config=config,
                        fold_root=fold_root,
                    )
                elif variant == "oracle_xi":
                    trained_variant_modules[variant] = _train_debiasing_variant(
                        mode="oracle_xi",
                        train_subset=train_subset,
                        solution_module=solution_module,
                        bundle=bundle,
                        config=config,
                        fold_root=fold_root,
                    )

            eval_loader = DataLoader(eval_subset, batch_size=config.batch_size, shuffle=False)
            plugin_terms: list[float] = []
            correction_terms: list[float] = []
            pseudo_outcomes: list[float] = []
            true_terms: list[float] = []
            solution_sq_errors: list[float] = []
            beta_sq_errors: list[float] = []
            fold_variant_terms: dict[str, list[float]] = {name: [] for name in variant_names}

            for batch in eval_loader:
                coeff = batch["coeff"]
                x_grid = batch["x_grid"][:, :]
                obs_x = batch["obs_x"]
                obs_y = batch["obs_y"]
                quad_weights = batch["quad_weights"][:, :]
                true_solution = batch["solution"]

                with torch.no_grad():
                    s_grid = solution_module.predict_grid(coeff, x_grid)
                    s_points = solution_module.predict_points(coeff, obs_x, x_grid)
                    b_grid, b_points = debiasing_module.compose_beta(
                        coeff,
                        x_grid,
                        obs_x,
                        {"u_hat": s_grid, **batch},
                    )

                    plug_in = _functional_values(
                        config.functional,
                        s_grid,
                        quad_weights,
                        x_grid,
                        functional_params=config.functional_params,
                    )
                    correction = torch.mean(b_points * (obs_y - s_points), dim=-1)
                    pseudo = plug_in + correction
                    truth = _functional_values(
                        config.functional,
                        true_solution,
                        quad_weights,
                        x_grid,
                        functional_params=config.functional_params,
                    )
                    true_beta = batch["xi"] * riesz_density(
                        config.functional,
                        _flatten_batch_grid(true_solution),
                        _flatten_batch_grid(quad_weights),
                        _flatten_batch_coordinates(x_grid, true_solution),
                        **config.functional_params,
                    ).reshape_as(batch["xi"])
                    solution_sq = _weighted_l2_error(s_grid, true_solution, quad_weights)
                    beta_sq = _weighted_l2_error(b_grid, true_beta, quad_weights)

                    for variant in variant_names:
                        if variant == "oracle_beta":
                            variant_b_grid = true_beta
                            variant_b_points = interpolate_from_grid(x_grid, true_beta, obs_x)
                        else:
                            variant_module = trained_variant_modules[variant]
                            variant_b_grid, variant_b_points = variant_module.compose_beta(
                                coeff,
                                x_grid,
                                obs_x,
                                {"u_hat": s_grid, **batch},
                            )
                        variant_correction = torch.mean(variant_b_points * (obs_y - s_points), dim=-1)
                        variant_pseudo = plug_in + variant_correction
                        fold_variant_terms[variant].extend(variant_pseudo.cpu().tolist())

                    for level in solution_bias_levels:
                        corrupted_s_grid = _apply_solution_bias(s_grid, true_solution, float(level))
                        corrupted_s_points = interpolate_from_grid(x_grid, corrupted_s_grid, obs_x)
                        corrupted_plugin = _functional_values(
                            config.functional,
                            corrupted_s_grid,
                            quad_weights,
                            x_grid,
                            functional_params=config.functional_params,
                        )
                        corrupted_correction = torch.mean(b_points * (obs_y - corrupted_s_points), dim=-1)
                        corrupted_onestep = corrupted_plugin + corrupted_correction
                        corrupted_solution_sq = _weighted_l2_error(corrupted_s_grid, true_solution, quad_weights)
                        solution_bias_payloads[float(level)]["plugin_terms"].extend(corrupted_plugin.cpu().tolist())
                        solution_bias_payloads[float(level)]["onestep_terms"].extend(corrupted_onestep.cpu().tolist())
                        solution_bias_payloads[float(level)]["truth_terms"].extend(truth.cpu().tolist())
                        solution_bias_payloads[float(level)]["solution_sq_errors"].extend(corrupted_solution_sq.cpu().tolist())
                        solution_bias_payloads[float(level)]["beta_sq_errors"].extend(beta_sq.cpu().tolist())

                    for level in beta_bias_levels:
                        corrupted_b_grid = _apply_beta_bias(b_grid, true_beta, float(level))
                        corrupted_b_points = interpolate_from_grid(x_grid, corrupted_b_grid, obs_x)
                        corrupted_correction = torch.mean(corrupted_b_points * (obs_y - s_points), dim=-1)
                        corrupted_onestep = plug_in + corrupted_correction
                        corrupted_beta_sq = _weighted_l2_error(corrupted_b_grid, true_beta, quad_weights)
                        beta_bias_payloads[float(level)]["onestep_terms"].extend(corrupted_onestep.cpu().tolist())
                        beta_bias_payloads[float(level)]["truth_terms"].extend(truth.cpu().tolist())
                        beta_bias_payloads[float(level)]["solution_sq_errors"].extend(solution_sq.cpu().tolist())
                        beta_bias_payloads[float(level)]["beta_sq_errors"].extend(corrupted_beta_sq.cpu().tolist())

                    for level in joint_bias_levels:
                        corrupted_s_grid = _apply_solution_bias(s_grid, true_solution, float(level))
                        corrupted_s_points = interpolate_from_grid(x_grid, corrupted_s_grid, obs_x)
                        corrupted_b_grid = _apply_beta_bias(b_grid, true_beta, float(level))
                        corrupted_b_points = interpolate_from_grid(x_grid, corrupted_b_grid, obs_x)
                        corrupted_plugin = _functional_values(
                            config.functional,
                            corrupted_s_grid,
                            quad_weights,
                            x_grid,
                            functional_params=config.functional_params,
                        )
                        corrupted_correction = torch.mean(corrupted_b_points * (obs_y - corrupted_s_points), dim=-1)
                        corrupted_onestep = corrupted_plugin + corrupted_correction
                        corrupted_solution_sq = _weighted_l2_error(corrupted_s_grid, true_solution, quad_weights)
                        corrupted_beta_sq = _weighted_l2_error(corrupted_b_grid, true_beta, quad_weights)
                        joint_bias_payloads[float(level)]["onestep_terms"].extend(corrupted_onestep.cpu().tolist())
                        joint_bias_payloads[float(level)]["truth_terms"].extend(truth.cpu().tolist())
                        joint_bias_payloads[float(level)]["solution_sq_errors"].extend(corrupted_solution_sq.cpu().tolist())
                        joint_bias_payloads[float(level)]["beta_sq_errors"].extend(corrupted_beta_sq.cpu().tolist())

                plugin_terms.extend(plug_in.cpu().tolist())
                correction_terms.extend(correction.cpu().tolist())
                pseudo_outcomes.extend(pseudo.cpu().tolist())
                true_terms.extend(truth.cpu().tolist())
                solution_sq_errors.extend(solution_sq.cpu().tolist())
                beta_sq_errors.extend(beta_sq.cpu().tolist())

            all_plugin_terms.extend(plugin_terms)
            all_correction_terms.extend(correction_terms)
            all_pseudo_outcomes.extend(pseudo_outcomes)
            all_true_terms.extend(true_terms)
            all_solution_sq_errors.extend(solution_sq_errors)
            all_beta_sq_errors.extend(beta_sq_errors)
            for variant in variant_names:
                all_variant_terms[variant].extend(fold_variant_terms[variant])

            fold_plugin_estimate = float(np.mean(plugin_terms))
            fold_onestep_estimate = float(np.mean(pseudo_outcomes))
            fold_true_theta = float(np.mean(true_terms))
            fold_plugin_metrics = _scalar_error_metrics(fold_plugin_estimate, fold_true_theta)
            fold_onestep_metrics = _scalar_error_metrics(fold_onestep_estimate, fold_true_theta)

            fold_payload = {
                "fold_index": fold_idx,
                "train_indices": train_indices,
                "eval_indices": eval_indices,
                "solution_model_path": None,
                "debiasing_model_path": None,
                "plugin_sample_terms": plugin_terms,
                "correction_sample_terms": correction_terms,
                "pseudo_outcome_samples": pseudo_outcomes,
                "true_functional_samples": true_terms,
                "solution_sq_errors": solution_sq_errors,
                "beta_sq_errors": beta_sq_errors,
                "plugin_estimate": fold_plugin_metrics["estimate"],
                "onestep_estimate": fold_onestep_metrics["estimate"],
                "true_theta": fold_true_theta,
                "plugin_bias": fold_plugin_metrics["bias"],
                "onestep_bias": fold_onestep_metrics["bias"],
                "solution_rmse": float(np.sqrt(np.mean(solution_sq_errors))),
                "beta_rmse": float(np.sqrt(np.mean(beta_sq_errors))),
                "plugin_rmse": fold_plugin_metrics["rmse"],
                "onestep_rmse": fold_onestep_metrics["rmse"],
            }
            if variant_names:
                fold_payload["variant_model_paths"] = {}
                for variant in variant_names:
                    if variant == "oracle_beta":
                        fold_payload["variant_model_paths"][variant] = "oracle"
                    else:
                        fold_payload["variant_model_paths"][variant] = None
                    variant_estimate = float(np.mean(fold_variant_terms[variant]))
                    variant_metrics = _scalar_error_metrics(variant_estimate, fold_true_theta)
                    fold_payload[f"{variant}_pseudo_outcome_samples"] = fold_variant_terms[variant]
                    fold_payload[f"{variant}_estimate"] = variant_metrics["estimate"]
                    fold_payload[f"{variant}_bias"] = variant_metrics["bias"]
                    fold_payload[f"{variant}_rmse"] = variant_metrics["rmse"]
                    low, high, length = _normal_ci(fold_variant_terms[variant])
                    fold_payload[f"{variant}_ci_low"] = low
                    fold_payload[f"{variant}_ci_high"] = high
                    fold_payload[f"{variant}_ci_length"] = length

            _save_json(fold_root / "diagnostics.json", fold_payload)
            with tracker.nested_run(
                run_name=f"fold_{fold_idx}",
                tags={"kind": "fold", "fold_index": str(fold_idx)},
            ):
                tracker.log_metrics_from_dict(
                    {
                        "solution_rmse": fold_payload["solution_rmse"],
                        "beta_rmse": fold_payload["beta_rmse"],
                        "plugin_rmse": fold_payload["plugin_rmse"],
                        "onestep_rmse": fold_payload["onestep_rmse"],
                    },
                    prefix="fold",
                )
                if variant_names:
                    variant_fold_metrics = {
                        variant: {
                            "rmse": fold_payload[f"{variant}_rmse"],
                            "ci_low": fold_payload[f"{variant}_ci_low"],
                            "ci_high": fold_payload[f"{variant}_ci_high"],
                            "ci_length": fold_payload[f"{variant}_ci_length"],
                        }
                        for variant in variant_names
                    }
                    tracker.log_metrics_from_dict(variant_fold_metrics, prefix="ablation")
                tracker.log_artifact(fold_root / "diagnostics.json", artifact_path=f"folds/fold_{fold_idx}")

        plugin_estimate = float(np.mean(all_plugin_terms))
        onestep_estimate = float(np.mean(all_pseudo_outcomes))
        if config.dataset_config_path is None:
            raise ValueError("Accepted PK runs require dataset_config_path for disjoint truth-pool evaluation.")
        if config.ppi_study is None:
            true_theta = _compute_standard_truth_pool_theta(config, dataset.metadata)
            target_source = "standard_truth_pool"
            target_pool_size: int | None = 2000
        else:
            true_theta = _compute_ppi_truth_pool_theta(config, dataset.metadata)
            target_source = "ppi_truth_pool"
            target_pool_size = int(config.ppi_study.truth_pool_size)
        plugin_metrics = _scalar_error_metrics(plugin_estimate, true_theta)
        onestep_metrics = _scalar_error_metrics(onestep_estimate, true_theta)
        metrics = {
            "plugin_estimate": plugin_metrics["estimate"],
            "onestep_estimate": onestep_metrics["estimate"],
            "true_theta": true_theta,
            "target_source": target_source,
            "target_pool_size": target_pool_size,
            "plugin_bias": plugin_metrics["bias"],
            "onestep_bias": onestep_metrics["bias"],
            "plugin_rmse": plugin_metrics["rmse"],
            "onestep_rmse": onestep_metrics["rmse"],
            "solution_rmse": float(np.sqrt(np.mean(all_solution_sq_errors))),
            "beta_rmse": float(np.sqrt(np.mean(all_beta_sq_errors))),
            "num_samples": len(dataset),
            "num_folds": config.num_folds,
        }
        plugin_ci_low, plugin_ci_high, plugin_ci_length = _normal_ci(all_plugin_terms)
        onestep_ci_low, onestep_ci_high, onestep_ci_length = _normal_ci(all_pseudo_outcomes)
        metrics["plugin_ci_low"] = plugin_ci_low
        metrics["plugin_ci_high"] = plugin_ci_high
        metrics["plugin_ci_length"] = plugin_ci_length
        metrics["plugin_ci_covers_true_theta"] = plugin_ci_low <= true_theta <= plugin_ci_high
        metrics["onestep_ci_low"] = onestep_ci_low
        metrics["onestep_ci_high"] = onestep_ci_high
        metrics["onestep_ci_length"] = onestep_ci_length
        metrics["onestep_ci_covers_true_theta"] = onestep_ci_low <= true_theta <= onestep_ci_high
        if variant_names:
            metrics["ablation_metrics"] = {}
            for variant in variant_names:
                variant_estimate = float(np.mean(all_variant_terms[variant]))
                variant_metrics = _scalar_error_metrics(variant_estimate, true_theta)
                ci_low, ci_high, ci_length = _normal_ci(all_variant_terms[variant])
                metrics["ablation_metrics"][variant] = {
                    "estimate": variant_metrics["estimate"],
                    "bias": variant_metrics["bias"],
                    "rmse": variant_metrics["rmse"],
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_length": ci_length,
                    "ci_covers_true_theta": ci_low <= true_theta <= ci_high,
                }
        ppi_payload = None
        if config.ppi_study is not None:
            unlabeled_plugin_terms = _compute_unlabeled_plugin_terms(
                config,
                dataset.metadata,
                trained_solution_modules,
            )
            ppi_payload = _build_ppi_payload(
                unlabeled_counts=config.ppi_study.unlabeled_counts,
                labeled_plugin_terms=all_plugin_terms,
                labeled_correction_terms=all_correction_terms,
                fixed_true_theta=true_theta,
                unlabeled_plugin_terms=unlabeled_plugin_terms,
            )

        summary = _single_run_summary(
            config=config,
            resolved_spec=resolved_spec,
            num_samples=len(dataset),
            metrics={
                **metrics,
                "nuisance_study": (
                    {
                        "solution_bias": _build_corruption_payload(solution_bias_levels, solution_bias_payloads, include_plugin=True),
                        "beta_bias": _build_corruption_payload(beta_bias_levels, beta_bias_payloads, include_plugin=False),
                        "joint_bias": _build_corruption_payload(joint_bias_levels, joint_bias_payloads, include_plugin=False),
                    }
                    if config.nuisance_study is not None
                    else None
                ),
                "ppi_study": ppi_payload,
            },
        )
        _save_json(run_root / "summary.json", summary)
        tracker.log_metrics_from_dict(summary["metrics"])
        tracker.log_json_artifact(summary, "summary.json")
        tracker.log_artifact(run_root / "summary.json", artifact_path="run_outputs")
        tracker.log_artifacts(run_root, artifact_path="full_run")
    finally:
        tracker.end()
    return run_root
