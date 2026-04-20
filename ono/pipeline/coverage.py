from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math

from ono.data.generate import load_dataset_config, write_dataset
from ono.pipeline.config import Phase3Config
from ono.pipeline.phase3 import run_phase3
from ono.tracking import MlflowTracker


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _root_mean_square(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _optional_std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return None
    return _sample_std(values)


def _aggregate_estimator_metrics(run_metrics: list[dict], key: str) -> dict[str, float | None]:
    estimates = [item["estimators"][key]["estimate_mean"] for item in run_metrics]
    biases = [item["estimators"][key]["bias_mean"] for item in run_metrics]
    abs_errors = [abs(bias) for bias in biases]
    coverages = [item["estimators"][key]["ci_coverage"] for item in run_metrics]
    ci_lengths = [item["estimators"][key]["ci_length_mean"] for item in run_metrics]
    return {
        "estimate_mean": _mean(estimates),
        "estimate_std": _optional_std(estimates),
        "bias_mean": _mean(biases),
        "bias_std": _optional_std(biases),
        "rmse_mean": _root_mean_square(biases),
        "rmse_std": _optional_std(abs_errors),
        "ci_coverage": _mean(coverages),
        "ci_length_mean": _mean(ci_lengths),
        "ci_length_std": _optional_std(ci_lengths),
    }


def _aggregate_nuisance_study(repeat_payloads: list[dict]) -> dict[str, dict[str, dict[str, float | None]]]:
    first_summary = repeat_payloads[0]["summary"]
    nuisance_study = first_summary.get("nuisance_study")
    if nuisance_study is None:
        return {}
    aggregated: dict[str, dict[str, dict[str, float | None]]] = {}
    for target, level_payload in nuisance_study.items():
        aggregated[target] = {}
        for level_key in level_payload:
            level_runs = [item["summary"]["nuisance_study"][target][level_key] for item in repeat_payloads]
            numeric_keys = sorted(level_runs[0].keys())
            entry: dict[str, float | None] = {}
            for key in numeric_keys:
                values = [run[key] for run in level_runs]
                if key == "level":
                    entry[key] = float(values[0])
                elif key in {"plugin_rmse", "onestep_rmse"}:
                    bias_key = key.replace("_rmse", "_bias")
                    biases = [run[bias_key] for run in level_runs]
                    entry[f"{key}_mean"] = _root_mean_square(biases)
                    entry[f"{key}_std"] = _optional_std([abs(value) for value in biases])
                else:
                    entry[f"{key}_mean"] = _mean(values)
                    entry[f"{key}_std"] = _optional_std(values)
            aggregated[target][level_key] = entry
    return aggregated


def _aggregate_ppi_study(repeat_payloads: list[dict]) -> dict[str, dict[str, float | None]]:
    first_summary = repeat_payloads[0]["summary"]
    ppi_study = first_summary.get("ppi_study")
    if ppi_study is None:
        return {}
    aggregated: dict[str, dict[str, float | None]] = {}
    for level_key in ppi_study:
        level_runs = [item["summary"]["ppi_study"][level_key] for item in repeat_payloads]
        numeric_keys = sorted(level_runs[0].keys())
        entry: dict[str, float | None] = {}
        for key in numeric_keys:
            values = [run[key] for run in level_runs]
            if key in {"n1", "n2", "total_plugin_samples"}:
                entry[key] = float(values[0])
            elif key in {"plugin_rmse", "onestep_rmse"}:
                bias_key = key.replace("_rmse", "_bias")
                biases = [run[bias_key] for run in level_runs]
                entry[f"{key}_mean"] = _root_mean_square(biases)
                entry[f"{key}_std"] = _optional_std([abs(value) for value in biases])
            else:
                entry[f"{key}_mean"] = _mean(values)
                entry[f"{key}_std"] = _optional_std(values)
        aggregated[level_key] = entry
    return aggregated


def _build_repeated_run_summary(
    config: Phase3Config,
    pipeline_config_path: str | Path | None,
    resolved_spec: dict | None,
    repeat_payloads: list[dict],
) -> dict:
    run_metrics = [item["summary"]["metrics"] for item in repeat_payloads]
    true_thetas = [item["target"]["true_theta_mean"] for item in run_metrics]
    target_sources = [item["target"].get("target_source") for item in run_metrics]
    target_pool_sizes = [item["target"].get("target_pool_size") for item in run_metrics]
    solution_rmses = [item["nuisance"]["solution_rmse_mean"] for item in run_metrics]
    beta_rmses = [item["nuisance"]["beta_rmse_mean"] for item in run_metrics]
    variant_names = sorted({name for item in run_metrics for name in item["ablations"]})
    summary = {
        "pipeline_config": asdict(config),
        "pipeline_config_path": str(pipeline_config_path) if pipeline_config_path is not None else None,
        "model_config": resolved_spec["model_config"] if resolved_spec is not None else None,
        "dataset_metadata": resolved_spec["dataset_metadata"] if resolved_spec is not None else None,
        "num_repeats": len(repeat_payloads),
        "num_samples_per_run": repeat_payloads[0]["summary"]["num_samples_per_run"],
        "num_folds": config.num_folds,
        "metrics": {
            "target": {
                "true_theta_mean": _mean(true_thetas),
                "true_theta_std": _optional_std(true_thetas),
                "target_source": target_sources[0] if target_sources and all(value == target_sources[0] for value in target_sources) else None,
                "target_pool_size": (
                    target_pool_sizes[0]
                    if target_pool_sizes and all(value == target_pool_sizes[0] for value in target_pool_sizes)
                    else None
                ),
            },
            "estimators": {
                "plugin": _aggregate_estimator_metrics(run_metrics, "plugin"),
                "onestep": _aggregate_estimator_metrics(run_metrics, "onestep"),
            },
            "nuisance": {
                "solution_rmse_mean": _mean(solution_rmses),
                "solution_rmse_std": _optional_std(solution_rmses),
                "beta_rmse_mean": _mean(beta_rmses),
                "beta_rmse_std": _optional_std(beta_rmses),
            },
            "ablations": {},
        },
        "repeats": repeat_payloads,
    }
    nuisance_study = _aggregate_nuisance_study(repeat_payloads)
    if nuisance_study:
        summary["nuisance_study"] = nuisance_study
    ppi_study = _aggregate_ppi_study(repeat_payloads)
    if ppi_study:
        summary["ppi_study"] = ppi_study
    for variant in variant_names:
        variant_metrics = [item["ablations"][variant] for item in run_metrics if variant in item["ablations"]]
        if not variant_metrics:
            continue
        summary["metrics"]["ablations"][variant] = {
            "estimate_mean": _mean([item["estimate_mean"] for item in variant_metrics]),
            "estimate_std": _optional_std([item["estimate_mean"] for item in variant_metrics]),
            "bias_mean": _mean([item["bias_mean"] for item in variant_metrics]),
            "bias_std": _optional_std([item["bias_mean"] for item in variant_metrics]),
            "rmse_mean": _root_mean_square([item["bias_mean"] for item in variant_metrics]),
            "rmse_std": _optional_std([abs(item["bias_mean"]) for item in variant_metrics]),
            "ci_coverage": _mean([item["ci_coverage"] for item in variant_metrics]),
            "ci_length_mean": _mean([item["ci_length_mean"] for item in variant_metrics]),
            "ci_length_std": _optional_std([item["ci_length_mean"] for item in variant_metrics]),
        }
    return summary


def _run_repeated_study(
    config: Phase3Config,
    pipeline_config_path: str | Path | None = None,
    enable_mlflow: bool = True,
) -> Path:
    if config.dataset_config_path is None:
        raise ValueError("Repeated runs require dataset_config_path in the pipeline config.")
    if config.repeated_runs is None:
        raise ValueError("Repeated runs require a repeated_runs block in the pipeline config.")

    dataset_template = load_dataset_config(config.dataset_config_path)
    study_name = config.repeated_runs.study_name or f"{config.run_name}_repeats"
    study_root = Path(config.repeated_runs.output_root) / study_name
    study_root.mkdir(parents=True, exist_ok=True)
    repeats: list[dict] = []
    resolved_spec: dict | None = None
    tracker = MlflowTracker(
        experiment_name=study_name,
        run_name=study_name,
        enabled=enable_mlflow,
    )
    tracker.start(
        tags={
            "entrypoint": "run_pipeline_config",
            "kind": "repeated_run_study",
            "base_run_name": config.run_name,
        }
    )
    try:
        tracker.log_params_from_dict("pipeline", asdict(config))
        if pipeline_config_path is not None:
            tracker.log_params_from_dict(
                "execution",
                {"pipeline_config_path": str(pipeline_config_path)},
            )

        for repeat_idx in range(config.repeated_runs.num_repeats):
            dataset_seed = dataset_template.seed + config.repeated_runs.seed_offset + repeat_idx
            dataset_name = f"{dataset_template.dataset_name}_{study_name}_rep{repeat_idx}"
            dataset_config = dataset_template.__class__(
                **{**dataset_template.__dict__, "seed": dataset_seed, "dataset_name": dataset_name}
            )
            dataset_root = write_dataset(dataset_config)

            run_name = f"{config.run_name}_{study_name}_rep{repeat_idx}"
            repeat_config = Phase3Config(
                **{
                    **config.__dict__,
                    "dataset_root": str(dataset_root),
                    "run_name": run_name,
                }
            )
            run_root = run_phase3(repeat_config, enable_mlflow=False)
            summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
            if resolved_spec is None:
                resolved_spec = {
                    "model_config": summary.get("model_config"),
                    "dataset_metadata": summary.get("dataset_metadata"),
                }
            repeat_payload = {
                "repeat_index": repeat_idx,
                "dataset_seed": dataset_seed,
                "train_seed": config.seed,
                "dataset_root": str(dataset_root),
                "run_root": str(run_root),
                "summary": summary,
            }
            repeats.append(repeat_payload)
            with tracker.nested_run(
                run_name=f"repeat_{repeat_idx}",
                tags={"kind": "repeat", "repeat_index": str(repeat_idx)},
            ):
                tracker.log_params_from_dict(
                    "repeat_execution",
                    {
                        "repeat_index": repeat_idx,
                        "dataset_seed": dataset_seed,
                        "train_seed": config.seed,
                        "dataset_root": str(dataset_root),
                        "run_root": str(run_root),
                    },
                )
                tracker.log_artifacts(run_root, artifact_path=f"repeats/repeat_{repeat_idx}")
        summary = _build_repeated_run_summary(
            config=config,
            pipeline_config_path=pipeline_config_path,
            resolved_spec=resolved_spec,
            repeat_payloads=repeats,
        )
        (study_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tracker.log_metrics_from_dict(summary["metrics"])
        tracker.log_json_artifact(summary, "summary.json")
        tracker.log_artifacts(study_root, artifact_path="coverage_outputs")
    finally:
        tracker.end()
    return study_root


def run_pipeline_config(
    config: Phase3Config,
    pipeline_config_path: str | Path | None = None,
    enable_mlflow: bool = True,
) -> Path:
    if config.repeated_runs is not None and config.repeated_runs.num_repeats > 1:
        return _run_repeated_study(
            config=config,
            pipeline_config_path=pipeline_config_path,
            enable_mlflow=enable_mlflow,
        )
    return run_phase3(config, enable_mlflow=enable_mlflow)
