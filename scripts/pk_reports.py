from __future__ import annotations

import csv
from datetime import datetime
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = ROOT / "results" / "reports"
COVERAGE_ROOT = ROOT / "results" / "coverage"

ABLATION_FUNCTIONALS = ["auc", "soft_cmax", "smooth_tat"]
NUISANCE_FUNCTIONALS = ["auc", "smooth_tat"]
PPI_FUNCTIONALS = ["auc", "soft_cmax", "smooth_tat"]
RHO_LABELS = [
    ("r000", 0.0),
    ("r125", 0.125),
    ("r250", 0.25),
    ("r375", 0.375),
    ("r500", 0.5),
    ("r625", 0.625),
    ("r750", 0.75),
    ("r875", 0.875),
    ("r1000", 1.0),
]
ABLATION_METHODS = [
    ("plugin", "plugin"),
    ("onestep", "one-step"),
    ("oracle_beta", "oracle_beta"),
    ("oracle_wg", "oracle_wg"),
    ("oracle_xi", "oracle_xi"),
    ("structured", "structured"),
]
PPI_MULTIPLIERS = [0, 1, 2, 4, 8, 12, 16, 32, 64]
PPI_N1 = 64
PPI_N2_GRID = [multiplier * PPI_N1 for multiplier in PPI_MULTIPLIERS]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _optional_std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return None
    return _sample_std(values)


def _root_mean_square(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _aggregate_scalar_estimator(items: list[dict], estimate_key: str, bias_key: str, ci_coverage_key: str, ci_length_key: str) -> dict[str, float | None]:
    estimates = [float(item[estimate_key]) for item in items]
    biases = [float(item[bias_key]) for item in items]
    abs_errors = [abs(value) for value in biases]
    coverages = [float(item[ci_coverage_key]) for item in items]
    ci_lengths = [float(item[ci_length_key]) for item in items]
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


def _aggregate_scalar_estimator_no_ci(items: list[dict], estimate_key: str, bias_key: str) -> dict[str, float | None]:
    estimates = [float(item[estimate_key]) for item in items]
    biases = [float(item[bias_key]) for item in items]
    abs_errors = [abs(value) for value in biases]
    return {
        "estimate_mean": _mean(estimates),
        "estimate_std": _optional_std(estimates),
        "bias_mean": _mean(biases),
        "bias_std": _optional_std(biases),
        "rmse_mean": _root_mean_square(biases),
        "rmse_std": _optional_std(abs_errors),
        "ci_coverage": None,
        "ci_length_mean": None,
        "ci_length_std": None,
    }


def _format_value(mean: float | None, std: float | None) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.6f}"
    return f"{mean:.6f} +/- {std:.6f}"


def _compare_value(old_rows: list[dict], key_fields: dict[str, object], metric_name: str, corrected_value: float) -> dict[str, float | None]:
    for row in old_rows:
        if all(str(row[field]) == str(value) for field, value in key_fields.items()):
            old_value = float(row[metric_name])
            return {
                "old": old_value,
                "corrected": corrected_value,
                "delta": corrected_value - old_value,
            }
    return {"old": None, "corrected": corrected_value, "delta": None}


def _resolve_existing_directory(parent: Path, preferred_name: str, glob_pattern: str) -> Path:
    preferred = parent / preferred_name
    if preferred.exists():
        return preferred
    matches = sorted(path for path in parent.glob(glob_pattern) if path.is_dir())
    if matches:
        return matches[0]
    return preferred


def _resolve_existing_file(parent: Path, preferred_name: str, glob_pattern: str) -> Path:
    preferred = parent / preferred_name
    if preferred.exists():
        return preferred
    matches = sorted(path for path in parent.glob(glob_pattern) if path.is_file())
    if matches:
        return matches[0]
    return preferred


def _resolve_summary_path(preferred_name: str, glob_pattern: str) -> Path:
    preferred = COVERAGE_ROOT / preferred_name / "summary.json"
    if preferred.exists():
        return preferred
    matches = sorted(COVERAGE_ROOT.glob(f"{glob_pattern}/summary.json"))
    if matches:
        return matches[0]
    return preferred


def _main_summary_path(functional: str, rho_label: str) -> Path:
    preferred = f"pk_main_{functional}_{rho_label}_repeats50"
    preferred_path = _resolve_summary_path(preferred, preferred)
    if preferred_path.exists():
        return preferred_path
    # Coverage outputs are intentionally not renamed, so the report script falls back to the legacy stored names.
    legacy_pattern = f"pk_ablation_{functional}_irreg_unit_{rho_label}*repeats50*"
    return _resolve_summary_path(preferred, legacy_pattern)


def _nuisance_summary_path(functional: str) -> Path:
    preferred = f"pk_nuisance_{functional}_r500_repeats50"
    pattern = f"pk_nuisance_{functional}_r500*repeats50*"
    return _resolve_summary_path(preferred, pattern)


def _ppi_summary_path(functional: str) -> Path:
    preferred = f"pk_ppi_{functional}_r500_repeats50"
    preferred_path = _resolve_summary_path(preferred, preferred)
    if preferred_path.exists():
        return preferred_path
    # Coverage outputs are intentionally not renamed, so the report script falls back to the legacy stored names.
    legacy_pattern = f"pk_ppi*_{functional}_irreg_unit_r500*repeats50*"
    return _resolve_summary_path(preferred, legacy_pattern)


def _recompute_main_reports() -> dict:
    report_root = _resolve_existing_directory(REPORTS_ROOT, "pk_main", "pk_ablation_irreg_unit*")
    report_root.mkdir(parents=True, exist_ok=True)
    old_long_path = _resolve_existing_file(report_root, "pk_main_long.csv", "pk_ablation*_long.csv")
    old_long_rows = list(csv.DictReader(old_long_path.open(encoding="utf-8", newline="")))

    long_rows: list[dict] = []
    rmse_wide_rows: list[dict] = []
    bias_wide_rows: list[dict] = []
    source_summaries: dict[str, str] = {}
    representative_diffs: dict[str, dict[str, float | None]] = {}

    for functional in ABLATION_FUNCTIONALS:
        for rho_label, rho_value in RHO_LABELS:
            summary_path = _main_summary_path(functional, rho_label)
            summary = _load_json(summary_path)
            source_summaries[f"{functional}_{rho_label}"] = str(summary_path.relative_to(ROOT))

            rmse_row = {"functional": functional, "rho": rho_value}
            bias_row = {"functional": functional, "rho": rho_value}
            repeats = summary["repeats"]
            for method_key, method_label in ABLATION_METHODS:
                if method_key in {"plugin", "onestep"}:
                    repeat_items = [repeat["summary"]["metrics"]["estimators"][method_key] for repeat in repeats]
                else:
                    repeat_items = [repeat["summary"]["metrics"]["ablations"][method_key] for repeat in repeats]
                metrics = _aggregate_scalar_estimator(
                    repeat_items,
                    estimate_key="estimate_mean",
                    bias_key="bias_mean",
                    ci_coverage_key="ci_coverage",
                    ci_length_key="ci_length_mean",
                )
                long_rows.append(
                    {
                        "functional": functional,
                        "rho": rho_value,
                        "rho_label": rho_label,
                        "method": method_label,
                        "rmse_mean": metrics["rmse_mean"],
                        "rmse_std": metrics["rmse_std"],
                        "bias_mean": metrics["bias_mean"],
                        "bias_std": metrics["bias_std"],
                        "ci_coverage": metrics["ci_coverage"],
                        "ci_length_mean": metrics["ci_length_mean"],
                        "ci_length_std": metrics["ci_length_std"],
                    }
                )
                rmse_row[method_label] = _format_value(metrics["rmse_mean"], metrics["rmse_std"])
                bias_row[method_label] = _format_value(metrics["bias_mean"], metrics["bias_std"])
                if functional == "auc" and rho_label == "r000" and method_key in {"plugin", "onestep"}:
                    representative_diffs[f"{functional}_{rho_label}_{method_key}"] = _compare_value(
                        old_long_rows,
                        {
                            "functional": functional,
                            "rho_label": rho_label,
                            "method": method_label,
                        },
                        "rmse_mean",
                        float(metrics["rmse_mean"]),
                    )
            rmse_wide_rows.append(rmse_row)
            bias_wide_rows.append(bias_row)

    long_csv = report_root / "pk_main_long.csv"
    rmse_csv = report_root / "pk_main_rmse_wide.csv"
    bias_csv = report_root / "pk_main_bias_wide.csv"
    manifest_path = report_root / "pk_main_manifest.json"

    _write_csv(
        long_csv,
        long_rows,
        ["functional", "rho", "rho_label", "method", "rmse_mean", "rmse_std", "bias_mean", "bias_std", "ci_coverage", "ci_length_mean", "ci_length_std"],
    )
    _write_csv(rmse_csv, rmse_wide_rows, ["functional", "rho"] + [label for _, label in ABLATION_METHODS])
    _write_csv(bias_csv, bias_wide_rows, ["functional", "rho"] + [label for _, label in ABLATION_METHODS])

    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source_summaries": source_summaries,
                "artifacts": {
                    "long_csv": str(long_csv.relative_to(ROOT)),
                    "rmse_wide_csv": str(rmse_csv.relative_to(ROOT)),
                    "bias_wide_csv": str(bias_csv.relative_to(ROOT)),
                },
                "representative_rmse_diffs": representative_diffs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "package": "pk_main",
        "artifacts": [long_csv, rmse_csv, bias_csv, manifest_path],
        "representative_diffs": representative_diffs,
    }


def _sorted_level_entries(level_payload: dict[str, dict]) -> list[dict]:
    return [entry for _, entry in sorted(level_payload.items(), key=lambda item: float(item[1]["level"]))]


def _recompute_nuisance_reports() -> dict:
    report_root = _resolve_existing_directory(REPORTS_ROOT, "pk_nuisance_r500", "pk_nuisance_r500*")
    report_root.mkdir(parents=True, exist_ok=True)
    old_solution_path = _resolve_existing_file(report_root, "pk_nuisance_solution_bias.csv", "pk_nuisance*_solution_bias.csv")
    old_solution_rows = list(csv.DictReader(old_solution_path.open(encoding="utf-8", newline="")))

    solution_rows: list[dict] = []
    beta_rows: list[dict] = []
    joint_rows: list[dict] = []
    source_summaries: dict[str, str] = {}
    representative_diffs: dict[str, dict[str, float | None]] = {}

    for functional in NUISANCE_FUNCTIONALS:
        summary_path = _nuisance_summary_path(functional)
        summary = _load_json(summary_path)
        source_summaries[functional] = str(summary_path.relative_to(ROOT))
        repeats = summary["repeats"]

        for level_entry in _sorted_level_entries(summary["repeats"][0]["summary"]["nuisance_study"]["solution_bias"]):
            level = float(level_entry["level"])
            repeat_items = [repeat["summary"]["nuisance_study"]["solution_bias"][f"{level:.4f}"] for repeat in repeats]
            plugin_metrics = _aggregate_scalar_estimator_no_ci(
                repeat_items,
                estimate_key="plugin_estimate",
                bias_key="plugin_bias",
            )
            onestep_metrics = _aggregate_scalar_estimator_no_ci(
                repeat_items,
                estimate_key="onestep_estimate",
                bias_key="onestep_bias",
            )
            solution_rows.append(
                {
                    "functional": functional,
                    "corruption_level": level,
                    "solution_rmse_mean": _mean([float(item["solution_rmse"]) for item in repeat_items]),
                    "solution_rmse_std": _optional_std([float(item["solution_rmse"]) for item in repeat_items]),
                    "plugin_rmse_mean": plugin_metrics["rmse_mean"],
                    "plugin_rmse_std": plugin_metrics["rmse_std"],
                    "plugin_bias_mean": plugin_metrics["bias_mean"],
                    "plugin_bias_std": plugin_metrics["bias_std"],
                    "onestep_rmse_mean": onestep_metrics["rmse_mean"],
                    "onestep_rmse_std": onestep_metrics["rmse_std"],
                    "onestep_bias_mean": onestep_metrics["bias_mean"],
                    "onestep_bias_std": onestep_metrics["bias_std"],
                }
            )
            if functional == "auc" and math.isclose(level, 0.0):
                representative_diffs[f"{functional}_solution_{level:.1f}_plugin"] = _compare_value(
                    old_solution_rows,
                    {
                        "functional": functional,
                        "corruption_level": level,
                    },
                    "plugin_rmse_mean",
                    float(plugin_metrics["rmse_mean"]),
                )

        for level_entry in _sorted_level_entries(summary["repeats"][0]["summary"]["nuisance_study"]["beta_bias"]):
            level = float(level_entry["level"])
            repeat_items = [repeat["summary"]["nuisance_study"]["beta_bias"][f"{level:.4f}"] for repeat in repeats]
            onestep_biases = [float(item["onestep_bias"]) for item in repeat_items]
            beta_rows.append(
                {
                    "functional": functional,
                    "corruption_level": level,
                    "beta_rmse_mean": _mean([float(item["beta_rmse"]) for item in repeat_items]),
                    "beta_rmse_std": _optional_std([float(item["beta_rmse"]) for item in repeat_items]),
                    "onestep_rmse_mean": _root_mean_square(onestep_biases),
                    "onestep_rmse_std": _optional_std([abs(value) for value in onestep_biases]),
                    "onestep_bias_mean": _mean(onestep_biases),
                    "onestep_bias_std": _optional_std(onestep_biases),
                }
            )

        for level_entry in _sorted_level_entries(summary["repeats"][0]["summary"]["nuisance_study"]["joint_bias"]):
            level = float(level_entry["level"])
            repeat_items = [repeat["summary"]["nuisance_study"]["joint_bias"][f"{level:.4f}"] for repeat in repeats]
            onestep_biases = [float(item["onestep_bias"]) for item in repeat_items]
            joint_rows.append(
                {
                    "functional": functional,
                    "corruption_level": level,
                    "solution_rmse_mean": _mean([float(item["solution_rmse"]) for item in repeat_items]),
                    "solution_rmse_std": _optional_std([float(item["solution_rmse"]) for item in repeat_items]),
                    "beta_rmse_mean": _mean([float(item["beta_rmse"]) for item in repeat_items]),
                    "beta_rmse_std": _optional_std([float(item["beta_rmse"]) for item in repeat_items]),
                    "onestep_rmse_mean": _root_mean_square(onestep_biases),
                    "onestep_rmse_std": _optional_std([abs(value) for value in onestep_biases]),
                    "onestep_bias_mean": _mean(onestep_biases),
                    "onestep_bias_std": _optional_std(onestep_biases),
                }
            )

    solution_csv = report_root / "pk_nuisance_solution_bias_corrected.csv"
    beta_csv = report_root / "pk_nuisance_beta_bias_corrected.csv"
    joint_csv = report_root / "pk_nuisance_joint_bias_corrected.csv"
    manifest_path = report_root / "pk_nuisance_manifest_corrected.json"

    _write_csv(
        solution_csv,
        solution_rows,
        ["functional", "corruption_level", "solution_rmse_mean", "solution_rmse_std", "plugin_rmse_mean", "plugin_rmse_std", "plugin_bias_mean", "plugin_bias_std", "onestep_rmse_mean", "onestep_rmse_std", "onestep_bias_mean", "onestep_bias_std"],
    )
    _write_csv(
        beta_csv,
        beta_rows,
        ["functional", "corruption_level", "beta_rmse_mean", "beta_rmse_std", "onestep_rmse_mean", "onestep_rmse_std", "onestep_bias_mean", "onestep_bias_std"],
    )
    _write_csv(
        joint_csv,
        joint_rows,
        ["functional", "corruption_level", "solution_rmse_mean", "solution_rmse_std", "beta_rmse_mean", "beta_rmse_std", "onestep_rmse_mean", "onestep_rmse_std", "onestep_bias_mean", "onestep_bias_std"],
    )

    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source_summaries": source_summaries,
                "artifacts": {
                    "solution_bias_csv": str(solution_csv.relative_to(ROOT)),
                    "beta_bias_csv": str(beta_csv.relative_to(ROOT)),
                    "joint_bias_csv": str(joint_csv.relative_to(ROOT)),
                },
                "representative_rmse_diffs": representative_diffs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "package": "pk_nuisance_r500",
        "artifacts": [solution_csv, beta_csv, joint_csv, manifest_path],
        "representative_diffs": representative_diffs,
    }


def _recompute_ppi_reports() -> dict:
    report_root = _resolve_existing_directory(REPORTS_ROOT, "pk_ppi_r500", "pk_ppi*_r500*")
    report_root.mkdir(parents=True, exist_ok=True)
    old_long_path = _resolve_existing_file(report_root, "pk_ppi_r500_long.csv", "pk_ppi*_r500*_long.csv")
    old_long_rows = list(csv.DictReader(old_long_path.open(encoding="utf-8", newline="")))

    long_rows: list[dict] = []
    source_summaries: dict[str, str] = {}
    representative_diffs: dict[str, dict[str, float | None]] = {}

    for functional in PPI_FUNCTIONALS:
        summary_path = _ppi_summary_path(functional)
        summary = _load_json(summary_path)
        source_summaries[functional] = str(summary_path.relative_to(ROOT))
        repeats = summary["repeats"]
        for multiplier, n2 in zip(PPI_MULTIPLIERS, PPI_N2_GRID):
            repeat_items = [repeat["summary"]["ppi_study"][str(n2)] for repeat in repeats]
            plugin_metrics = _aggregate_scalar_estimator(
                repeat_items,
                estimate_key="plugin_estimate",
                bias_key="plugin_bias",
                ci_coverage_key="plugin_ci_covers_true_theta",
                ci_length_key="plugin_ci_length",
            )
            onestep_metrics = _aggregate_scalar_estimator(
                repeat_items,
                estimate_key="onestep_estimate",
                bias_key="onestep_bias",
                ci_coverage_key="onestep_ci_covers_true_theta",
                ci_length_key="onestep_ci_length",
            )
            long_rows.append(
                {
                    "functional": functional,
                    "rho": 0.5,
                    "n1": int(repeat_items[0]["n1"]),
                    "truth_pool_size": 2000,
                    "multiplier": multiplier,
                    "n2": int(repeat_items[0]["n2"]),
                    "total_plugin_samples": int(repeat_items[0]["total_plugin_samples"]),
                    "plugin_estimate_mean": plugin_metrics["estimate_mean"],
                    "plugin_estimate_std": plugin_metrics["estimate_std"],
                    "plugin_rmse_mean": plugin_metrics["rmse_mean"],
                    "plugin_rmse_std": plugin_metrics["rmse_std"],
                    "plugin_bias_mean": plugin_metrics["bias_mean"],
                    "plugin_bias_std": plugin_metrics["bias_std"],
                    "plugin_ci_coverage_mean": plugin_metrics["ci_coverage"],
                    "plugin_ci_length_mean": plugin_metrics["ci_length_mean"],
                    "onestep_estimate_mean": onestep_metrics["estimate_mean"],
                    "onestep_estimate_std": onestep_metrics["estimate_std"],
                    "onestep_rmse_mean": onestep_metrics["rmse_mean"],
                    "onestep_rmse_std": onestep_metrics["rmse_std"],
                    "onestep_bias_mean": onestep_metrics["bias_mean"],
                    "onestep_bias_std": onestep_metrics["bias_std"],
                    "onestep_ci_coverage_mean": onestep_metrics["ci_coverage"],
                    "onestep_ci_length_mean": onestep_metrics["ci_length_mean"],
                }
            )
            if functional == "auc" and multiplier == 0:
                representative_diffs[f"{functional}_m{multiplier}_plugin"] = _compare_value(
                    old_long_rows,
                    {"functional": functional, "multiplier": multiplier},
                    "plugin_rmse_mean",
                    float(plugin_metrics["rmse_mean"]),
                )

    long_csv = report_root / "pk_ppi_r500_long_corrected.csv"
    manifest_path = report_root / "pk_ppi_r500_manifest_corrected.json"
    _write_csv(
        long_csv,
        long_rows,
        [
            "functional",
            "rho",
            "n1",
            "truth_pool_size",
            "multiplier",
            "n2",
            "total_plugin_samples",
            "plugin_estimate_mean",
            "plugin_estimate_std",
            "plugin_rmse_mean",
            "plugin_rmse_std",
            "plugin_bias_mean",
            "plugin_bias_std",
            "plugin_ci_coverage_mean",
            "plugin_ci_length_mean",
            "onestep_estimate_mean",
            "onestep_estimate_std",
            "onestep_rmse_mean",
            "onestep_rmse_std",
            "onestep_bias_mean",
            "onestep_bias_std",
            "onestep_ci_coverage_mean",
            "onestep_ci_length_mean",
        ],
    )

    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source_summaries": source_summaries,
                "artifacts": {
                    "long_csv": str(long_csv.relative_to(ROOT)),
                },
                "representative_rmse_diffs": representative_diffs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "package": "pk_ppi_r500",
        "artifacts": [long_csv, manifest_path],
        "representative_diffs": representative_diffs,
    }


def main() -> None:
    outputs = [
        _recompute_main_reports(),
        _recompute_nuisance_reports(),
        _recompute_ppi_reports(),
    ]
    for output in outputs:
        for artifact in output["artifacts"]:
            print(artifact)
        print(json.dumps({"package": output["package"], "representative_diffs": output["representative_diffs"]}, indent=2))


if __name__ == "__main__":
    main()

