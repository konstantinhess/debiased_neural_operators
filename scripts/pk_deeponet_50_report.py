from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = ROOT / "results" / "reports"
REPORT_ROOT = REPORTS_ROOT / "pk_main_deeponet"
FUNCTIONALS = ["auc", "smooth_tat"]
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
METHODS = [
    ("plugin", "plugin"),
    ("onestep", "one-step"),
    ("oracle_beta", "oracle_beta"),
    ("structured", "structured"),
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    center = _mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _optional_std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return None
    return _sample_std(values)


def _root_mean_square(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(mean: float | None, std: float | None) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.6f}"
    return f"{mean:.6f} +/- {std:.6f}"


def _resolve_existing_directory(parent: Path, preferred_name: str, glob_pattern: str) -> Path:
    preferred = parent / preferred_name
    if preferred.exists():
        return preferred
    matches = sorted(path for path in parent.glob(glob_pattern) if path.is_dir())
    if matches:
        return matches[0]
    return preferred


def _summary_path(functional: str, rho_label: str) -> Path:
    coverage_root = ROOT / "results" / "coverage"
    preferred = (
        coverage_root
        / f"pk_main_deeponet_{functional}_{rho_label}_repeats50"
        / "summary.json"
    )
    if preferred.exists():
        return preferred
    # Coverage outputs are intentionally not renamed, so the report script falls back to the legacy stored names.
    matches = sorted(
        coverage_root.glob(f"pk_ablation_deeponet_{functional}_irreg_unit_{rho_label}*repeats50*/summary.json")
    )
    if matches:
        return matches[0]
    return preferred


def _aggregate(repeats: list[dict], method_key: str) -> dict[str, float | None]:
    if method_key in {"plugin", "onestep"}:
        items = [repeat["summary"]["metrics"]["estimators"][method_key] for repeat in repeats]
    else:
        items = [repeat["summary"]["metrics"]["ablations"][method_key] for repeat in repeats]
    estimates = [float(item["estimate_mean"]) for item in items]
    biases = [float(item["bias_mean"]) for item in items]
    coverages = [float(item["ci_coverage"]) for item in items]
    ci_lengths = [float(item["ci_length_mean"]) for item in items]
    return {
        "estimate_mean": _mean(estimates),
        "estimate_std": _optional_std(estimates),
        "bias_mean": _mean(biases),
        "bias_std": _optional_std(biases),
        "rmse_mean": _root_mean_square(biases),
        "rmse_std": _optional_std([abs(value) for value in biases]),
        "ci_coverage": _mean(coverages),
        "ci_length_mean": _mean(ci_lengths),
        "ci_length_std": _optional_std(ci_lengths),
    }


def main() -> None:
    report_root = _resolve_existing_directory(REPORTS_ROOT, "pk_main_deeponet", "pk_ablation_deeponet*")
    report_root.mkdir(parents=True, exist_ok=True)
    long_rows: list[dict] = []
    rmse_wide_rows: list[dict] = []
    bias_wide_rows: list[dict] = []
    sources: dict[str, str] = {}

    for functional in FUNCTIONALS:
        for rho_label, rho_value in RHO_LABELS:
            summary_path = _summary_path(functional, rho_label)
            summary = _load_json(summary_path)
            repeats = summary["repeats"]
            sources[f"{functional}_{rho_label}"] = str(summary_path.relative_to(ROOT))
            rmse_row = {"functional": functional, "rho": rho_value}
            bias_row = {"functional": functional, "rho": rho_value}
            for method_key, method_label in METHODS:
                metrics = _aggregate(repeats, method_key)
                long_rows.append(
                    {
                        "functional": functional,
                        "rho": rho_value,
                        "rho_label": rho_label,
                        "method": method_label,
                        "repeat_count": len(repeats),
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
            rmse_wide_rows.append(rmse_row)
            bias_wide_rows.append(bias_row)

    long_csv = report_root / "pk_main_deeponet_long.csv"
    rmse_csv = report_root / "pk_main_deeponet_rmse_wide.csv"
    bias_csv = report_root / "pk_main_deeponet_bias_wide.csv"
    manifest_path = report_root / "pk_main_deeponet_manifest.json"

    _write_csv(
        long_csv,
        long_rows,
        ["functional", "rho", "rho_label", "method", "repeat_count", "rmse_mean", "rmse_std", "bias_mean", "bias_std", "ci_coverage", "ci_length_mean", "ci_length_std"],
    )
    _write_csv(rmse_csv, rmse_wide_rows, ["functional", "rho"] + [label for _, label in METHODS])
    _write_csv(bias_csv, bias_wide_rows, ["functional", "rho"] + [label for _, label in METHODS])
    manifest_path.write_text(
        json.dumps(
            {
                "source_summaries": sources,
                "artifacts": {
                    "long_csv": str(long_csv.relative_to(ROOT)),
                    "rmse_wide_csv": str(rmse_csv.relative_to(ROOT)),
                    "bias_wide_csv": str(bias_csv.relative_to(ROOT)),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(long_csv)
    print(rmse_csv)
    print(bias_csv)
    print(manifest_path)


if __name__ == "__main__":
    main()

