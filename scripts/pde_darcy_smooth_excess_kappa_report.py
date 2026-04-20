from __future__ import annotations

import csv
from datetime import datetime
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "results" / "reports" / "pde_darcy_smooth_excess_kappa"
KAPPA_SPECS = [
    ("k0p0", 0.0),
    ("k0p2", 0.2),
    ("k0p4", 0.4),
    ("k0p6", 0.6),
    ("k0p8", 0.8),
    ("k1p0", 1.0),
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


def _root_mean_square(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _summary_path(kappa_label: str) -> Path:
    preferred = (
        ROOT
        / "results"
        / "coverage"
        / f"pde_twobench_darcy_2d_smooth_excess_above_threshold_inverseenergy_r1000_{kappa_label}_repeats50"
        / "summary.json"
    )
    if preferred.exists():
        return preferred
    coverage_root = ROOT / "results" / "coverage"
    matches = sorted(
        coverage_root.glob(
            f"pde_twobench_darcy_2d_smooth_excess_above_threshold_inverseenergy_r1000*{kappa_label}_repeats50/summary.json"
        )
    )
    if matches:
        return matches[0]
    return preferred


def _aggregate_estimator(repeats: list[dict], estimator_key: str) -> dict[str, float]:
    estimator_items = [repeat["summary"]["metrics"]["estimators"][estimator_key] for repeat in repeats]
    estimates = [float(item["estimate_mean"]) for item in estimator_items]
    biases = [float(item["bias_mean"]) for item in estimator_items]
    ci_coverages = [float(item["ci_coverage"]) for item in estimator_items]
    ci_lengths = [float(item["ci_length_mean"]) for item in estimator_items]
    return {
        "estimate_mean": _mean(estimates),
        "estimate_std": _sample_std(estimates),
        "bias_mean": _mean(biases),
        "bias_std": _sample_std(biases),
        "rmse_mean": _root_mean_square(biases),
        "rmse_std": _sample_std([abs(value) for value in biases]),
        "ci_coverage": _mean(ci_coverages),
        "ci_length_mean": _mean(ci_lengths),
        "ci_length_std": _sample_std(ci_lengths),
    }


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    manifest_rows: list[dict] = []
    for visible_label, visible_kappa in KAPPA_SPECS:
        summary_path = _summary_path(visible_label)
        summary = _load_json(summary_path)
        repeats = summary["repeats"]
        plugin = _aggregate_estimator(repeats, "plugin")
        onestep = _aggregate_estimator(repeats, "onestep")
        rows.append(
            {
                "kappa": visible_kappa,
                "repeat_count": 50,
                "plugin_estimate_mean": plugin["estimate_mean"],
                "plugin_estimate_std": plugin["estimate_std"],
                "plugin_rmse_mean": plugin["rmse_mean"],
                "plugin_rmse_std": plugin["rmse_std"],
                "plugin_bias_mean": plugin["bias_mean"],
                "plugin_bias_std": plugin["bias_std"],
                "plugin_ci_coverage": plugin["ci_coverage"],
                "plugin_ci_length_mean": plugin["ci_length_mean"],
                "plugin_ci_length_std": plugin["ci_length_std"],
                "onestep_estimate_mean": onestep["estimate_mean"],
                "onestep_estimate_std": onestep["estimate_std"],
                "onestep_rmse_mean": onestep["rmse_mean"],
                "onestep_rmse_std": onestep["rmse_std"],
                "onestep_bias_mean": onestep["bias_mean"],
                "onestep_bias_std": onestep["bias_std"],
                "onestep_ci_coverage": onestep["ci_coverage"],
                "onestep_ci_length_mean": onestep["ci_length_mean"],
                "onestep_ci_length_std": onestep["ci_length_std"],
                "onestep_minus_plugin_rmse": onestep["rmse_mean"] - plugin["rmse_mean"],
            }
        )
        manifest_rows.append(
            {
                "kappa": visible_kappa,
                "summary": str(summary_path.relative_to(ROOT)),
            }
        )

    csv_path = REPORT_ROOT / "pde_darcy_smooth_excess_inverseenergy_kappa_repeats50.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)

    manifest_path = REPORT_ROOT / "pde_darcy_smooth_excess_inverseenergy_kappa_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "mappings": manifest_rows,
                "artifact": str(csv_path.relative_to(ROOT)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()

