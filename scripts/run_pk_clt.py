from __future__ import annotations

import argparse
from pathlib import Path
import csv
import json

from _common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the AUC confidence-interval table from the main PK run."
    )
    parser.add_argument(
        "--summary",
        default=str(ROOT / "outputs" / "pk_main_auc" / "summary.json"),
        help="Main AUC summary produced by run_pk_main.py.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "pk_clt_auc"),
    )
    args = parser.parse_args()
    source = Path(args.summary)
    if not source.exists():
        raise FileNotFoundError(
            f"{source} does not exist. Run 'python scripts/run_pk_main.py --functional auc' first."
        )
    main_summary = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for rho, condition in main_summary["conditions"].items():
        for method in ("plugin", "dope"):
            metrics = condition["methods"][method]
            rows.append({
                "rho": float(rho),
                "method": method,
                "coverage": metrics["coverage"],
                "mean_interval_length": metrics["mean_interval_length"],
                "interval_length_std": metrics["interval_length_std"],
                "num_repeats": condition["num_repeats"],
            })
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps({"source": str(source), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
