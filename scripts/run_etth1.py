from __future__ import annotations

import argparse

from _common import ROOT, add_smoke_overrides, parse_float_list
from ono.etth1 import run_etth1_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the final ETTh1 semi-synthetic experiments.")
    parser.add_argument("--functional", choices=["auc", "soft_cmax", "all"], default="all")
    parser.add_argument("--data", help="Override data/ETTh1.csv.")
    parser.add_argument("--rho", help="Optional comma-separated subset of rho values.")
    add_smoke_overrides(parser, include_sizes=False)
    args = parser.parse_args()
    names = ["auc", "soft_cmax"] if args.functional == "all" else [args.functional]
    for name in names:
        output = run_etth1_study(
            ROOT / "configs" / f"etth1_{name}.json",
            data_path=args.data,
            repeats=args.repeats,
            epochs=args.epochs,
            rhos=parse_float_list(args.rho),
            output_root=args.output_root,
        )
        print(output)


if __name__ == "__main__":
    main()
