from __future__ import annotations

import argparse

from _common import ROOT, add_smoke_overrides, parse_float_list
from ono.synthetic import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the main FNO PK experiments.")
    parser.add_argument("--functional", choices=["auc", "smooth_tat", "all"], default="all")
    parser.add_argument("--rho", help="Optional comma-separated subset of rho values.")
    add_smoke_overrides(parser)
    args = parser.parse_args()
    names = ["auc", "smooth_tat"] if args.functional == "all" else [args.functional]
    for name in names:
        output = run_study(
            ROOT / "configs" / f"pk_main_{name}.json",
            repeats=args.repeats,
            epochs=args.epochs,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            truth_size=args.truth_size,
            conditions=parse_float_list(args.rho),
            output_root=args.output_root,
        )
        print(output)


if __name__ == "__main__":
    main()

