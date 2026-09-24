from __future__ import annotations

import argparse

from _common import ROOT, add_smoke_overrides, parse_float_list
from ono.synthetic import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the final Darcy-flow experiment.")
    parser.add_argument("--kappa", help="Optional comma-separated subset of visible kappa values.")
    add_smoke_overrides(parser)
    args = parser.parse_args()
    print(run_study(
        ROOT / "configs" / "darcy.json",
        repeats=args.repeats,
        epochs=args.epochs,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        truth_size=args.truth_size,
        conditions=parse_float_list(args.kappa),
        output_root=args.output_root,
    ))


if __name__ == "__main__":
    main()

