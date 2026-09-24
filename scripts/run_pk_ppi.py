from __future__ import annotations

import argparse

from _common import ROOT, add_smoke_overrides
from ono.synthetic import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the corrected PK PPI experiments.")
    parser.add_argument("--functional", choices=["auc", "soft_cmax", "all"], default="all")
    parser.add_argument("--n2", help="Optional comma-separated unlabeled-sample grid.")
    add_smoke_overrides(parser)
    args = parser.parse_args()
    names = ["auc", "soft_cmax"] if args.functional == "all" else [args.functional]
    for name in names:
        output = run_study(
            ROOT / "configs" / f"pk_ppi_{name}.json",
            repeats=args.repeats,
            epochs=args.epochs,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            truth_size=args.truth_size,
            unlabeled_counts=(
                [int(value) for value in args.n2.split(",")]
                if args.n2 is not None else None
            ),
            output_root=args.output_root,
        )
        print(output)


if __name__ == "__main__":
    main()
