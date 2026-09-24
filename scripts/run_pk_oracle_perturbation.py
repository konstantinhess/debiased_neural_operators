from __future__ import annotations

import argparse

from _common import ROOT
from ono.oracle_perturbation import run_oracle_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the oracle-centered smooth-TAT perturbation study.")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--truth-size", type=int)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    print(run_oracle_study(
        ROOT / "configs" / "pk_oracle_perturbation.json",
        repeats=args.repeats,
        test_size=args.test_size,
        truth_size=args.truth_size,
        output_root=args.output_root,
    ))


if __name__ == "__main__":
    main()

