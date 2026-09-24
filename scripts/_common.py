from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def add_smoke_overrides(parser: argparse.ArgumentParser, *, include_sizes: bool = True) -> None:
    parser.add_argument("--repeats", type=int, help="Override the paper repeat count.")
    parser.add_argument("--epochs", type=int, help="Override both fixed epoch budgets.")
    if include_sizes:
        parser.add_argument("--train-size", type=int)
        parser.add_argument("--val-size", type=int)
        parser.add_argument("--test-size", type=int)
        parser.add_argument("--truth-size", type=int)
    parser.add_argument("--output-root", help="Override the configured output root.")


def parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in value.split(",")]

