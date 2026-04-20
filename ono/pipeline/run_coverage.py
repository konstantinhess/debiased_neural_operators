from __future__ import annotations

import argparse

from ono.pipeline.config import Phase3Config
from ono.pipeline.coverage import run_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a unified pipeline config through the canonical pipeline runner."
    )
    parser.add_argument("--config", required=True, help="Path to a pipeline JSON config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Phase3Config.from_path(args.config)
    output_root = run_pipeline_config(config, pipeline_config_path=args.config)
    print(output_root)


if __name__ == "__main__":
    main()
