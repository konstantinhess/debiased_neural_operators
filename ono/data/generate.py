from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Any

import numpy as np

from ono.data.poisson_2d import (
    Poisson2DConfig,
    dataset_metadata as poisson_dataset_metadata,
    generate_split as poisson_generate_split,
)
from ono.data.poisson_1d import (
    Poisson1DConfig,
    dataset_metadata as poisson_1d_dataset_metadata,
    generate_split as poisson_1d_generate_split,
)
from ono.data.darcy_2d import (
    Darcy2DConfig,
    dataset_metadata as darcy_2d_dataset_metadata,
    generate_split as darcy_2d_generate_split,
)
from ono.data.navier_stokes_2d import (
    NavierStokes2DConfig,
    dataset_metadata as navier_stokes_2d_dataset_metadata,
    generate_split as navier_stokes_2d_generate_split,
)
from ono.data.burgers_1d import (
    Burgers1DConfig,
    dataset_metadata as burgers_1d_dataset_metadata,
    generate_split as burgers_1d_generate_split,
)
from ono.data.pharmacokinetics_1d import (
    Pharmacokinetics1DConfig,
    dataset_metadata as pharmacokinetics_dataset_metadata,
    generate_split as pharmacokinetics_generate_split,
)
from ono.data.reaction_diffusion import (
    ReactionDiffusionConfig,
    dataset_metadata as reaction_dataset_metadata,
    generate_split as reaction_generate_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark datasets from a dataset JSON config.")
    parser.add_argument("--config", required=True, help="Path to a dataset JSON config.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the config output root.",
    )
    return parser.parse_args()


def load_dataset_config(
    path: str | Path,
) -> ReactionDiffusionConfig | Poisson2DConfig | Pharmacokinetics1DConfig | Poisson1DConfig | Darcy2DConfig | Burgers1DConfig:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    pde_class = str(payload["equation"]["pde_class"])
    if pde_class == "reaction_diffusion_1d":
        return ReactionDiffusionConfig.from_dict(payload)
    if pde_class == "poisson_1d":
        return Poisson1DConfig.from_dict(payload)
    if pde_class == "poisson_2d":
        return Poisson2DConfig.from_dict(payload)
    if pde_class == "darcy_2d":
        return Darcy2DConfig.from_dict(payload)
    if pde_class == "navier_stokes_2d":
        return NavierStokes2DConfig.from_dict(payload)
    if pde_class == "burgers_1d":
        return Burgers1DConfig.from_dict(payload)
    if pde_class == "pharmacokinetics_1d":
        return Pharmacokinetics1DConfig.from_dict(payload)
    raise ValueError(f"unsupported dataset pde_class: {pde_class}")


def write_dataset(
    config: ReactionDiffusionConfig | Poisson2DConfig | Pharmacokinetics1DConfig | Poisson1DConfig | Darcy2DConfig | NavierStokes2DConfig | Burgers1DConfig,
) -> Path:
    output_root = Path(config.output_root)
    dataset_root = output_root / config.dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    if isinstance(config, ReactionDiffusionConfig):
        metadata_fn = reaction_dataset_metadata
        split_fn = reaction_generate_split
    elif isinstance(config, Poisson1DConfig):
        metadata_fn = poisson_1d_dataset_metadata
        split_fn = poisson_1d_generate_split
    elif isinstance(config, Darcy2DConfig):
        metadata_fn = darcy_2d_dataset_metadata
        split_fn = darcy_2d_generate_split
    elif isinstance(config, NavierStokes2DConfig):
        metadata_fn = navier_stokes_2d_dataset_metadata
        split_fn = navier_stokes_2d_generate_split
    elif isinstance(config, Burgers1DConfig):
        metadata_fn = burgers_1d_dataset_metadata
        split_fn = burgers_1d_generate_split
    elif isinstance(config, Pharmacokinetics1DConfig):
        metadata_fn = pharmacokinetics_dataset_metadata
        split_fn = pharmacokinetics_generate_split
    else:
        metadata_fn = poisson_dataset_metadata
        split_fn = poisson_generate_split

    (dataset_root / "metadata.json").write_text(
        json.dumps(metadata_fn(config), indent=2),
        encoding="utf-8",
    )

    seed_sequence = np.random.SeedSequence(config.seed)
    split_sequences = seed_sequence.spawn(len(config.splits))

    for (split_name, num_samples), split_seed in zip(config.splits.items(), split_sequences):
        split_rng = np.random.default_rng(split_seed)
        arrays = split_fn(
            config=config,  # type: ignore[arg-type]
            split_name=split_name,
            num_samples=num_samples,
            rng=split_rng,
        )
        np.savez_compressed(dataset_root / f"{split_name}.npz", **arrays)

    return dataset_root


def main() -> None:
    args = parse_args()
    config = load_dataset_config(args.config)
    if args.output_root is not None:
        config = config.__class__(
            **{**config.__dict__, "output_root": args.output_root},
        )
    dataset_root = write_dataset(config)
    print(dataset_root)


if __name__ == "__main__":
    main()
