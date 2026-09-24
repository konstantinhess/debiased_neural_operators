from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
from typing import Any

import numpy as np

from ono.data.darcy_2d import Darcy2DConfig, dataset_metadata as darcy_metadata, generate_split as darcy_split
from ono.data.pharmacokinetics_1d import (
    Pharmacokinetics1DConfig,
    dataset_metadata as pk_metadata,
    generate_split as pk_split,
)


DatasetConfig = Pharmacokinetics1DConfig | Darcy2DConfig


def load_dataset_config(payload: dict[str, Any]) -> DatasetConfig:
    pde_class = str(payload["equation"]["pde_class"])
    if pde_class == "pharmacokinetics_1d":
        return Pharmacokinetics1DConfig.from_dict(payload)
    if pde_class == "darcy_2d":
        return Darcy2DConfig.from_dict(payload)
    raise ValueError(f"unsupported paper dataset: {pde_class}")


def with_dataset_overrides(config: DatasetConfig, **overrides: Any) -> DatasetConfig:
    return replace(config, **overrides)


def write_dataset(config: DatasetConfig) -> Path:
    root = Path(config.output_root) / config.dataset_name
    root.mkdir(parents=True, exist_ok=True)
    if isinstance(config, Pharmacokinetics1DConfig):
        metadata_fn, split_fn = pk_metadata, pk_split
    elif isinstance(config, Darcy2DConfig):
        metadata_fn, split_fn = darcy_metadata, darcy_split
    else:
        raise TypeError(type(config))

    (root / "metadata.json").write_text(json.dumps(metadata_fn(config), indent=2), encoding="utf-8")
    seed_sequence = np.random.SeedSequence(config.seed)
    split_sequences = seed_sequence.spawn(len(config.splits))
    for (split_name, size), split_seed in zip(config.splits.items(), split_sequences):
        arrays = split_fn(config, split_name, size, np.random.default_rng(split_seed))
        np.savez_compressed(root / f"{split_name}.npz", **arrays)
    return root
