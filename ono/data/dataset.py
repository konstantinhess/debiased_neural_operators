from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class GeneratedSplitDataset(Dataset):
    SHARED_ARRAY_KEYS = {"x_grid", "quad_weights"}

    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root)
        self.split = split
        self.metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        split_path = self.root / f"{split}.npz"
        if not split_path.exists():
            raise FileNotFoundError(f"missing split file: {split_path}")
        with np.load(split_path) as loaded:
            self.arrays = {key: loaded[key] for key in loaded.files}
        self.length = int(self.arrays["coeff"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {}
        for key, value in self.arrays.items():
            if key == "split_name":
                continue
            if key not in self.SHARED_ARRAY_KEYS and value.shape[:1] == (self.length,):
                item = value[index]
                if isinstance(item, np.ndarray):
                    sample[key] = torch.from_numpy(item)
                else:
                    sample[key] = torch.as_tensor(item)
            else:
                sample[key] = torch.from_numpy(value)
        return sample


ReactionDiffusionSplitDataset = GeneratedSplitDataset

