from __future__ import annotations

from pathlib import Path
import json
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset

from ono.etth1 import early_block_observations
from ono.synthetic import _ppi_payload
from ono.training import SparseNuisanceDataset


ROOT = Path(__file__).resolve().parents[1]


class _CompleteDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "coeff": torch.zeros(1, 4),
            "x_grid": torch.linspace(0, 1, 4),
            "obs_x": torch.tensor([0.0, 1.0]),
            "obs_y": torch.zeros(2),
            "obs_index": torch.tensor([0, 3]),
            "quad_weights": torch.full((4,), 0.25),
            "solution": torch.ones(4),
            "xi": torch.ones(4),
            "design_prob": torch.full((4,), 0.25),
        }


class ProtocolTests(unittest.TestCase):
    def test_training_view_excludes_full_outcomes_and_oracle_weights(self) -> None:
        sample = SparseNuisanceDataset(_CompleteDataset())[0]
        self.assertNotIn("solution", sample)
        self.assertNotIn("xi", sample)
        self.assertNotIn("design_prob", sample)
        self.assertIn("obs_y", sample)

    def test_etth1_early_block_is_without_replacement(self) -> None:
        future = np.zeros((20, 24), dtype=np.float32)
        indices, _, probabilities = early_block_observations(future, 8, 1.0, 1729)
        self.assertTrue(all(len(np.unique(row)) == 8 for row in indices))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

    def test_ppi_correction_uses_labeled_sample_only(self) -> None:
        result = _ppi_payload(
            [0, 2],
            labeled_plugin=[1.0, 3.0],
            labeled_correction=[0.5, 0.5],
            unlabeled_plugin=[5.0, 7.0],
            truth=0.0,
        )
        self.assertAlmostEqual(result["0"]["dope_estimate"], 2.5)
        self.assertAlmostEqual(result["2"]["plugin_estimate"], 4.0)
        self.assertAlmostEqual(result["2"]["dope_estimate"], 4.5)

    def test_all_configs_are_flat_and_final(self) -> None:
        configs = list((ROOT / "configs").glob("*.json"))
        self.assertEqual(len(configs), 10)
        for path in configs:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("base_config_path", payload)
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("legacy", text)
            self.assertNotIn("smoke", text)


if __name__ == "__main__":
    unittest.main()
