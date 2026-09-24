from __future__ import annotations

import unittest

import torch

from ono.functionals import get_functional, riesz_density


class FunctionalTests(unittest.TestCase):
    def test_auc_riesz_density_is_one(self) -> None:
        solution = torch.tensor([[0.1, 0.2, 0.3]])
        weights = torch.full_like(solution, 1 / 3)
        self.assertTrue(torch.equal(riesz_density("auc", solution, weights), torch.ones_like(solution)))

    def test_soft_cmax_derivative_uses_configured_lambda(self) -> None:
        solution = torch.tensor([[0.0, 1.0]])
        weights = torch.full_like(solution, 0.5)
        value = get_functional("soft_cmax")(solution, weights, lam=2.0)
        derivative = riesz_density("soft_cmax", solution, weights, lam=2.0)
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.isfinite(derivative).all())
        self.assertAlmostEqual(float(torch.sum(weights * derivative)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
