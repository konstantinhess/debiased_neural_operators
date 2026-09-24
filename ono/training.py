from __future__ import annotations

import random

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ono.models import DebiasingOperatorModule, NuisanceModelBundleConfig, SolutionOperatorModule


class SparseNuisanceDataset(Dataset):
    """Expose only sparse observations and covariates to nuisance training."""

    ALLOWED = {"coeff", "x_grid", "obs_x", "obs_y", "obs_index", "quad_weights"}

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[index]
        return {key: value for key, value in sample.items() if key in self.ALLOWED}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


class DebiasingTrainingDataset(Dataset):
    """Training data for beta; it contains only train/validation samples."""

    def __init__(self, base: Dataset, solution: SolutionOperatorModule) -> None:
        self.base = base
        self.solution = solution
        self.solution.freeze_model()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[index]
        with torch.no_grad():
            u_hat = self.solution.predict_grid(
                sample["coeff"].unsqueeze(0), sample["x_grid"].unsqueeze(0)
            ).squeeze(0)
        return {**sample, "u_hat": u_hat}


class BestValidationState(pl.Callback):
    """Restore the lowest-validation-loss state from a fixed epoch budget."""

    def __init__(self) -> None:
        super().__init__()
        self.best_loss = float("inf")
        self.best_epoch: int | None = None
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_penalty: float | None = None

    def on_validation_epoch_end(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get("val_loss")
        if metric is None:
            return
        loss = float(metric.detach().cpu())
        if loss >= self.best_loss:
            return
        self.best_loss = loss
        self.best_epoch = int(trainer.current_epoch)
        self.best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        penalty = trainer.callback_metrics.get("val_riesz_penalty")
        self.best_penalty = None if penalty is None else float(penalty.detach().cpu())

    def restore(self, module: pl.LightningModule) -> dict[str, float | int | None]:
        if self.best_state is None:
            raise RuntimeError("validation did not produce a selectable model state")
        module.load_state_dict(self.best_state)
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_loss,
            "best_val_riesz_penalty": self.best_penalty,
            "best_val_unpenalized_loss": (
                None if self.best_penalty is None else self.best_loss - self.best_penalty
            ),
        }


def fit_with_validation(
    module: pl.LightningModule,
    train: Dataset,
    val: Dataset,
    batch_size: int,
    max_epochs: int,
) -> dict[str, float | int | None]:
    selector = BestValidationState()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
        inference_mode=False,
        callbacks=[selector],
    )
    trainer.fit(
        module,
        train_dataloaders=DataLoader(train, batch_size=batch_size, shuffle=True),
        val_dataloaders=DataLoader(val, batch_size=batch_size, shuffle=False),
    )
    return selector.restore(module)


def train_solution_and_riesz(
    train: Dataset,
    val: Dataset,
    bundle: NuisanceModelBundleConfig,
    functional: str,
    functional_params: dict[str, float],
    batch_size: int,
    debiasing_mode: str,
) -> tuple[SolutionOperatorModule, DebiasingOperatorModule, dict]:
    solution = SolutionOperatorModule(bundle.model, bundle.solution_module)
    solution_selection = fit_with_validation(
        solution, train, val, batch_size, bundle.solution_module.max_epochs
    )
    solution.freeze_model()

    riesz_train = DebiasingTrainingDataset(train, solution)
    riesz_val = DebiasingTrainingDataset(val, solution)
    riesz = DebiasingOperatorModule(
        bundle.model,
        bundle.debiasing_module,
        functional_name=functional,
        functional_params=functional_params,
        frozen_solution_model=solution.model,
        mode=debiasing_mode,
    )
    riesz_selection = fit_with_validation(
        riesz, riesz_train, riesz_val, batch_size, bundle.debiasing_module.max_epochs
    )
    return solution, riesz, {
        "solution": solution_selection,
        "riesz": riesz_selection,
    }


def train_additional_riesz(
    mode: str,
    train: Dataset,
    val: Dataset,
    solution: SolutionOperatorModule,
    bundle: NuisanceModelBundleConfig,
    functional: str,
    functional_params: dict[str, float],
    batch_size: int,
) -> tuple[DebiasingOperatorModule, dict]:
    riesz = DebiasingOperatorModule(
        bundle.model,
        bundle.debiasing_module,
        functional_name=functional,
        functional_params=functional_params,
        frozen_solution_model=solution.model,
        mode=mode,
    )
    selection = fit_with_validation(
        riesz,
        DebiasingTrainingDataset(train, solution),
        DebiasingTrainingDataset(val, solution),
        batch_size,
        bundle.debiasing_module.max_epochs,
    )
    return riesz, selection
