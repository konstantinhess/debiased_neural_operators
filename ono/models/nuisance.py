from __future__ import annotations

from dataclasses import dataclass

import pytorch_lightning as pl
import torch

from ono.models.deeponet1d import DeepONet1dConfig, DeepONet1D
from ono.models.fno1d import FNO1dConfig, NeuralOperator1D
from ono.models.fno2d import FNO2dConfig, NeuralOperator2D
from ono.models.fno3d import FNO3dConfig, NeuralOperator3D
from ono.functionals import functional_jvp, riesz_density


Tensor = torch.Tensor


@dataclass(frozen=True)
class NuisanceModuleConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 20
    riesz_penalty_lambda: float = 1e-3


class BaseNuisanceModule(pl.LightningModule):
    def __init__(
        self,
        model_config: FNO1dConfig | FNO2dConfig | FNO3dConfig | DeepONet1dConfig | None = None,
        module_config: NuisanceModuleConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config or FNO1dConfig()
        self.module_config = module_config or NuisanceModuleConfig()
        if isinstance(self.model_config, FNO1dConfig):
            self.model = NeuralOperator1D(self.model_config)
        elif isinstance(self.model_config, DeepONet1dConfig):
            self.model = DeepONet1D(self.model_config)
        elif isinstance(self.model_config, FNO2dConfig):
            self.model = NeuralOperator2D(self.model_config)
        elif isinstance(self.model_config, FNO3dConfig):
            self.model = NeuralOperator3D(self.model_config)
        else:
            raise TypeError(f"unsupported model config type: {type(self.model_config)!r}")

    def predict_grid(self, coeff: Tensor, x_grid: Tensor) -> Tensor:
        return self.model.predict_grid(coeff, x_grid)

    def predict_points(self, coeff: Tensor, x_query: Tensor, x_grid: Tensor) -> Tensor:
        return self.model.predict_points(coeff, x_query, x_grid)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.module_config.learning_rate,
            weight_decay=self.module_config.weight_decay,
        )

    def freeze_model(self) -> None:
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False


class SolutionOperatorModule(BaseNuisanceModule):
    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        coeff = batch["coeff"]
        x_grid = batch["x_grid"]
        obs_x = batch["obs_x"]
        obs_y = batch["obs_y"]
        preds = self.predict_points(coeff, obs_x, x_grid)
        loss = torch.mean((preds - obs_y) ** 2)
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss


class DebiasingOperatorModule(BaseNuisanceModule):
    def __init__(
        self,
        model_config: FNO1dConfig | FNO2dConfig | FNO3dConfig | DeepONet1dConfig | None = None,
        module_config: NuisanceModuleConfig | None = None,
        functional_name: str | None = None,
        functional_params: dict[str, float] | None = None,
        frozen_solution_model: NeuralOperator1D | DeepONet1D | NeuralOperator2D | NeuralOperator3D | None = None,
        mode: str = "full",
    ) -> None:
        super().__init__(model_config=model_config, module_config=module_config)
        self.functional_name = functional_name
        self.functional_params = functional_params or {}
        self.frozen_solution_model = frozen_solution_model
        self.mode = mode
        if self.frozen_solution_model is not None:
            self.frozen_solution_model.eval()
            for param in self.frozen_solution_model.parameters():
                param.requires_grad = False

    def _compose_beta(
        self,
        coeff: Tensor,
        x_grid: Tensor,
        obs_x: Tensor,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        raw_grid = self.predict_grid(coeff, x_grid)
        raw_points = self.predict_points(coeff, obs_x, x_grid)

        if self.mode == "full":
            return raw_grid, raw_points

        if self.functional_name is None:
            raise ValueError("functional_name is required for structured and semi-oracle modes")

        if self.mode == "structured":
            if "u_hat" not in batch:
                raise KeyError("structured mode requires u_hat in batch")
            w_grid = _gridwise_riesz_density(
                self.functional_name,
                batch["u_hat"],
                batch["quad_weights"],
                batch["x_grid"],
                functional_params=self.functional_params,
            )
            w_points = interpolate_from_grid(x_grid, w_grid, obs_x)
            return raw_grid * w_grid, raw_points * w_points

        if self.mode == "oracle_wg":
            if "solution" not in batch:
                raise KeyError("oracle_wg mode requires solution in batch")
            w_grid = _gridwise_riesz_density(
                self.functional_name,
                batch["solution"],
                batch["quad_weights"],
                batch["x_grid"],
                functional_params=self.functional_params,
            )
            w_points = interpolate_from_grid(x_grid, w_grid, obs_x)
            return raw_grid * w_grid, raw_points * w_points

        if self.mode == "oracle_xi":
            xi_grid = batch["xi"]
            xi_points = interpolate_from_grid(x_grid, xi_grid, obs_x)
            return raw_grid * xi_grid, raw_points * xi_points

        raise ValueError(f"unsupported debiasing mode: {self.mode}")

    def compose_beta(
        self,
        coeff: Tensor,
        x_grid: Tensor,
        obs_x: Tensor,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        return self._compose_beta(coeff, x_grid, obs_x, batch)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        coeff = batch["coeff"]
        x_grid = batch["x_grid"]
        obs_x = batch["obs_x"]
        quad_weights = batch["quad_weights"]

        if self.functional_name is not None and self.frozen_solution_model is not None and "u_hat" in batch:
            u_hat = batch["u_hat"]
            beta_grid, beta_points = self._compose_beta(coeff, x_grid, obs_x, batch)

            if quad_weights.ndim == 2:
                quad_weights_batch = quad_weights
            elif quad_weights.ndim == u_hat.ndim - 1:
                quad_weights_batch = quad_weights.unsqueeze(0).expand(coeff.size(0), *quad_weights.shape)
            else:
                quad_weights_batch = quad_weights

            jvp_terms = []
            for idx in range(coeff.size(0)):
                u_hat_flat = batch["u_hat"][idx].reshape(-1)
                beta_flat = beta_grid[idx].reshape(-1)
                quad_flat = quad_weights_batch[idx].reshape(-1)
                x_grid_flat = _flatten_coordinates(batch["x_grid"][idx], batch["u_hat"][idx])
                jvp_terms.append(
                    functional_jvp(
                        self.functional_name,
                        u_hat_flat,
                        beta_flat,
                        quad_flat,
                        x_grid=x_grid_flat,
                        **self.functional_params,
                    )
                )
            jvp = torch.stack(jvp_terms, dim=0)
            quadratic = torch.mean(beta_points ** 2, dim=-1)
            penalty = self.module_config.riesz_penalty_lambda * torch.mean(beta_grid ** 2)
            loss = torch.mean(quadratic - 2.0 * jvp) + penalty
        else:
            target = batch.get("beta_target")
            if target is None:
                raise KeyError("batch must contain beta_target or u_hat for DebiasingOperatorModule training")
            preds = self.predict_grid(coeff, x_grid)
            mse = torch.mean((preds - target) ** 2)
            penalty = self.module_config.riesz_penalty_lambda * torch.mean(preds ** 2)
            loss = mse + penalty
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("riesz_penalty", penalty, prog_bar=False, on_step=False, on_epoch=True)
        return loss


def _flatten_coordinates(x_grid: Tensor, reference: Tensor) -> Tensor:
    if x_grid.ndim == reference.ndim:
        return x_grid.reshape(-1)
    if x_grid.ndim == reference.ndim + 1:
        return x_grid.reshape(-1, x_grid.size(-1))
    raise ValueError("x_grid shape is incompatible with the reference grid")


def _gridwise_riesz_density(
    functional_name: str,
    u_grid: Tensor,
    quad_weights: Tensor,
    x_grid: Tensor,
    functional_params: dict[str, float] | None = None,
) -> Tensor:
    batch_size = u_grid.size(0)
    flat = riesz_density(
        functional_name,
        u_grid.reshape(batch_size, -1),
        quad_weights.reshape(batch_size, -1),
        x_grid.reshape(batch_size, -1, x_grid.size(-1)) if x_grid.ndim == u_grid.ndim + 1 else x_grid.reshape(batch_size, -1),
        **(functional_params or {}),
    )
    return flat.reshape_as(u_grid)


def interpolate_from_grid(x_grid: Tensor, factor_grid: Tensor, obs_x: Tensor) -> Tensor:
    outputs = []
    for idx in range(factor_grid.size(0)):
        xg = x_grid[idx]
        fg = factor_grid[idx]
        oq = obs_x[idx]
        if xg.ndim == 1:
            clamped = torch.clamp(oq, min=float(xg[0]), max=float(xg[-1]))
            right = torch.searchsorted(xg, clamped, right=False)
            right = torch.clamp(right, 1, xg.numel() - 1)
            left = right - 1
            x_left = xg[left]
            x_right = xg[right]
            f_left = fg[left]
            f_right = fg[right]
            alpha = (clamped - x_left) / torch.clamp(x_right - x_left, min=1e-8)
            outputs.append(f_left + alpha * (f_right - f_left))
            continue

        if xg.ndim == 4:
            t_axis = xg[:, 0, 0, 0].contiguous()
            x_axis = xg[0, :, 0, 1].contiguous()
            y_axis = xg[0, 0, :, 2].contiguous()
            query_t = torch.clamp(oq[:, 0], min=float(t_axis[0]), max=float(t_axis[-1]))
            query_x = torch.clamp(oq[:, 1], min=float(x_axis[0]), max=float(x_axis[-1]))
            query_y = torch.clamp(oq[:, 2], min=float(y_axis[0]), max=float(y_axis[-1]))
            right_t = torch.clamp(torch.searchsorted(t_axis, query_t, right=False), 1, t_axis.numel() - 1)
            right_x = torch.clamp(torch.searchsorted(x_axis, query_x, right=False), 1, x_axis.numel() - 1)
            right_y = torch.clamp(torch.searchsorted(y_axis, query_y, right=False), 1, y_axis.numel() - 1)
            left_t = right_t - 1
            left_x = right_x - 1
            left_y = right_y - 1
            t0 = t_axis[left_t]
            t1 = t_axis[right_t]
            x0 = x_axis[left_x]
            x1 = x_axis[right_x]
            y0 = y_axis[left_y]
            y1 = y_axis[right_y]
            at = (query_t - t0) / torch.clamp(t1 - t0, min=1e-8)
            ax = (query_x - x0) / torch.clamp(x1 - x0, min=1e-8)
            ay = (query_y - y0) / torch.clamp(y1 - y0, min=1e-8)
            c000 = fg[left_t, left_x, left_y]
            c001 = fg[left_t, left_x, right_y]
            c010 = fg[left_t, right_x, left_y]
            c011 = fg[left_t, right_x, right_y]
            c100 = fg[right_t, left_x, left_y]
            c101 = fg[right_t, left_x, right_y]
            c110 = fg[right_t, right_x, left_y]
            c111 = fg[right_t, right_x, right_y]
            c00 = c000 * (1.0 - ay) + c001 * ay
            c01 = c010 * (1.0 - ay) + c011 * ay
            c10 = c100 * (1.0 - ay) + c101 * ay
            c11 = c110 * (1.0 - ay) + c111 * ay
            c0 = c00 * (1.0 - ax) + c01 * ax
            c1 = c10 * (1.0 - ax) + c11 * ax
            outputs.append(c0 * (1.0 - at) + c1 * at)
            continue

        x_axis = xg[:, 0, 0].contiguous()
        y_axis = xg[0, :, 1].contiguous()
        query_x = torch.clamp(oq[:, 0], min=float(x_axis[0]), max=float(x_axis[-1]))
        query_y = torch.clamp(oq[:, 1], min=float(y_axis[0]), max=float(y_axis[-1]))
        right_x = torch.searchsorted(x_axis, query_x, right=False)
        right_y = torch.searchsorted(y_axis, query_y, right=False)
        right_x = torch.clamp(right_x, 1, x_axis.numel() - 1)
        right_y = torch.clamp(right_y, 1, y_axis.numel() - 1)
        left_x = right_x - 1
        left_y = right_y - 1
        x0 = x_axis[left_x]
        x1 = x_axis[right_x]
        y0 = y_axis[left_y]
        y1 = y_axis[right_y]
        tx = (query_x - x0) / torch.clamp(x1 - x0, min=1e-8)
        ty = (query_y - y0) / torch.clamp(y1 - y0, min=1e-8)
        q00 = fg[left_x, left_y]
        q01 = fg[left_x, right_y]
        q10 = fg[right_x, left_y]
        q11 = fg[right_x, right_y]
        outputs.append(
            (1.0 - tx) * (1.0 - ty) * q00
            + (1.0 - tx) * ty * q01
            + tx * (1.0 - ty) * q10
            + tx * ty * q11
        )
    return torch.stack(outputs, dim=0)
