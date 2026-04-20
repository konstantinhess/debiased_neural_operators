from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

from ono.models.deeponet1d import DeepONet1dConfig
from ono.models.fno1d import FNO1dConfig
from ono.models.fno2d import FNO2dConfig
from ono.models.fno3d import FNO3dConfig
from ono.models.nuisance import NuisanceModuleConfig


@dataclass(frozen=True)
class NuisanceModelBundleConfig:
    model: FNO1dConfig | FNO2dConfig | FNO3dConfig | DeepONet1dConfig
    solution_module: NuisanceModuleConfig
    debiasing_module: NuisanceModuleConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NuisanceModelBundleConfig":
        model_payload = payload["model"]
        family = model_payload.get("family")
        if family == "fno1d":
            model = FNO1dConfig(
                in_channels=int(model_payload["in_channels"]),
                hidden_channels=int(model_payload["hidden_channels"]),
                out_channels=int(model_payload["out_channels"]),
                num_layers=int(model_payload["num_layers"]),
                num_modes=int(model_payload["num_modes"]),
            )
        elif family == "deeponet1d":
            model = DeepONet1dConfig(
                in_channels=int(model_payload["in_channels"]),
                branch_hidden_channels=int(model_payload["branch_hidden_channels"]),
                trunk_hidden_channels=int(model_payload["trunk_hidden_channels"]),
                latent_dim=int(model_payload["latent_dim"]),
                branch_layers=int(model_payload["branch_layers"]),
                trunk_layers=int(model_payload["trunk_layers"]),
                out_channels=int(model_payload["out_channels"]),
            )
        elif family == "fno2d":
            model = FNO2dConfig(
                in_channels=int(model_payload["in_channels"]),
                hidden_channels=int(model_payload["hidden_channels"]),
                out_channels=int(model_payload["out_channels"]),
                num_layers=int(model_payload["num_layers"]),
                num_modes_x=int(model_payload["num_modes_x"]),
                num_modes_y=int(model_payload["num_modes_y"]),
            )
        elif family == "fno3d":
            model = FNO3dConfig(
                in_channels=int(model_payload["in_channels"]),
                hidden_channels=int(model_payload["hidden_channels"]),
                out_channels=int(model_payload["out_channels"]),
                num_layers=int(model_payload["num_layers"]),
                num_modes_t=int(model_payload["num_modes_t"]),
                num_modes_x=int(model_payload["num_modes_x"]),
                num_modes_y=int(model_payload["num_modes_y"]),
            )
        else:
            raise ValueError(f"unsupported model family: {model_payload.get('family')}")
        solution = NuisanceModuleConfig(
            learning_rate=float(payload["solution_module"]["learning_rate"]),
            weight_decay=float(payload["solution_module"]["weight_decay"]),
            max_epochs=int(payload["solution_module"]["max_epochs"]),
            riesz_penalty_lambda=0.0,
        )
        debiasing = NuisanceModuleConfig(
            learning_rate=float(payload["debiasing_module"]["learning_rate"]),
            weight_decay=float(payload["debiasing_module"]["weight_decay"]),
            max_epochs=int(payload["debiasing_module"]["max_epochs"]),
            riesz_penalty_lambda=float(payload["debiasing_module"]["riesz_penalty_lambda"]),
        )
        return cls(model=model, solution_module=solution, debiasing_module=debiasing)

    @classmethod
    def from_path(cls, path: str | Path) -> "NuisanceModelBundleConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
