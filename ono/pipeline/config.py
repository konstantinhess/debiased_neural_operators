from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


@dataclass(frozen=True)
class RepeatedRunConfig:
    num_repeats: int
    seed_offset: int = 0
    output_root: str = "results/coverage"
    study_name: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RepeatedRunConfig":
        return cls(
            num_repeats=int(payload.get("num_repeats", 3)),
            seed_offset=int(payload.get("seed_offset", 0)),
            output_root=str(payload.get("output_root", "results/coverage")),
            study_name=(
                str(payload["study_name"])
                if payload.get("study_name") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class NuisanceStudyConfig:
    solution_bias_levels: list[float]
    beta_bias_levels: list[float]
    joint_bias_levels: list[float]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NuisanceStudyConfig":
        return cls(
            solution_bias_levels=[float(value) for value in payload.get("solution_bias_levels", [])],
            beta_bias_levels=[float(value) for value in payload.get("beta_bias_levels", [])],
            joint_bias_levels=[float(value) for value in payload.get("joint_bias_levels", [])],
        )


@dataclass(frozen=True)
class PPIStudyConfig:
    unlabeled_counts: list[int]
    truth_pool_size: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PPIStudyConfig":
        return cls(
            unlabeled_counts=[int(value) for value in payload.get("unlabeled_counts", [])],
            truth_pool_size=(
                int(payload["truth_pool_size"])
                if payload.get("truth_pool_size") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class Phase3Config:
    dataset_config_path: str | None
    dataset_root: str
    split: str
    model_config_path: str
    output_root: str
    run_name: str
    functional: str
    functional_params: dict[str, Any]
    num_folds: int
    batch_size: int
    seed: int
    trainer_max_epochs: int | None
    ablation_variants: list[str]
    repeated_runs: RepeatedRunConfig | None
    nuisance_study: NuisanceStudyConfig | None
    ppi_study: PPIStudyConfig | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Phase3Config":
        return cls(
            dataset_config_path=(
                str(payload["dataset_config_path"])
                if payload.get("dataset_config_path") is not None
                else None
            ),
            dataset_root=str(payload["dataset_root"]),
            split=str(payload.get("split", "train")),
            model_config_path=str(payload["model_config_path"]),
            output_root=str(payload.get("output_root", "results")),
            run_name=str(payload["run_name"]),
            functional=str(payload["functional"]),
            functional_params=dict(payload.get("functional_params", {})),
            num_folds=int(payload.get("num_folds", 2)),
            batch_size=int(payload.get("batch_size", 4)),
            seed=int(payload.get("seed", 1234)),
            trainer_max_epochs=(
                int(payload["trainer"].get("max_epochs"))
                if "trainer" in payload and payload["trainer"].get("max_epochs") is not None
                else None
            ),
            ablation_variants=list(payload.get("ablation_variants", [])),
            repeated_runs=(
                RepeatedRunConfig.from_dict(payload["repeated_runs"])
                if payload.get("repeated_runs") is not None
                else None
            ),
            nuisance_study=(
                NuisanceStudyConfig.from_dict(payload["nuisance_study"])
                if payload.get("nuisance_study") is not None
                else None
            ),
            ppi_study=(
                PPIStudyConfig.from_dict(payload["ppi_study"])
                if payload.get("ppi_study") is not None
                else None
            ),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "Phase3Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
