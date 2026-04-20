# Orthogonal Neural Operator Experiments

This README is the practical entry point for reproducing the results of the paper "Debiased neural operators for estimating functional".

## Repo layout

- `configs/datasets/`
  - dataset-generation configs
- `configs/pipeline/`
  - experiment configs passed to `ono.pipeline.run_coverage`
- `configs/models/`
  - model backbones and training defaults
- `generated/datasets/`
  - generated benchmark datasets
- `results/coverage/`
  - repeated-run outputs and `summary.json` files
- `results/reports/`
  - report CSVs built from saved coverage outputs
- `scripts/`
  - report builders


## MLflow

Install MLflow support:

```powershell
pip install -r requirements-mlflow.txt
```

Start the local tracking server from the repo root:

```powershell
python -m mlflow server --host 127.0.0.1 --port 5555 --backend-store-uri sqlite:///mlflow/tracking.db --artifacts-destination ./mlflow/artifacts --serve-artifacts
```

Set the tracking URI in the shell where you run experiments:

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5555"
```

Optional:

```powershell
$env:ONO_ENABLE_MLFLOW="0"
```

MLflow state is stored locally in:

- `mlflow/tracking.db`
- `mlflow/artifacts/`

## Reproduce pharmacokinetics data results

Generate the PK datasets once:

```powershell
Get-ChildItem configs/datasets/pharmacokinetics_1d_pk_r*.json | Sort-Object Name | ForEach-Object {
  python -m ono.data.generate --config $_.FullName
}
```

### Pharmacokinetics data main results

Run all non-DeepONet PK main configs currently present in the repo:

```powershell
Get-ChildItem configs/pipeline/pk_main_*_repeats50.json |
  Where-Object { $_.Name -notlike 'pk_main_deeponet_*' } |
  Sort-Object Name |
  ForEach-Object {
    python -m ono.pipeline.run_coverage --config $_.FullName
  }
```

### Pharmacokinetics data nuisance results

Run:

```powershell
python -m ono.pipeline.run_coverage --config configs/pipeline/pk_nuisance_auc_r500_repeats50.json
python -m ono.pipeline.run_coverage --config configs/pipeline/pk_nuisance_smooth_tat_r500_repeats50.json
```

### Pharmacokinetics data PPI results

Run:

```powershell
python -m ono.pipeline.run_coverage --config configs/pipeline/pk_ppi_auc_r500_repeats50.json
python -m ono.pipeline.run_coverage --config configs/pipeline/pk_ppi_soft_cmax_r500_repeats50.json
```

### Build PK report CSVs

After the PK runs above are complete, build the PK report tables:

```powershell
python scripts/pk_reports.py
```

Outputs:

- PK main:
  - `results/reports/pk_main/pk_main_long.csv`
  - `results/reports/pk_main/pk_main_rmse_wide.csv`
  - `results/reports/pk_main/pk_main_bias_wide.csv`
- PK nuisance:
  - `results/reports/pk_nuisance_r500/pk_nuisance_solution_bias.csv`
  - `results/reports/pk_nuisance_r500/pk_nuisance_beta_bias.csv`
  - `results/reports/pk_nuisance_r500/pk_nuisance_joint_bias.csv`
- PK PPI:
  - `results/reports/pk_ppi_r500/pk_ppi_r500_long.csv`

Coverage summaries for all PK runs are written to `results/coverage/<study_name>/summary.json`.

## Reproduce the Darcy results

Generate the Darcy dataset:

```powershell
python -m ono.data.generate --config configs/datasets/pde_twobench_darcy_2d_inverseenergy_r1000.json
```

Run the 50-repeat kappa grid:

```powershell
Get-ChildItem configs/pipeline/pde_twobench_darcy_2d_smooth_excess_above_threshold_inverseenergy_r1000_k*_repeats50.json |
  Sort-Object Name |
  ForEach-Object {
    python -m ono.pipeline.run_coverage --config $_.FullName
  }
```

Build the Darcy report CSV:

```powershell
python scripts/pde_darcy_smooth_excess_kappa_report.py
```

Outputs:

- `results/reports/pde_darcy_smooth_excess_kappa/pde_darcy_smooth_excess_inverseenergy_kappa_repeats50.csv`
- `results/reports/pde_darcy_smooth_excess_kappa/pde_darcy_smooth_excess_inverseenergy_kappa_manifest.json`

Coverage summaries are written to `results/coverage/pde_twobench_darcy_2d_smooth_excess_above_threshold_inverseenergy_r1000_<kappa>_repeats50/summary.json`.
