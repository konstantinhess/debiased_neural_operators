# Debiased neural operators: paper reproduction code

This repository reproduces the experiments for debiased neural
operators. It contains the paper experiments, their data-generating
processes, and lightweight local JSON/CSV reporting.

## Experiments

- PK main study with the FNO backbone: AUC and smooth TAT; plug-in, DOPE,
  structured w_g, and oracle beta_0.
- PK PPI study: AUC and soft Cmax over the unlabeled-size grid.
- Oracle-centered smooth-TAT perturbation study and fitted log-log slopes.
- PK DeepONet backbone ablation: AUC and smooth TAT.
- AUC confidence-interval coverage and mean interval-length table.
- Darcy-flow smooth-excess experiment over kappa values.
- ETTh1 semi-synthetic experiment: mean future OT and soft Cmax.

All default configs are the full paper configurations. Command-line overrides
are intended only for smoke testing.

## Setup

Python 3.10 or newer is required. From this repository root:

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

On macOS or Linux, activate with:

    source .venv/bin/activate

Installing the local package is optional because the launch scripts locate it
relative to themselves. For an editable installation:

    python -m pip install -e .

The experiments are CPU-compatible, but the complete grids contain many
independent fits and can require many hours on a desktop CPU.

## External data

ETTh1 is the only external dataset. Download the unmodified ETTh1 CSV and put it
at:

    data/ETTh1.csv

The expected checksum and schema are documented in data/README.md. The ETTh1
runner gives an actionable error when the file is absent.

## Full reproduction commands

Run these commands from the repository root.

Main PK FNO tables:

    python scripts/run_pk_main.py

PPI results:

    python scripts/run_pk_ppi.py

Oracle-centered smooth-TAT data, slopes, and figures:

    python scripts/run_pk_oracle_perturbation.py

DeepONet ablation:

    python scripts/run_pk_deeponet.py

Confidence-interval table, after the main AUC study has completed:

    python scripts/run_pk_clt.py

Darcy-flow table:

    python scripts/run_darcy.py

ETTh1 AUC and soft-Cmax results:

    python scripts/run_etth1.py

Each launcher also accepts a functional or condition subset. Use --help for the
complete interface.

## Smoke tests

These commands exercise every computational path with intentionally tiny
settings. These results are not paper results.

FNO main:

    python scripts/run_pk_main.py --functional auc --rho 0.5 --repeats 1 --epochs 1 --train-size 8 --val-size 4 --test-size 8 --truth-size 16 --output-root outputs/smoke

PPI:

    python scripts/run_pk_ppi.py --functional auc --n2 0,8 --repeats 1 --epochs 1 --train-size 8 --val-size 4 --test-size 8 --truth-size 16 --output-root outputs/smoke

DeepONet:

    python scripts/run_pk_deeponet.py --functional auc --rho 0.5 --repeats 1 --epochs 1 --train-size 8 --val-size 4 --test-size 8 --truth-size 16 --output-root outputs/smoke

Oracle perturbation:

    python scripts/run_pk_oracle_perturbation.py --repeats 3 --test-size 16 --truth-size 64 --output-root outputs/smoke

Darcy:

    python scripts/run_darcy.py --kappa 0 --repeats 1 --epochs 1 --train-size 4 --val-size 2 --test-size 4 --truth-size 8 --output-root outputs/smoke

ETTh1, after placing the dataset:

    python scripts/run_etth1.py --functional auc --rho 0 --repeats 1 --epochs 1 --output-root outputs/smoke

Run the lightweight protocol tests with:

    python -m unittest discover -s tests -v

## Outputs

Outputs are written under outputs/<experiment-name>/.

Synthetic studies write:

- resolved_config.json;
- one JSON file per repeat;
- raw_repeats.json and raw_repeats.csv;
- summary.json;
- table.csv;
- generated synthetic arrays under the experiment's private _generated folder.

The oracle-centered experiment additionally writes summary.csv,
local_slopes.csv, and publication PNG/PDF figures.

ETTh1 writes one directory per rho, per-repeat sparse-design files,
raw_repeats.csv, summary.json, and a combined table.csv.

The output directory is ignored by Git except for outputs/.gitkeep.

## Protocols

### PK

The PK design uses 128 grid points on [0,24], K=24 observations sampled
with replacement, and a fixed early-time window centered at t=1 with half-width
1. The mixture weight is gamma(rho)=0.45 sqrt(rho). The main and DeepONet
studies use:

    rho = 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1
    repeats = 50

The independently generated population reference contains 2,000 subjects per
repeat. The oracle-centered study uses rho=0.5, 256 evaluation subjects,
20,000 reference subjects, and perturbations:

    delta = 0, 0.025, 0.05, 0.1, 0.2, 0.3

### Darcy

The Darcy experiment uses a 17 by 17 field, K=24 observations sampled
with replacement from the inverse-energy design, a 256/64/256 split, 50
repeats, and:

    kappa = 0, 0.2, 0.4, 0.6, 0.8, 1

### ETTh1

The ETTh1 experiment uses the previous 96 hourly observations of all seven
channels to predict the next 24 OT values, stride 24, a chronological
512/64/128 split, and training-only standardization. It samples K=8 future
locations without replacement under the early-block design at:

    rho = 0, 0.25, 0.5, 0.75, 1

## Leakage safeguards

The solution and Riesz operators receive only training covariates and sparse
training observations. Validation selects the best state within the fixed
epoch budget. Held-out test observations are used only after both nuisance
models are frozen. A dedicated training-dataset view removes full simulated
solutions, inverse-design quantities, and design probabilities before fitting.

For ETTh1, complete future trajectories are retained outside the PyTorch
training datasets and are used only to calculate the fixed evaluation target.
For PPI, unlabeled inputs contribute only plug-in predictions; the correction
term is computed only from labeled test subjects.

## Configuration

All paper choices are visible in the standalone JSON files under configs/.
They are fully resolved and contain no inherited search, smoke, or historical
configuration.
