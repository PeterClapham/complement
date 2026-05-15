# Entropy Paper Code

Minimal research scaffold for a machine learning paper.

## Setup

Create and activate a virtual environment, then install the project in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Run Tests

```powershell
pytest
```

## Run A Single Training Job

```powershell
python scripts/train.py --config configs/smoke.yaml
```

The smoke config runs a tiny two-epoch synthetic dataset check with the
Variational GON model. It loads YAML config, sets random seeds, creates a run
directory under `results/`, writes `metrics.jsonl`, checkpoints each epoch, and
saves `model.pt`.

To inspect the beta phase-diagram grid:

```powershell
python scripts/train.py --config configs/default.yaml --print-grid
```

To run one explicit beta configuration:

```powershell
python scripts/train.py --config configs/default.yaml --dataset mnist --seed 0 --beta-inf 0.01 --beta-opt 10
```

For a quick five-epoch MNIST speed check on a small real-MNIST subset:

```powershell
python scripts/run_grid.py --config configs/mnist_speed.yaml
```

## Run The Grid

The default config is the 300-epoch MNIST grid: three seeds, MNIST, and the 7x7
`beta_inf`/`beta_opt` phase diagram. Each coordinate has a deterministic run
directory and a `checkpoint.pt`. If training is interrupted, rerun the same
command and completed or partially completed coordinates will resume from their
saved state.

```powershell
python scripts/run_grid.py --config configs/default.yaml
```

## Run On Colab

Open `notebooks/colab_training.ipynb` in Google Colab. The notebook mounts
Google Drive, clones or updates this repo, writes Drive-backed Colab configs,
runs a smoke test, and can launch the full resumable grid when
`RUN_FULL_GRID = True`.

## Layout

- `configs/`: YAML experiment configs.
- `notebooks/`: Colab and analysis notebooks.
- `scripts/`: training and experiment entry points.
- `src/data/`: dataset loading and preprocessing.
- `src/metrics/`: metric implementations.
- `src/models/`: model definitions.
- `src/training/`: training loops and optimization helpers.
- `src/utils/`: reusable utilities for config, seeds, and logging.
- `tests/`: pytest coverage for reusable code.
