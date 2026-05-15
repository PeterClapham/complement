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

The default config is the 15-epoch MNIST grid: three seeds, MNIST, and the 7x7
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

## Run On Slurm

`configs/slurm.yaml` defines a one-seed 7x7 MNIST grid. The Slurm array script
maps each array task to one beta coordinate, so all 49 models can be scheduled
independently while keeping the same checkpoint/resume behavior.

Check the number of configured tasks:

```bash
python scripts/run_slurm_task.py --config configs/slurm.yaml --print-count
```

Submit the array after adapting `slurm/run_gon_grid.sbatch` to your cluster's
modules and environment setup:

```bash
sbatch slurm/run_gon_grid.sbatch
```

The provided script requests one GPU per task and uses `#SBATCH --array=0-48`
for the 49 beta configurations in `configs/slurm.yaml`. If you change seeds,
datasets, or beta values, update the array bounds to match the printed count.

## Select The Epoch Budget

`scripts/run_epoch_probe.py` runs an exploratory train/validation probe with the
same variational inference procedure used at evaluation time. It logs validation
`ELBO_opt`, reconstruction, KL, and the train/validation gap to
`epoch_metrics.csv`.

```bash
python scripts/run_epoch_probe.py --config configs/epoch_probe.yaml --beta-inf 1 --beta-opt 1
```

Representative MNIST probes selected 15 epochs as the shared grid budget:
`beta=(1, 1)` peaked at epoch 6, `beta=(10, 10)` at epoch 11, and
`beta=(0.01, 0.01)` at epoch 15.

For a reconstruction-focused study that benchmarks batch sizes first and then
runs long probes at the low corner, center, and high corner of the beta grid:

```bash
python scripts/run_reconstruction_study.py --config configs/reconstruction_probe.yaml
```

## Run The Test Grid

`scripts/run_test_grid.py` loads the completed model grid and evaluates:

- standard ELBO with beta fixed to `1`
- latent representation perplexity
- latent representation entropy

It saves a CSV table and one heatmap per metric.

```bash
python scripts/run_test_grid.py --config configs/test_grid.yaml
```

Each completed training run also saves `reconstruction_grid.png` in its run
directory. To regenerate one grid or backfill every completed model in a config:

```bash
python scripts/save_reconstruction_grid.py --config configs/test_grid.yaml --beta-inf 1 --beta-opt 1
python scripts/save_reconstruction_grid.py --config configs/test_grid.yaml --all
```

To save milestone model states and reconstruction grids during training, set
`training.artifact_epochs` in the config, for example:

```yaml
training:
  epochs: 100
  artifact_epochs: [50, 63, 75, 88, 100]
```

Pure data-loading throughput options can also be configured without changing
the model objective or update rule:

```yaml
training:
  num_workers: 4
  pin_memory: true
  persistent_workers: true
```

MNIST can also cache the already-transformed tensors under its configured
`data_dir`:

```yaml
datasets:
  - name: mnist
    cache_tensors: true
```

On the local RTX 3070 workstation, the fastest measured conservative setting
for `batch_size: 128` was `num_workers: 2` with pinned memory, persistent
workers, cached MNIST tensors, and progress bars disabled.

## Layout

- `configs/`: YAML experiment configs.
- `notebooks/`: Colab and analysis notebooks.
- `scripts/`: training and experiment entry points.
- `slurm/`: Slurm batch submission templates.
- `src/data/`: dataset loading and preprocessing.
- `src/metrics/`: metric implementations.
- `src/models/`: model definitions.
- `src/training/`: training loops and optimization helpers.
- `src/utils/`: reusable utilities for config, seeds, and logging.
- `tests/`: pytest coverage for reusable code.
