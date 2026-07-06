# AGENTS.md

This is a research codebase for a machine learning paper.

## Style
- Use Python 3.11+.
- Use PyTorch for models and training.
- Use 4-space indentation.
- Prefer explicit, readable code over clever abstractions.
- Do not use Unicode symbols in script outputs.
- Keep experiment code reproducible.

## Project rules
- All training entry points go in `scripts/`.
- Core reusable code goes in `src/`.
- Configs go in `configs/` as YAML.
- Never hard-code dataset paths.
- Every experiment should save:
  - config copy
  - random seed
  - git commit hash if available
  - metrics as CSV or JSON
  - model checkpoints only when requested

## Testing
- Add or update tests for reusable functions.
- Run `pytest` after meaningful changes.

## Research constraints
- Prioritise clarity and inspectability.
- Do not silently change mathematical definitions.
- When implementing metrics, include docstrings describing the estimator.

## Resource and memory discipline

When editing or running this project, treat memory growth as a first-class correctness issue.

Before making changes:
- Identify code paths that allocate tensors, arrays, datasets, dataloaders, models, cached outputs, logs, plots, or subprocesses.
- Avoid changes that retain references across epochs/runs unless explicitly intended.
- Be suspicious of append-only lists, global caches, closures, hooks, callbacks, WandB/logging buffers, matplotlib figures, CUDA tensors, and dataloader workers.

When adding training/evaluation loops:
- Do not store full tensors, activations, losses, predictions, or batches across iterations unless needed.
- Store scalars via `.item()` where appropriate.
- Use `detach()`, `cpu()`, and deletion deliberately for tensors that should not keep graphs alive.
- Wrap evaluation/inference in `torch.no_grad()` or `torch.inference_mode()`.
- Close matplotlib figures with `plt.close(fig)`.
- Clear or reuse large buffers between runs.
- Avoid accumulating computation graphs accidentally.

When running experiments:
- Prefer small smoke tests before full runs.
- Check whether memory increases between repeated runs with identical settings.
- If possible, report CPU RAM and GPU VRAM before and after repeated runs.
- If memory grows monotonically across runs, stop and investigate before continuing.

For PyTorch/CUDA:
- Use `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` when CUDA is available.
- Use `gc.collect()` and `torch.cuda.empty_cache()` only as diagnostics or cleanup, not as a substitute for fixing retained references.
- Look for retained tensors, hooks, dataloader workers, cached batches, and graph references before assuming CUDA cache behaviour is the cause.

When proposing changes:
- Mention any memory-leak risks introduced or removed.
- Prefer simple, bounded-memory implementations.
- Add lightweight memory regression checks for experiment scripts where practical.