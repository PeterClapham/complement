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