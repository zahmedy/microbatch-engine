# Contributing

Thanks for checking out microbatch-engine! This project exists to make micro-batch training transparent and reproducible. Here is how to get set up and contribute without surprises.

## Environment
- Use Python 3.10+.
- Create a virtual environment and install dependencies:
  ```bash
  pip install -e .
  pip install -r requirements-dev.txt
  ```
- Datasets (Fashion-MNIST) are downloaded automatically to `data/`.

## Development workflow
- Keep changes focused and explain the shape/gradient impact in commit messages when relevant.
- Run tests before sending changes:
  ```bash
  pytest
  ```
- Format and lint (optional but encouraged):
  ```bash
  black .
  ruff .
  ```
- Avoid changing public APIs without updating docs.

## Testing notes
- Gradient equivalence tests use seeded runs; please avoid introducing non-deterministic layers without seeding.
- If you add new tests, prefer shape assertions and explicit tolerance thresholds for numeric checks.

## Filing issues
- Include PyTorch/torchvision versions, OS, and device (`cpu`, `cuda`, `mps`).
- Share minimal repro snippets and expected vs actual shapes or metrics.
