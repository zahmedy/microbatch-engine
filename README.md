# microbatch-engine

[![Tests](https://github.com/zahmedy/microbatch-engine/actions/workflows/python-app.yml/badge.svg)](https://github.com/zahmedy/microbatch-engine/actions/workflows/python-app.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


A minimal PyTorch project to master **shapes, batch dimensions, and training loops** by building a micro-batch training engine from scratch.

> Goal: demonstrate clean micro-batch training, gradient accumulation, and reproducibility on Fashion-MNIST without hiding the math or shapes.

## Highlights
- Micro-batch engine that accumulates gradients and steps once per full batch.
- Simple CNN backbone sized for 28x28 grayscale inputs.
- Fashion-MNIST data pipeline with normalization and train/test loaders.
- Training + evaluation script for quick benchmarking on CPU or Apple Silicon (MPS).
- Pytest scaffolding for gradient-accumulation equivalence.

## Project layout
```
microbatch-engine/
├─ README.md
├─ LICENSE
├─ CONTRIBUTING.md
├─ pyproject.toml
├─ scripts/
│  └─ train_mnist.py
├─ src/microbatch_engine/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ engine.py
│  ├─ models.py
│  └─ utils.py
└─ tests/
   ├─ test_engine_shapes.py
   └─ test_grad_accum_equivalence.py
```

## Getting started
1) Create and activate a Python 3.10+ environment.
2) Install the project in editable mode:
   ```bash
   pip install -e .
   ```
3) (Optional) install dev tools:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Train and evaluate
This fetches Fashion-MNIST automatically:
```bash
python scripts/train_mnist.py
```
The script trains for five epochs and reports average train loss per sample and final test accuracy.

### Run tests
```bash
pytest
```

## Test status
- Current environment: tests **not run** here; use `pytest` locally after installing dependencies.
- CI: not configured yet.

## Design notes
- **Micro-batching:** full batch `(B, C, H, W)` is split into `microbatches` chunks along `B`. Each micro-loss is scaled by `len(chunk) / B` before `backward()` so accumulated grads match a single large-batch pass.
- **Shapes-first:** assertions catch unexpected dimensionality early; configs keep kernel sizes, strides, and padding in one place.
- **Reproducibility:** `seed_all` seeds Python, NumPy, and PyTorch (CPU/MPS) to compare full vs micro-batch gradient flows.

## Roadmap (snapshot)
- Phase 1: core micro-batch engine and CNN baseline ✅
- Phase 2: AMP (autocast + GradScaler) ◻︎
- Phase 3: single-machine “fake DataParallel” ◻︎

## License
MIT — see `LICENSE`.
