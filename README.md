# microbatch-engine

[![Tests](https://github.com/zahmedy/microbatch-engine/actions/workflows/python-app.yml/badge.svg)](https://github.com/zahmedy/microbatch-engine/actions/workflows/python-app.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch-based training framework implementing micro-batch gradient accumulation to enable training on memory-constrained hardware. The engine splits large batches into smaller chunks, processes them sequentially, and accumulates gradients to match standard batch training while reducing peak memory usage.

## Features

**Core Engine**
- Gradient accumulation across micro-batches with proper loss scaling
- Memory-efficient training on CPU and Apple Silicon (MPS)
- Gradient clipping and configurable batch subdivision

**Implementation**
- CNN architectures (SimpleCNN for Fashion-MNIST, DeepCNN for high-resolution inputs)
- Fashion-MNIST and synthetic dataset pipelines with normalization
- Comprehensive test suite validating gradient equivalence between full-batch and micro-batch training
- Reproducible training with deterministic seeding across PyTorch, NumPy, and Python's random module

**Development**
- Modular architecture separating data loading, model definitions, training logic, and configuration
- Shape assertions throughout the pipeline to catch dimensionality issues early
- Demo script showcasing OOM prevention with micro-batching

## Architecture

```
microbatch-engine/
├─ scripts/
│  ├─ train_mnist.py       # Fashion-MNIST training script
│  └─ demo_oom.py          # OOM prevention demonstration
├─ src/microbatch_engine/
│  ├─ engine.py            # MicroBatchEngine with gradient accumulation
│  ├─ models.py            # SimpleCNN and DeepCNN architectures
│  ├─ data.py              # Dataset loaders and synthetic data generation
│  ├─ config.py            # Training configuration and hyperparameters
│  └─ utils.py             # Reproducibility utilities
└─ tests/
   ├─ test_engine_shapes.py
   └─ test_grad_accum_equivalence.py
```

## Installation

Create a Python 3.10+ environment and install the package:

```bash
pip install -e .
```

For development with testing dependencies:
```bash
pip install -r requirements-dev.txt
```

## Usage

### Training on Fashion-MNIST

```bash
python scripts/train_mnist.py
```

Downloads Fashion-MNIST automatically, trains for five epochs, and reports training loss and test accuracy.

### Memory Efficiency Demo

```bash
python scripts/demo_oom.py
```

Demonstrates how micro-batching prevents OOM errors on large batches with high-resolution inputs. Configure batch size and micro-batch count in [config.py](src/microbatch_engine/config.py).

### Testing

```bash
pytest
```

Validates gradient equivalence between standard and micro-batch training to ensure mathematical correctness.

## Technical Details

### Gradient Accumulation

The engine splits a batch tensor `(B, C, H, W)` into `n` micro-batches along the batch dimension. Each micro-batch computes:

```python
loss_i = loss_fn(model(x_i), y_i) * (b_i / B)
loss_i.backward()
```

where `b_i` is the micro-batch size and `B` is the full batch size. This scaling ensures accumulated gradients are mathematically equivalent to processing the full batch at once.

### Memory vs Throughput

Micro-batching trades computation time for memory efficiency:

- `microbatches=1`: Processes the full batch in one forward/backward pass (fastest, highest memory)
- `microbatches=8`: Splits into 8 sequential passes (slower, 8x lower peak activation memory)

On memory-constrained devices (Apple Silicon, consumer GPUs), increasing micro-batch count can be the difference between OOM failure and successful training.

### Reproducibility

The `seed_all()` utility deterministically seeds PyTorch (CPU/CUDA/MPS), NumPy, and Python's RNG. This enables exact gradient comparison between full-batch and micro-batch execution, verified in the test suite.

## Future Work

- Automatic mixed precision (AMP) integration with `torch.cuda.amp.autocast` and `GradScaler`
- Multi-device gradient accumulation for data-parallel training
- Integration with distributed training frameworks

## License

MIT License - see [LICENSE](LICENSE) for details.
