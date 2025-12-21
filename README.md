# microbatch-engine

A minimal PyTorch project to master **shapes, batch dimensions, and training loops** by building a
micro-batch training engine from scratch.

```microbatch-engine/
  README.md
  pyproject.toml
  .gitignore
  src/
    microbatch_engine/
      __init__.py
      config.py
      engine.py
      models.py
      data.py
      utils.py
  tests/
    test_engine_shapes.py
    test_grad_accum_equivalence.py
  scripts/
    train_mnist.py
```

## Why this repo exists
I want to stop “hoping” my tensors align and start **knowing**:
- how `(B, ...)` gets split into micro-batches
- how loss reduction affects gradient accumulation
- how AMP changes dtypes without changing shapes
- how to debug shape + broadcasting bugs fast

## Shape contrast 
### Input (B, 1, 28, 28) (Batch, Channel , Hieght, Width)
### After Conv1/ReLU/Pool (B, 8, 14, 14)
### After Conv2/ReLU/Pool (B, 16, 7, 7)
### After flatten (B, 16*7*7)
### Model output logits (B, 10)
### Labels (B, )

## Batch Format
### (x, y)
- x: (B, 1, 28, 28)
- y: (B, )

## Micro-batch split
### Engine Split along dim=0
### Produce N micro-batches:
- x_i: (B/N, 1, 28, 28)
- y_i: (B/N, )

## Engine Contract
- Batch: (x, y)
- Engine only splits on dim=0 (batch axis)
- Call zero_grad() once before micro-batch loop
- For each micro-batch: forward → loss → scale loss → backward
- After loop: optimizer.step()
- Return metrics: full-batch loss (and optionally grad norm)

## Features (planned)
### Phase 1 — Micro-batch engine
- Split a batch into N micro-batches
- Accumulate gradients across micro-batches
- Step optimizer once per full batch
- Optional shape tracing per module

### Phase 2 — AMP (Mixed Precision)
- `autocast`
- `GradScaler`
- Correct handling of overflow / skipped steps

### Phase 3 — “Fake DataParallel” (single machine)
- Model replicas
- Scatter micro-batches to replicas
- Gather grads back to a master model
- Sync weights

## Quickstart (after implementation)
```bash
python -m scripts.train_mnist --microbatches 4 --amp
