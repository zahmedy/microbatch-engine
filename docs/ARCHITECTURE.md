# Architecture

This project keeps the micro-batch training loop explicit so shapes and gradient scaling stay easy to reason about.

## Modules
- `config.py` — hyperparameters and device selection; keeps kernel sizes, padding, learning rate, batch size, and micro-batch count in one place.
- `data.py` — Fashion-MNIST loaders with normalization; returns train/test `DataLoader` instances.
- `models.py` — `SimpleCNN` sized for 28x28 grayscale inputs; two Conv+ReLU+Pool blocks followed by a small MLP head.
- `engine.py` — `MicroBatchEngine` orchestrates splitting the batch, scaling micro-losses by `len(chunk)/B`, accumulating grads, and stepping the optimizer once.
- `utils.py` — helpers for seeding and (in the future) shared training utilities.
- `scripts/train_mnist.py` — end-to-end training + evaluation example wiring the components together.

## Training flow
1. Fetch a full batch `(B, C, H, W)` from the dataloader.
2. Split into `microbatches` chunks along `B`.
3. For each chunk:
   - Forward pass through `SimpleCNN`.
   - Compute loss.
   - Scale loss by `len(chunk) / B` so accumulated gradients equal the full-batch loss.
   - Backpropagate.
4. After all chunks: optimizer `step()` once, then move to the next batch.

## Shapes (Fashion-MNIST)
- Input: `(B, 1, 28, 28)`
- After Conv1/ReLU/Pool: `(B, 8, 14, 14)`
- After Conv2/ReLU/Pool: `(B, 16, 7, 7)`
- After flatten: `(B, 16*7*7)`
- Logits: `(B, 10)`
- Labels: `(B,)`

## Testing strategy
- `tests/test_grad_accum_equivalence.py` compares gradients from full-batch vs micro-batch passes to ensure the scaling contract holds.
- `tests/test_engine_shapes.py` is reserved for future shape assertions (e.g., confirming splits and outputs).
