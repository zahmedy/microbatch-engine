# Roadmap

The project progresses in three phases to keep micro-batch mechanics clear and testable.

## Phase 1 — Micro-batch engine (current)
- Split a full batch into `N` micro-batches along the batch axis.
- Scale each micro-loss by `len(chunk) / B` to keep gradients equivalent to full-batch training.
- Accumulate gradients across micro-batches and step once per full batch.
- Add assertions around shapes to catch silent broadcasting.

## Phase 2 — AMP (Mixed Precision)
- Add `autocast` coverage for forward passes.
- Integrate `GradScaler` to prevent underflow.
- Validate gradient equivalence vs FP32 with tolerances.

## Phase 3 — Single-machine “Fake DataParallel”
- Replicate models per device and scatter micro-batches.
- Accumulate grads on a primary replica and synchronize weights.
- Provide profiling hooks to measure per-replica throughput.

## Stretch ideas
- Optional shape tracing per module.
- CLI for experiment configs (microbatches, AMP, devices).
- TensorBoard/CSV logging for losses and grad norms.
