# Remaining Work

This document tracks all open technical debt, unfinished features, and planned architectural extensions across SushiAI.

## Memory Planning and Allocation

- **Scratch memory arena packing:** `MemoryPlanner` packs graph IR values into activation arenas, but lowering-stage temporaries (`TensorPool::scratch()`) and shared constants (`ones()`) receive individual `Storage` allocations. Unifying these allocations requires an explicit host-lane scratch specification shared between `MemoryPlanner` and `Lowering` (`src/graph/memory_planner.cpp`, `src/graph/lowering.cpp`).
- **Batch feeding without staging copies:** Feeding an input batch issues two `blas().copy()` calls per step because each batch presents new buffer addresses to recorded tasks. Removing these copies requires either runtime address late-binding or datasets that materialize directly into plan-allocated input buffers (`src/train/trainer.cpp`).

## Operator Fusion and IR

- **Row-wise log-softmax kernel fusion:** `LOG_SOFTMAX` lowers to nine SushiBLAS calls. Folding these calls into a single fused operation requires adding an entry to the fusion catalog that uses `SushiBLAS::ReductionOps::max_along` and `sum_along` (`src/graph/fusion.cpp`, `src/graph/lowering.cpp`).
- **Fused NLL loss with transform functor:** `NLL_LOSS` lowers to three calls because SushiBLAS whole-tensor reductions lack a transform functor for elementwise multiplication (`src/graph/lowering.cpp`).
- **Standalone `LOG_SOFTMAX` backward IR node:** `LOG_SOFTMAX` autodiff currently functions only when fused with `NLL_LOSS`. Supporting standalone differentiation requires an explicit IR operation identity, builder method, shape inference rule, and lowering rule for row-wise sum reductions (`include/SushiAI/core/ops.hpp`, `src/autograd/rules/activation.cpp`).

## Training and Precision

- **Whole-step loss scaling skip:** In mixed precision, `LossScaler` evaluates overflow per parameter tensor. PyTorch's `GradScaler` behavior freezes all model parameters when any gradient overflows. Adding global step skipping requires expressing step-level dependency branches across `execute_zip_inplace` calls (`include/SushiAI/train/loss_scaler.hpp`).
- **bfloat16 data type support:** `MixedPrecisionPolicy` accepts narrow data types via constructor arguments, but `SushiBLAS::Core::DataType` does not yet define bfloat16 (`include/SushiAI/graph/precision.hpp`).

## Dataset and Objective Generalization

- **Dataset contract generalization beyond classification:** `Data::Dataset::num_classes()` is pure virtual and `Data::Batch` assumes classification targets. Non-classification workloads (regression, object detection, physics-informed neural networks) require decoupling dataset interfaces from label-based classification (`include/SushiAI/data/dataset.hpp`).
- **Arbitrary objective leaves in Trainer:** `Train::Trainer` explicitly feeds `StepSpec::targets`. Objectives with arbitrary input leaves require general leaf binding in `Trainer::step` (`src/train/trainer.cpp`, `include/SushiAI/train/objective.hpp`).

## Diagnostics and Verification

- **Reference loss curve tracking:** `sa demo mlp` verifies compilation count and steady-state allocations, but does not pin regression loss trajectories against golden runs (`cli/sushiai/commands/demo.py`).
- **Per-node profiling output:** Emitting detailed per-node execution timings requires plumbing `SushiRuntime::RuntimeSettings::profiling` into CLI demo runners (`cli/sushiai/commands/demo.py`).
