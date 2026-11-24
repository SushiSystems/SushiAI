# tests/layers/train/test_trainer

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_compile_once.cpp` | Tests that a training step compiles once and that every replay of it dispatches work. |
| `test_convergence.cpp` | Tests that an MLP converges on the synthetic dataset and that its updates follow their rules. |
| `test_dense_objectives.cpp` | Tests the binary, multi-label and regression objectives through Train::Trainer. |
| `test_determinism.cpp` | Tests that training, initialisation and a replayed step repeat bit for bit. |
| `test_dropout.cpp` | Checks DROPOUT on an engine: the mask and its output, the seeds, the loss over the masked value, and a model with dropout trained through the Trainer. |
| `test_fused_optimizer.cpp` | Tests that the fused optimizer update matches the separate operations byte for byte. |
| `test_fusion_numerics.cpp` | Tests that the fused graph trains byte for byte like the unfused graph it replaces. |
| `test_learning_curve.cpp` | Tests that the per-step loss trajectory improves from one epoch to the next. |
| `test_loss_scaling.cpp` | Tests that loss scaling detects an fp16 overflow, skips the step and never recompiles. |
| `test_metric_values.cpp` | Tests the values an objective publishes to its metrics through the step builders. |
| `test_mixed_precision.cpp` | Tests mixed precision: execution, tensor dtypes, the policy check and the float32 master weights. |
| `test_resilience.cpp` | Tests loud failures, boundary extents, numerical extremes and resource discipline. |
| `test_static_memory.cpp` | Tests that the memory planner packs buffers without allocating or changing a number. |
| `test_unwritten_leaf.cpp` | Tests that Trainer::run_step refuses an objective whose feed leaves a leaf unwritten. |
| `test_zones.cpp` | Tests that Trainer::fit writes one period per step, holding the zones of its stages. |
