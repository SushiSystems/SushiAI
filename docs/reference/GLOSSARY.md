# Glossary

This document defines core terminology and abbreviations used across SushiAI and the wider SushiStack ecosystem.

## Core Concepts

### IR (Intermediate Representation)
The declarative, graph-based representation of neural network computation in SushiAI. The IR separates high-level model semantics from low-level execution hardware. Graph structures reside under the `SushiAI::ir` namespace (`ir::Graph`, `ir::Node`, `ir::Value`).

### Node
A discrete operation in the IR graph representing computation (e.g. `MatMulNode`, `ReluNode`, `CrossEntropyLossNode`). Nodes take zero or more input `Value` instances and yield one or more output `Value` instances.

### Value
A typed, shaped Static Single Assignment (SSA) handle flowing between IR nodes. Each `Value` tracks its producing node, data type (`float32`, `float16`), and multidimensional tensor shape.

### CompiledStep
An immutable, lowered execution plan produced by the compilation pipeline (`Compiler::compile`). It packages low-level kernel dispatch sequences, buffer address bindings, and device command queue captures. Repeated training or inference steps dispatch the `CompiledStep` directly without graph traversal overhead.

### Memory Arena / Scratch Arena
A contiguous region of device memory allocated upfront and managed by SushiAI's linear or lifetime-aware allocator. Intermediate activation buffers and gradient workspaces borrow memory from this arena, eliminating dynamic runtime allocations during the training loop.

### Loss Scaler
A numerical stabilization mechanism used during mixed-precision training. Because gradient values in FP16 often underflow to zero, the loss scaler multiplies the forward loss by a scale factor $S$ before backward autograd. After backward propagation, gradients are scaled down by $1/S$ prior to optimizer weight updates.

### Dynamic Loss Scaling
An adaptive loss scaling technique that monitors gradient values for `inf` or `nan` occurrences. If an overflow occurs, the optimizer skips the weight update, reduces the scale factor by a configured backoff factor, and resets the growth counter. If steps complete cleanly for a configured window, the scale factor doubles.

### Epilogue
An operation fused directly onto the output phase of a matrix multiplication or convolution kernel. Common epilogues include bias addition, scaling, and activation functions (e.g. ReLU, GELU). Epilogue fusion eliminates an extra global memory round-trip.

### SACP (Sushi Autograd & Compilation Pipeline)
The multi-pass compiler subsystem that ingests eager model definitions, generates reverse-mode autograd nodes, executes graph optimization passes (constant folding, dead-code elimination, kernel fusion), and lowers the resulting graph to hardware dispatchables.

### D3 Layered Resolution
The dependency lookup strategy enforced across all SushiStack repositories:
1. First, check for an in-tree CMake target (used when built within a monorepo or superproject).
2. Second, search for an installed package via CMake's `find_package()`.
3. Third, search for sibling checkout directories (`../sushiruntime`, `../sushiblas`).

### OpenCL / SPIR-V
The standard cross-platform compute backend and intermediate bytecode format utilized by SushiAI's hardware runtime layer. Compute kernels are ingested as SPIR-V or compiled at runtime via OpenCL drivers across Intel, NVIDIA, and AMD hardware.
