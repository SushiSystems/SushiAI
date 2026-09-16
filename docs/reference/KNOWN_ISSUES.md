# Known Issues

This document records measured limitations, open edge cases, and known architectural trade-offs in SushiAI. Active implementation items tracking these issues are listed in [REMAINING_WORK.md](../design/REMAINING_WORK.md).

## Runtime and Numerical Stability

### 1. Per-Parameter Loss Scaling Skip
- **Behavior:** When dynamic loss scaling detects an `inf` or `nan` in a parameter's gradient tensor during mixed-precision training, that specific parameter update is skipped. The optimizer does not currently abort the entire optimizer step across all parameters simultaneously.
- **Impact:** While infrequent, an overflow in one layer can cause parameter updates to become desynchronized across layers during that specific step.
- **Tracking:** See [REMAINING_WORK.md](../design/REMAINING_WORK.md#1-whole-step-loss-scaling-skip).

### 2. Unpacked Scratch Memory Arena
- **Behavior:** The memory allocator assigns distinct device memory slots to intermediate activations rather than calculating lifetime overlaps to reuse memory addresses.
- **Impact:** Device memory consumption during training scales linearly with the number of intermediate nodes rather than the maximum live cut size, limiting the maximum batch size on memory-constrained GPUs.
- **Tracking:** See [REMAINING_WORK.md](../design/REMAINING_WORK.md#2-scratch-memory-arena-packing).

## Kernel Optimization

### 3. Separate Kernels for Row-Wise Log-Softmax
- **Behavior:** Softmax normalization and logarithm operations currently execute as sequential global-memory passes rather than a single fused row reduction kernel.
- **Impact:** Memory bandwidth utilization is sub-optimal for classification output heads with large vocabulary or class dimensions.
- **Tracking:** See [REMAINING_WORK.md](../design/REMAINING_WORK.md#3-row-wise-log-softmax-fusion).

### 4. Absence of Standalone Log-Softmax Backward IR Node
- **Behavior:** The autograd compiler decomposes log-softmax backward passes into elemental gradient calculations rather than lowering directly to a dedicated backward kernel.
- **Impact:** Generates redundant intermediate graph nodes and temporary device buffers during cross-entropy backpropagation.
- **Tracking:** See [REMAINING_WORK.md](../design/REMAINING_WORK.md#4-standalone-log-softmax-backward-ir-node).

## Ingestion and Pipeline

### 5. Host-Copy Staging in Dataset Ingestion
- **Behavior:** The standard dataset loader copies batch tensors from host memory to an intermediate buffer before device transfer, rather than using direct pinned memory or asynchronous DMA streams.
- **Impact:** Host memory bandwidth creates a throughput bottleneck when training shallow networks with small input tensors at high step rates.
- **Tracking:** See [REMAINING_WORK.md](../design/REMAINING_WORK.md#6-batch-ingestion-host-copy-overhead).

## Hardware Compatibility

### 6. NVIDIA Pascal (sm_61) SPIR-V Ingestion
- **Behavior:** NVIDIA GPUs based on the Pascal architecture (such as the GeForce GTX 1060, Compute Capability 6.1) fail during OpenCL SPIR-V binary ingestion under oneAPI Level Zero / OpenCL loader.
- **Workaround:** When running on host systems equipped with Pascal hardware, set the environment variable `ONEAPI_DEVICE_SELECTOR=opencl:cpu` to target the OpenCL CPU runtime for CLI operations, tests, and training:
  ```powershell
  $env:ONEAPI_DEVICE_SELECTOR="opencl:cpu"; pytest cli/tests -q
  ```
