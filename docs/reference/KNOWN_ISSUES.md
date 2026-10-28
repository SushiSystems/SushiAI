# Known issues

Next id: KI-18

## Library behaviour

### KI-3 `gradcheck` does not check the fp16 path of a mixed-precision graph
- Origin: own
- Where: `modules/core/autograd/include/SushiAI/autograd/gradcheck.hpp`
- Cause: it takes its point in FLOAT64
- Recognise: a wrong fp16 rule passes the check
- Workaround: check the rule in FLOAT64
- Fix: none, a finite-difference check needs FLOAT64 precision

### KI-4 NVIDIA Pascal GPUs fail when the OpenCL loader ingests SPIR-V
- Origin: toolchain
- Cause: the oneAPI OpenCL loader on compute capability 6.1, such as a GTX 1060
- Recognise: a kernel launch fails on a Pascal card through OpenCL
- Workaround: send the CLI, the tests and training to the OpenCL CPU runtime with
  `ONEAPI_DEVICE_SELECTOR="opencl:cpu"`


### KI-17 Two runtimes of one process on one CUDA GPU on Windows fault a host copy
- Origin: toolchain
- Where: `modules/core/distribution/source/replica_step.cpp`
- Cause: the CUDA driver on Windows gives no concurrent managed access, so the host may not touch
  `Residency::Shared` memory while any runtime of the process runs a kernel on that GPU; a replica
  copies its parameters and results into its model's shared tensors on the host
- Recognise: exception 0xC0000006 in `memmove` under a replica's host copy, or
  `CUDA_ERROR_UNKNOWN` at a kernel launch, while a second runtime of the process works
- Workaround: give each runtime of one process its own device; the in-process workers of
  `tests/core/distribution/exchange_test_support.hpp` and
  `tests/applications/programs/source/cluster_test_files.hpp` drive the CPU
- Fix: none, a limit of the driver. A cluster master holds its own runtime and replica 0's; its
  exchange orders its kernels before or after the replicas and its packs are pinned host memory,
  but no test runs a master on a CUDA GPU on Windows beside a busy runtime

## Documentation

### KI-10 The demo page's node counts were never checked against a run
- Origin: own
- Where: `docs/guides/DEMO.md`
- Cause: the transcript says "17 IR nodes over 22 values" and the text under it "21 graph nodes"
- Recognise: the two counts disagree
- Workaround: run the demo and read its counts
- Fix: RW-9
