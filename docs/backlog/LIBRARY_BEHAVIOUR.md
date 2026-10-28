# Library behaviour

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
| KI-3 | own | `gradcheck` does not check the fp16 path of a mixed-precision graph | check the rule in FLOAT64 | none, a finite-difference check needs FLOAT64 precision |  | 2026-10-19 |
| KI-4 | toolchain | NVIDIA Pascal GPUs fail when the OpenCL loader ingests SPIR-V | send the CLI, the tests and training to the OpenCL CPU runtime with `ONEAPI_DEVICE_SELECTOR="opencl:cpu"` |  |  | 2026-10-19 |
| KI-17 | toolchain | Two runtimes of one process on one CUDA GPU on Windows fault a host copy | give each runtime of one process its own device; the in-process workers of `tests/core/distribution/exchange_test_support.hpp` and… | none, a limit of the driver. A cluster master holds its own runtime and replica 0's; its exchange orders its kernels before or after the replicas and its… |  | 2026-10-25 |
