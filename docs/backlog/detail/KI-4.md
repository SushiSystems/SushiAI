# KI-4 NVIDIA Pascal GPUs fail when the OpenCL loader ingests SPIR-V

The row of this entry is in `docs/backlog/LIBRARY_BEHAVIOUR.md`. This file holds what did not fit the row.

- Origin: toolchain
- Cause: the oneAPI OpenCL loader on compute capability 6.1, such as a GTX 1060
- Recognise: a kernel launch fails on a Pascal card through OpenCL
- Workaround: send the CLI, the tests and training to the OpenCL CPU runtime with `ONEAPI_DEVICE_SELECTOR="opencl:cpu"`
