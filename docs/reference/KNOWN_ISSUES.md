# Known issues

Next id: KI-17

## Library behaviour

### KI-2 Differentiating `K_LOG_SOFTMAX` on its own throws
- Origin: own
- Where: `modules/core/autograd/source/rules/loss.cpp`
- Cause: it differentiates only as the fused `K_LOG_SOFTMAX` and `K_NLL_LOSS` pair
- Recognise: an error from the differentiator on a graph that uses log-softmax without the loss
- Workaround: pair it with `K_NLL_LOSS`
- Fix: RW-123

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

### KI-5 Content errors throw where the failure contract asks for a status
- Origin: own
- Where: `modules/core/failure/include/SushiAI/failure/error.hpp`
- Cause: 398 sites in 66 files throw `Error` through `SA_THROW_IF`, and `try` and `catch` appear
  outside `seam/seam.hpp`
- Recognise: an exception reaches a caller that expected a `Status` or a `Result`
- Workaround: call through `seam/seam.hpp`
- Fix: RW-11

## Documentation

### KI-10 The demo page's node counts were never checked against a run
- Origin: own
- Where: `docs/guides/DEMO.md`
- Cause: the transcript says "17 IR nodes over 22 values" and the text under it "21 graph nodes"
- Recognise: the two counts disagree
- Workaround: run the demo and read its counts
- Fix: RW-9
