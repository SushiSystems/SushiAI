# Known issues

Next id: KI-17

## Library behaviour

### KI-1 An overflowed gradient in mixed precision skips one parameter's update only
- Origin: own
- Where: `modules/core/optim/include/SushiAI/optim/loss_scaler.hpp`
- Cause: `LossScaler` screens each parameter's gradient and skips that parameter alone
- Recognise: one step updates some layers and not others
- Workaround: none
- Fix: RW-108

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

## Tests

### KI-6 The step bounds of YOLOv4 and YOLOv7 are wide and unexplained
- Origin: own
- Where: `tests/vision/importers/test_darknet_training/test_yolov4.cpp`,
  `tests/vision/importers/test_darknet_training/test_yolov7.cpp`
- Cause: unknown; no layer before the heads is compared
- Recognise: a head is held to 0.31 of the dump in YOLOv4 and 0.0079 in YOLOv7, 1.5 times the one
  measured difference
- Workaround: none
- Fix: RW-101

### KI-7 The one-batch overfit of YOLOv7 fails
- Origin: own
- Where: `tests/vision/importers/test_darknet_training/test_yolov7.cpp`
- Cause: the run is unstable on its batch, and the order of one sum decides the outcome
- Recognise: `Large_DarknetYolov7Training.TheCostOfOneBatchFallsUnderATenth` reads 11.1 at step
  199 against a bound of 3.56
- Workaround: none
- Fix: RW-103

### KI-8 The step bounds of YOLOv3 are wide and unexplained
- Origin: own
- Where: `tests/vision/importers/test_darknet_training/test_yolov3.cpp`
- Cause: unknown; no layer before the heads is compared
- Recognise: a head is held to 0.36 of the dump, 1.5 times the one measured difference
- Workaround: none
- Fix: RW-104

## Documentation

### KI-10 The demo page's node counts were never checked against a run
- Origin: own
- Where: `docs/guides/DEMO.md`
- Cause: the transcript says "17 IR nodes over 22 values" and the text under it "21 graph nodes"
- Recognise: the two counts disagree
- Workaround: run the demo and read its counts
- Fix: RW-9

## Build, CI and root files

### KI-13 A pre-merge build checks out SushiRuntime and SushiBLAS at `main`
- Origin: own
- Where: `.github/workflows/ci.yml`, `.github/workflows/nightly.yml`
- Cause: SushiBLAS has no tag, and SushiRuntime's latest tag predates the runtime SushiBLAS
  `main` configures against
- Recognise: `SUSHIRUNTIME_REF` and `SUSHIBLAS_REF` are `main`
- Workaround: none
- Fix: RW-45

