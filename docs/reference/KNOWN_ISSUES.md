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

### KI-9 The agent files name an output folder and a command that are not so
- Origin: own
- Where: `CLAUDE.md`, `AGENTS.md`, `tests/CMakeLists.txt`
- Cause: `CLAUDE.md` names `docs/api-site/` where the Doxyfile writes `build/docs/api-site`, both
  files say `sa doxygen` for `sa docs`, and a test comment cites `docs/README.md` for the demo
  output now in `docs/guides/DEMO.md`
- Recognise: the named folder is empty; `sa doxygen` runs as an old word until 2027-03-25
- Workaround: use `sa docs` and read `build/docs/api-site`
- Fix: RW-124

### KI-10 The demo page's node counts were never checked against a run
- Origin: own
- Where: `docs/guides/DEMO.md`
- Cause: the transcript says "17 IR nodes over 22 values" and the text under it "21 graph nodes"
- Recognise: the two counts disagree
- Workaround: run the demo and read its counts
- Fix: RW-9

### KI-11 The security page links sibling policies at old paths
- Origin: own
- Where: `docs/reference/SECURITY.md`
- Cause: the links were written before the policies moved under `.github/`
- Recognise: the SushiBLAS and SushiRuntime links resolve nowhere
- Workaround: read each sibling's `.github/SECURITY.md`
- Fix: RW-125

### KI-12 No changelog entry records the `sushicore` floor moving from 0.4.0 to 0.7.0
- Origin: own
- Where: `docs/reference/CHANGELOG.md`
- Cause: the entry was never written
- Recognise: `cli/pyproject.toml` history has the raise; the changelog does not
- Workaround: read `git log`
- Fix: RW-126

## Build, CI and root files

### KI-13 A pre-merge build checks out SushiRuntime and SushiBLAS at `main`
- Origin: own
- Where: `.github/workflows/ci.yml`, `.github/workflows/nightly.yml`
- Cause: SushiBLAS has no tag, and SushiRuntime's latest tag predates the runtime SushiBLAS
  `main` configures against
- Recognise: `SUSHIRUNTIME_REF` and `SUSHIBLAS_REF` are `main`
- Workaround: none
- Fix: RW-45

### KI-14 The documentation states version v1
- Origin: own
- Where: `.config/doxygen/Doxyfile`
- Cause: `PROJECT_NUMBER` is "v1 (unreleased)" while the library and the CLI are 0.1.0
- Recognise: the API site names v1
- Workaround: read the version from `CMakeLists.txt`
- Fix: RW-127

### KI-15 Three ignore rules name a folder nothing writes
- Origin: own
- Where: `.gitignore`
- Cause: the rules of `docs/api-site/` outlived the output's move to `build/`
- Recognise: lines 23 to 25 of `.gitignore`
- Workaround: none needed
- Fix: RW-128

### KI-16 The manifest spells the repository URL in lower case
- Origin: own
- Where: `sushi-module.toml`
- Cause: the URL was written by hand
- Recognise: it differs from `SushiSystems/SushiAI`
- Workaround: none needed on a case-blind host
- Fix: RW-129
