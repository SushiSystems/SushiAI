# Known issues

Defects that exist today, each with where it sits. Planned work is in
[`REMAINING_WORK.md`](../design/REMAINING_WORK.md).

## Library behaviour

| Issue | Where |
| --- | --- |
| In mixed precision an overflowed gradient skips that parameter's update only; the other parameters still step, so one step can update some layers and not others | `modules/core/optim/include/SushiAI/optim/loss_scaler.hpp` |
| Differentiating `K_LOG_SOFTMAX` on its own throws; it differentiates only as the fused `K_LOG_SOFTMAX` + `K_NLL_LOSS` pair | `modules/core/autograd/source/rules/loss.cpp` |
| `gradcheck` takes its point in FLOAT64, so it does not check the fp16 path of a mixed-precision graph. Check the rule in FLOAT64 | `modules/core/autograd/include/SushiAI/autograd/gradcheck.hpp` |
| NVIDIA Pascal GPUs (GTX 1060, compute capability 6.1) fail when the OpenCL loader ingests SPIR-V. Workaround below | The oneAPI OpenCL loader, outside this repository |
| Content errors throw `Error` through `SA_THROW_IF`: 398 sites in 66 files of `modules/*/*/include/` and `modules/*/*/source/` (the count of `tools/contract/site_classes.tsv`), where the failure contract asks for a `Status` or a `Result`. Stage 2 of [`PROGRAMME.md`](../design/PROGRAMME.md) converts them | `modules/core/core/include/SushiAI/core/common.hpp` (the macro); `modules/core/graph/source/builder.cpp` holds the most, 95 calls |
| `try` and `catch` appear outside `core/seam.hpp`. `core/status.hpp` and `core/check.hpp` guard their own allocation and message composition; `config/value.cpp` catches the JSON parser's exception; `throwing_bridge.hpp` catches a layer's `Error`; five layer sources catch an `Error` from a shape or window check and throw it again with the layer's name (`conv2d.cpp`, `max_pool2d.cpp`, `avg_pool2d.cpp`, `upsample2d.cpp`, `layer_registry.cpp`) | `modules/core/core/include/SushiAI/core/status.hpp`, `modules/core/core/include/SushiAI/core/check.hpp`, `modules/core/config/source/value.cpp`, `modules/layers/nn/include/SushiAI/nn/throwing_bridge.hpp`, `modules/layers/nn/source/` |

On a Pascal machine, send the CLI, the tests and training to the OpenCL CPU runtime:

```powershell
$env:ONEAPI_DEVICE_SELECTOR="opencl:cpu"; pytest cli/tests -q
```

## Tests

| Issue | Where |
| --- | --- |
| A training step of YOLOv4 and of YOLOv7 at full size returns `OutOfMemory` under the default memory limit of `sa test`, so 12 cases fail; one step of YOLOv4 commits 11.1 GB | `tests/integration/test_darknet_yolov4_training.cpp`, `tests/integration/test_darknet_yolov7_training.cpp` |
| The step bounds of YOLOv3 hold a head to 0.36 of the dump, 1.5 times the one measured difference; no layer before the heads is compared | `tests/integration/test_darknet_yolov3_training.cpp:56-61` |

## Documentation

| Issue | Where |
| --- | --- |
| Names `docs/api-site/` as the Doxygen output; the `.config/doxygen/Doxyfile` writes `build/docs/api-site` | `CLAUDE.md:10` |
| Name the command `sa doxygen`; it is `sa docs`, and the old word runs until 2027-03-25 | `CLAUDE.md:57`, `AGENTS.md:21` |
| The demo transcript says "17 IR nodes over 22 values" and the text under it says "21 graph nodes"; neither was checked against a run | `README.md:278`, `README.md:321` |
| Comments cite `docs/ARCHITECTURE.md`; the file is `docs/architecture/ARCHITECTURE.md` | `CMakeLists.txt:72` |
| A comment cites `docs/README.md` for the demo's quoted output; that text is in the root `README.md` | `tests/CMakeLists.txt:118` |
| The links to the SECURITY pages of SushiBLAS and SushiRuntime were not checked against those repositories' `main` | `docs/reference/SECURITY.md` |
| No changelog entry records the `sushicore` floor moving from 0.4.0 to 0.7.0, and 0.7.0 was not confirmed on PyPI | `cli/pyproject.toml:16` |

## Build, CI and root files

| Issue | Where |
| --- | --- |
| The `cli` job installs the CLI with `pip install -e ./cli`, which needs `sushicore>=0.7.0` from PyPI; that version is not confirmed there (see Documentation) | `.github/workflows/ci.yml:162`, `cli/pyproject.toml:16` |
| The pyflakes step ends in `\|\| true` and cannot fail | `.github/workflows/ci.yml:167` |
| The workflow header cites `docs/CLAUDE.md`, which does not exist; the file is `CLAUDE.md` at the root | `.github/workflows/ci.yml:26` |
| The `checkers` job does not run `tools/contract/check_contract.py`, which its README asks to run before any report | `.github/workflows/ci.yml:214-223` |
| `PROJECT_NUMBER` is "v1 (unreleased)"; the library and the CLI declare 0.1.0 | `.config/doxygen/Doxyfile:14` |
| Three rules ignore a `docs/api-site/` nothing writes | `.gitignore:23-25` |
| The manifest spells the repository URL in lower case; the remote is `SushiSystems/SushiAI` | `sushi-module.toml:12` |
