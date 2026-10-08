# Known issues

Defects that exist today, each with where it sits. Planned work is in
[`REMAINING_WORK.md`](../design/REMAINING_WORK.md).

## Library behaviour

| Issue | Where |
| --- | --- |
| In mixed precision an overflowed gradient skips that parameter's update only; the other parameters still step, so one step can update some layers and not others | `include/SushiAI/optim/loss_scaler.hpp` |
| Differentiating `K_LOG_SOFTMAX` on its own throws; it differentiates only as the fused `K_LOG_SOFTMAX` + `K_NLL_LOSS` pair | `src/autograd/rules/loss.cpp` |
| `gradcheck` cannot verify a mixed-precision graph: a central difference in fp16 has a noise floor near 1e-2. Check the rule in FLOAT64 | `include/SushiAI/autograd/gradcheck.hpp` |
| NVIDIA Pascal GPUs (GTX 1060, compute capability 6.1) fail when the OpenCL loader ingests SPIR-V. Workaround below | The oneAPI OpenCL loader, outside this repository |

On a Pascal machine, send the CLI, the tests and training to the OpenCL CPU runtime:

```powershell
$env:ONEAPI_DEVICE_SELECTOR="opencl:cpu"; pytest cli/tests -q
```

## Documentation

| Issue | Where |
| --- | --- |
| Names `docs/api-site/` as the Doxygen output; the `Doxyfile` writes `build/docs/api-site` | `CLAUDE.md:10` |
| Name the command `sa doxygen`; it is `sa docs`, and the old word runs until 2027-03-25 | `CLAUDE.md:57`, `AGENTS.md:21` |
| The demo transcript says "17 IR nodes over 22 values" and the text under it says "21 graph nodes"; neither was checked against a run | `README.md:268`, `README.md:311` |
| Comments cite `docs/ARCHITECTURE.md`; the file is `docs/architecture/ARCHITECTURE.md` | `CMakeLists.txt:61`, `src/CMakeLists.txt:113` |
| Comments cite `docs/README.md` for the demo's quoted output; that text is in the root `README.md` | `examples/demo_mlp.cpp`, `tests/functional/regression/test_mixed_precision.cpp` |
| Comments cite `docs/TUTORIAL.md`; the file is `docs/getting_started/TUTORIAL.md` | `tests/functional/unit/test_module.cpp` |
| The links to the SECURITY pages of SushiBLAS and SushiRuntime were not checked against those repositories' `main` | `docs/reference/SECURITY.md` |
| No changelog entry records the `sushicore` floor moving from 0.4.0 to 0.7.0, and 0.7.0 was not confirmed on PyPI | `cli/pyproject.toml:15` |

## Build, CI and root files

| Issue | Where |
| --- | --- |
| The `cli` job installs `sushicore` from `./.sushistack/sushicore`, a folder SushiStack no longer has, so the CLI tests do not run | `.github/workflows/ci.yml:166-179` |
| The pyflakes step ends in `\|\| true` and cannot fail | `.github/workflows/ci.yml:184` |
| The workflow header cites `docs/CLAUDE.md` and says `sushicore` is on no index | `.github/workflows/ci.yml:26`, `.github/workflows/ci.yml:167` |
| No job runs the checkers under `tools/`; both jobs run on Linux only | `.github/workflows/ci.yml` |
| `PROJECT_NUMBER` is "v1 (unreleased)"; the library and the CLI declare 0.1.0 | `Doxyfile:14` |
| A comment says the Doxyfile is at `docs/Doxyfile`, and three rules ignore a `docs/api-site/` nothing writes | `.gitignore:22-25` |
| The manifest spells the repository URL in lower case; the remote is `SushiSystems/SushiAI` | `sushi-module.toml:12` |
| `rule_module_readmes` reads `modules/<tier>/` only, so it reports nothing in this single-module tree | `tools/documentation/check_docs_layout.py` |
