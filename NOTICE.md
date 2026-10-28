# Notices

SushiAI is Copyright (c) 2026-present Mustafa Garip & Sushi Systems and licensed under the PolyForm Noncommercial
License 1.0.0; see `LICENSE`. The material below keeps its own licence.

Nothing third-party is vendored, ported or redistributed from this tree: it has no
`third_party/` folder, no fetched source, no binary and no data set. Every component below is
installed on the build machine and found there.

## Third-party components

The repository pins no versions. The version column gives the declared range, or the version
installed in the SushiStack workspace on 2026-10-05 where the licence was read from that copy.

| Component | Where | Source | Version | Licence |
| --- | --- | --- | --- | --- |
| nlohmann/json | Found by `modules/core/config/CMakeLists.txt`; included by `modules/core/config/source/value.cpp` alone and compiled into the `sushiai` library | https://github.com/nlohmann/json | no pin; 3.12.0 installed (vcpkg `nlohmann-json`) | MIT, Copyright (c) 2013-2025 Niels Lohmann; read from the vcpkg `copyright` file |
| libjpeg-turbo | Found by `modules/vision/codecs/CMakeLists.txt`; included by that module's sources alone and linked into the `sushiai_vision` library | https://github.com/libjpeg-turbo/libjpeg-turbo | no pin; 3.1.4.1 installed (vcpkg `libjpeg-turbo`) | IJG licence and BSD-3-Clause, with the zlib licence on the SIMD code; read from the vcpkg `copyright` file |
| libpng | Found by `modules/vision/codecs/CMakeLists.txt`; included by that module's sources alone and linked into the `sushiai_vision` library | http://www.libpng.org/pub/png/libpng.html | no pin; 1.6.58 installed (vcpkg `libpng`) | PNG Reference Library License version 2, Copyright (c) 1995-2026 The PNG Reference Library Authors; read from the vcpkg `copyright` file |
| zlib | Not used directly; arrives through libpng | https://zlib.net | no pin; 1.3.2 installed (vcpkg `zlib`) | Zlib, (C) 1995-2026 Jean-loup Gailly and Mark Adler; read from the vcpkg `copyright` file |
| GoogleTest | Found by `tests/CMakeLists.txt`; linked by the test binaries only | https://github.com/google/googletest | no pin; 1.17.0 installed (vcpkg `gtest`) | BSD-3-Clause, Copyright 2008 Google Inc.; read from the vcpkg `copyright` file |
| hwloc | Not used directly; arrives through SushiRuntime (`cmake/Vcpkg.cmake`) | https://github.com/open-mpi/hwloc | no pin; 2.11.2 installed (vcpkg `hwloc`) | BSD-3-Clause; read from the vcpkg `copyright` file |
| typer | `cli/pyproject.toml`, runtime dependency of the `sa` CLI | https://github.com/fastapi/typer | `>=0.12`; 0.20.0 installed | MIT; read from the installed package metadata |
| rich | `cli/pyproject.toml`, runtime dependency of the `sa` CLI | https://github.com/Textualize/rich | `>=13.0`; 15.0.0 installed | MIT; read from the installed package metadata |
| tomli | `cli/pyproject.toml`, on Python older than 3.11 | https://github.com/hukkin/tomli | `>=2.0`; 2.4.0 installed | MIT; read from the installed package metadata |
| setuptools | `cli/pyproject.toml`, build backend | https://github.com/pypa/setuptools | `>=61.0`; 82.0.1 installed | MIT; read from the installed package metadata |
| pytest | `.github/workflows/ci.yml`, CLI tests only | https://github.com/pytest-dev/pytest | no pin; 9.1.1 installed | MIT; read from the installed package metadata |
| pyflakes | `.github/workflows/ci.yml`, lint step only | https://github.com/PyCQA/pyflakes | no pin | unverified; no copy was installed to read it from |
| SYCL toolchain (AdaptiveCpp or Intel oneAPI DPC++) | Chosen and installed by the SushiStack workspace; used through `cmake/SushiAISyclTarget.cmake` | https://github.com/AdaptiveCpp/AdaptiveCpp, https://github.com/intel/llvm | set by the workspace | unverified from this repository; SushiRuntime's `NOTICE.md` is where the runtime's terms are recorded |

## Sushi Systems modules

| Module | Where | Source | Version | Licence |
| --- | --- | --- | --- | --- |
| SushiRuntime | `cmake/Runtime.cmake`, `cli/sushistack.deps.toml` | https://github.com/SushiSystems/SushiRuntime | built as a package from the sibling checkout, no pin | Its own; see that repository's `LICENSE` and `NOTICE.md` |
| SushiBLAS | `cmake/BLAS.cmake`, `cli/sushistack.deps.toml` | https://github.com/SushiSystems/SushiBLAS | built as a package from the sibling checkout, no pin | Its own; see that repository's `LICENSE` and `NOTICE.md`, which lists the kernels ported from portBLAS |
| sushicore | `cli/pyproject.toml` | https://pypi.org/project/sushicore/ | `==0.9.7` | The licence the installed release carries |

## Earlier versions

No version of SushiAI has been tagged or released, and no package has been uploaded to PyPI.
The `LICENSE` file changed four times before this one, and each commit stays available under
the licence it carried to anyone who received it then.

| Commits | Dates | Licence in `LICENSE` |
| --- | --- | --- |
| `d45522b` up to, not including, `472ffa3` | 2025-06-08 to 2025-06-23 | CC0 1.0 Universal |
| `472ffa3` up to, not including, `f06c42a` | 2025-06-23 to 2025-07-02 | A text titled "BUSL-1.1 License for SushiAI", Copyright (c) 2025 Mustafa Garip |
| `f06c42a` up to, not including, `7988fd6` | 2025-07-02 to 2026-08-01 | MIT |
| `7988fd6` up to the commit before the one that added this file | 2026-08-01 to 2026-10-05 | Apache License 2.0 |

The CC0, BUSL-1.1 and MIT periods cover the 2025 prototype, which was deleted. Of the files
tracked today only `.gitignore`, `CMakeLists.txt`, `LICENSE` and `README.md` existed at a path
that predates `7988fd6`; everything else was first committed under Apache-2.0. The first commit
that carries this file is the first under the licence above.
