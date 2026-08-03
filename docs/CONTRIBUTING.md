# Contributing to SushiAI

SushiAI is currently a personal project developed by Mustafa Garip, at the
**AI-1 stage** — a graph-native autograd core that trains an MLP end to end,
with memory planning, fusion and mixed precision still ahead. To maintain a
unified vision while the architecture is still being designed, **external
pull requests are not being accepted at this stage** — but bug reports and
design discussion are welcome, and this document is the reference for how the
project is actually built.

By contributing you agree that your contributions are licensed under the
project's **Apache License, Version 2.0** (see [`LICENSE`](../LICENSE)).

> New to the codebase? Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first, then
> [`docs/slop/make_sushiai_great_again.md`](slop/make_sushiai_great_again.md)
> for what's actually planned and in what order. `sa demo mlp` is the
> fastest way to see the whole pipeline do something.

---

## 1. Getting set up

```bash
curl -fsSL https://sushisystems.io/install.sh | bash
```

```powershell
irm https://sushisystems.io/install.ps1 | iex
```

If you already have SushiRuntime and SushiBLAS sibling checkouts and the
shared bundled SYCL toolchain:

```bash
ss install-cli sushiai   # puts `sa` / `sushiai` on PATH
sa build                 # Release (default)
sa test                  # functional suite
```

See [`CLI_GUIDE.md`](CLI_GUIDE.md) for the full command reference.

---

## 2. Before you start coding

- This is a personal project; if you're not the maintainer, the productive
  path is an issue — a bug report or a design discussion — not a PR.
- **AI-2 onward is where the open design decisions are.** The IR, the
  differentiation transform, the layer/optimizer/data/training layers and the
  compile-once contract are all decided in code now; memory planning, fusion
  and mixed precision are not — and §25 of the master plan revises what §17–19
  originally said about them. Read it before assuming a shape.
- **Keep changes focused.** One logical change per commit.

---

## 3. Building and testing your change

```bash
sa build                          # Release
sa build --type debug             # Debug
sa build --clean                  # wipe the build tree first
sa build -D SB_SYCL_TARGETS=...   # any CMake cache variable, repeatable

sa test --suite unit              # fast, run this often
sa test --suite integration
sa test --suite regression
sa test --suite all

sa demo mlp --epochs 1 --batches 2   # the showcase, in a couple of seconds
```

Expectations for a mergeable change:

1. **The full test suite passes** (`sa test --suite all`).
2. **A new symbol ships with a test.** Fast, deterministic tests go in
   `tests/functional/unit/`; anything exercising the real Engine/sibling
   chain goes in `tests/functional/integration/` (see
   `test_engine_smoke.cpp` for the shape); a guarantee whose failure would be
   *silent* — the loop still converges, the run still reports success — goes
   in `tests/functional/regression/` (see `test_compile_once.cpp`).
   **The GTest suite-name prefix is the only thing that assigns a CTest
   label**, so a suite named without `Unit_`/`Integration_`/`Regression_` is
   invisible rather than failing.
3. **A CMake or CLI change is verified end to end** — configure, build,
   test, and (for a packaging change) `cmake --install` + `tests/package/`
   against the installed prefix.
4. **A change to `sa demo mlp`'s output keeps the compile-count line.** Two
   tests pin it: the `Integration_Demo.MlpRunsAndReportsOneCompile` CTest
   entry and `cli/tests/test_demo.py`.

---

## 4. Coding style

SushiAI follows the same conventions as its siblings (SushiRuntime,
SushiBLAS) — match the file you're editing.

### License header

Every file starts with this signature (replace `[filename]`):

```cpp
/**************************************************************************/
/* [filename]                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/
```

### C++ standard, naming, braces

- **C++17**, matching SushiBLAS and SushiRuntime. The whole stack has to agree
  on one standard because `SushiRuntime::span` picks between `std::span` and
  SushiRuntime's own shim on `__cplusplus`, and that choice reaches the
  compiled ABI — see
  [ARCHITECTURE.md §1](ARCHITECTURE.md#the-whole-stack-is-c17-and-that-is-not-a-style-choice)
  for the full reasoning and the table of C++17 spellings this implies
  (`SushiRuntime::span`, `SushiBLAS::Core::format`, `constexpr bool` traits in
  place of concepts, `constexpr` in place of `consteval`). 4-space indentation.
- **Classes/structs/namespaces**: `PascalCase` (`Engine`, `SushiAI`).
- **Functions and variables**: `snake_case`.
- **Private members**: trailing underscore.
- **Allman braces** for multi-statement blocks; a single-statement
  `if`/loop omits braces (see SushiBLAS's
  [`CONTRIBUTING.md` §4](https://github.com/SushiSystems/SushiBLAS/blob/main/docs/CONTRIBUTING.md#4-coding-style)
  for the full examples — the rule is identical here).

### Documentation

Doxygen-style comments on every public API: `@brief` for the summary,
`@param` per parameter, `@return` for the result.

### Build-system names are prefixed with the full program name

Every CMake target, function, macro, and internal artifact this repo
defines starts with `sushiai_` — `sushiai_apply_build_policy`,
`sushiai_apply_sycl`. Never the bare `sushi_` prefix.

This is a **SushiStack-wide convention**: `sushiruntime_`, `sushiblas_`,
`sushiengine_`, `sushiai_` each stay in their own namespace on the shared
`CMAKE_MODULE_PATH` D3's sibling `add_subdirectory` puts every checked-out
repo's `cmake/` directory on. A name defined twice in one configure (e.g. two
repos both defining a bare `sushi_apply_build_policy`) is a hard CMake error
with no scoping to fall back on — see
[ARCHITECTURE.md §4](ARCHITECTURE.md#4-packaging) for the exact collision this
prevents. `Vcpkg.cmake` and `Toolchain.cmake` are the deliberate exception:
their logic is idempotent regardless of which sibling's copy loads first, so
they stay unprefixed.

---

## 5. Documentation is part of the change

- **Changelog entries start at v1.0.0.** The project is pre-1.0. Until then,
  [`CHANGELOG.md`](CHANGELOG.md)'s `[Unreleased]` section documents
  prototype-era milestones informally.
- **A change to the dependency chain, the library target, or the packaging
  shape updates [`ARCHITECTURE.md`](ARCHITECTURE.md)** — every class, CMake
  function, and file it names must still exist.
- **Any CLI change updates [`CLI_GUIDE.md`](CLI_GUIDE.md)**, and the
  build/test snippets here and in the README that invoke it.
- **A new dependency updates [`cli/sushistack.deps.toml`](../cli/sushistack.deps.toml)**
  — see [§7](#7-adding-a-c-dependency).

---

## 6. Commits

### Branch names

```
feature/ai1-graph-api-skeleton
fix/blas-sibling-resolution-order
chore/bump-googletest
docs/architecture-packaging-section
```

### Commit headers

`type: short summary` — lowercase, imperative mood, no trailing period:

```
feat(ai0): add the two-sibling D3 dependency chain
fix(cli): resolve the C compiler slot for a clang++-personality binary
docs: document what AI-0 does and does not include
```

### Commit body

Bullet points, each starting with a past-tense verb, plain English,
technical, objective — no marketing language.

---

## 7. Adding a C++ dependency

Dependency provisioning is owned by [SushiStack](https://sushisystems.io)
(the `ss` CLI). SushiAI's own manifest fragment is
[`cli/sushistack.deps.toml`](../cli/sushistack.deps.toml):

```toml
[module]
depends_on = ["sushiruntime", "sushiblas"]
```

`depends_on` folds both siblings' own toolchain requirements into SushiAI's
readiness picture. If SushiAI ever needs a library neither sibling already
provisions, add a top-level TOML table the way SushiRuntime's own fragment
documents (description/required/linux_apt/windows_vcpkg/check_cmd), then
update `CMakeLists.txt` to find and link it. Verify with `ss doctor` and
`ss install`.

---

## 8. Reporting bugs and requesting features

Use the GitHub issue templates. A good bug report includes the platform and
toolchain, the build type, the exact `sa` command, and the smallest
reproducer you can manage. For anything that looks like a **security** issue,
do **not** open a public issue — follow [`SECURITY.md`](SECURITY.md).

---

## 9. Questions

Open a GitHub Discussion, join the community Discord (see
[`SECURITY.md`](SECURITY.md) for the current invite), or email
**mustafagarip@sushisystems.io**.

Please be respectful in all project spaces — see
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
