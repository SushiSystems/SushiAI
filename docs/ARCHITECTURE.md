# SushiAI Architecture

This document describes how SushiAI is built today. It is intentionally
short: SushiAI is at **AI-0**, a scope-reset skeleton, and this document
names only what actually exists under `include/SushiAI/`, `src/`, and
`cmake/`. It will grow section by section as AI-1 (graph-native autograd)
and beyond land — see
[`docs/slop/make_sushiai_great_again.md`](slop/make_sushiai_great_again.md)
for what's coming and in what order.

> If you only want to *use* the library, the top-level [README.md](../README.md)
> is enough.

---

## 1. What SushiAI is (and isn't, yet)

SushiAI is meant to become the AI/ML head of the SushiStack: a graph-native
autograd core with memory planning, kernel fusion, and mixed precision,
built on [SushiBLAS](https://github.com/SushiSystems/SushiBLAS)'s tensor and
op library. **None of that exists yet.** What exists is the scaffolding a
real library needs from day one so that no future commit ever has to be
"the one that finally adds packaging" or "the one that finally sets up
testing":

- A library target (`SushiAI::sushiai`) that actually compiles, links
  against both of its intended dependencies, and installs/exports correctly.
- A real dependency chain to two siblings, resolved the same layered way
  every other SushiStack module resolves its dependencies.
- A test suite structure (unit/integration, CTest labels) ready to grow.
- `find_package(SushiAI)` packaging, verified with a real out-of-tree
  consumer — not deferred to "later, when there's something worth
  packaging."
- A CLI (`sa`) with the full command surface (`build`/`test`/`run`/`clean`/
  `doxygen`/`config`/`env`) a real project needs, plus a `demo mlp` command
  that exists and correctly reports "not implemented yet" rather than
  faking output.

SushiAI touches no SYCL directly and defines no operations of its own. Every
compute call in the one example/test that exists is a `SushiBLAS::sushiblas`
call.

---

## 2. The sibling dependency chain (D3)

SushiAI depends on two siblings, resolved independently through the same
three-mode, layered CMake pattern used stack-wide (`cmake/Runtime.cmake`,
`cmake/BLAS.cmake`):

1. **TARGET-guard** — reuse an already-defined `SushiRuntime::SushiRuntime` /
   `SushiBLAS::sushiblas` if a superproject already brought it in.
2. **`find_package(... QUIET)`** — an installed package.
3. **Sibling `add_subdirectory`** — the common case: `../sushiruntime` /
   `../sushiblas` next to this checkout.

**Order matters.** `Runtime.cmake` runs before `BLAS.cmake` in the root
`CMakeLists.txt` specifically so that when SushiBLAS's own internal
`Runtime.cmake` runs (SushiBLAS `add_subdirectory`'d as SushiAI's dependency
also depends on SushiRuntime), its own TARGET-guard already finds
`SushiRuntime::SushiRuntime` defined and skips re-adding it — one runtime in
the whole build tree, not two competing copies.

Every consumer downstream — SushiAI's own targets, the CLI's `-D` flags, an
out-of-tree `find_package(SushiAI)` consumer — links the same two namespaced
aliases regardless of which mode actually resolved them.

---

## 3. The library target

`src/CMakeLists.txt` defines one target:

```cmake
add_library(sushiai STATIC version.cpp)
add_library(SushiAI::sushiai ALIAS sushiai)

target_link_libraries(sushiai PUBLIC
    SushiRuntime::SushiRuntime
    SushiBLAS::sushiblas
)
```

Both dependencies are linked `PUBLIC` — a future autograd core built inside
`sushiai` will need both, and every current consumer (examples, tests)
already needs `SushiBLAS::sushiblas` to do anything, so there is nothing to
gain by hiding the link. This also means SushiAI inherits SushiRuntime's
`sushiruntime_deterministic_fp` floating-point contract transitively; it
owns no copy of that guarantee itself.

`include/SushiAI/version.hpp` defines `SUSHIAI_VERSION_{MAJOR,MINOR,PATCH,STRING}`
and `SUSHIAI_VERSION_ENCODE`; `src/version.cpp` implements the one function
`SushiAI::version_string()` declared there. `include/SushiAI/SushiAI.h` is
the master header — today it includes only `version.hpp`, and grows one
include per header SushiAI adds.

SYCL device-compilation policy (`sushiai_apply_sycl()`,
`cmake/SushiAISyclTarget.cmake`) is applied to `sushiai` even though
`version.cpp` itself has no SYCL content — applied `PUBLIC` on the intel-llvm
path (`-fsycl`) so that examples and tests inherit it from linking `sushiai`
rather than reapplying it themselves, and so the flag is already in place
for the first source file that does need it.

---

## 4. Packaging

SushiAI is `find_package`-able from its very first commit, mirroring
SushiBLAS's SB-6 packaging shape exactly: `SushiAI::sushiai` (a `STATIC`
library, aliased in-tree and exported out-of-tree under the same name),
`install(EXPORT)` / `export(EXPORT)`, a generated `SushiAIConfig.cmake`
(`find_dependency(SushiRuntime)` and `find_dependency(SushiBLAS)`, each
guarded by a TARGET-guard so a consumer that already resolved either
dependency itself doesn't get a duplicate), `SameMinorVersion` package
version compatibility (pre-1.0), and a version-drift check between
`include/SushiAI/version.hpp` and the CMake project version at configure
time.

`sushiai_apply_build_policy()` (`cmake/SushiAIBuildPolicy.cmake`) composes
warnings/architecture/sanitizer policy targets and pulls their flag *values*
onto the real target via `target_compile_options`/
`target_compile_definitions`/`target_link_options` +
`$<TARGET_PROPERTY:...>`, never `target_link_libraries` — the STATIC-library
export-set-safe pattern SushiBLAS's own SB-6 work established (see
SushiBLAS's `ARCHITECTURE.md §10`): linking a PRIVATE policy target, even
`BUILD_INTERFACE`-wrapped, records its name in the exported
`INTERFACE_LINK_LIBRARIES` and breaks `install(EXPORT)`/`export(EXPORT)`.

`tests/package/` is a real out-of-tree consumer — `find_package(SushiAI)`
against an install prefix, no access to this source tree — that runs a real
op through both transitively-resolved siblings.

CMake module files that define SushiAI-specific functions
(`SushiAIBuildPolicy.cmake`, `SushiAICompilerWarnings.cmake`,
`SushiAIOptimization.cmake`, `SushiAISanitizers.cmake`, `SushiAISyclTarget.cmake`)
are prefixed with the program's full name, because D3's sibling
`add_subdirectory` puts every checked-out repo's `cmake/` directory on one
shared `CMAKE_MODULE_PATH` — an unprefixed `SyclTarget.cmake` here would
shadow (or be shadowed by) SushiBLAS's own file the moment both are
`add_subdirectory`'d together, exactly the collision SushiBLAS's own SB-6
work found and fixed. `Vcpkg.cmake` and `Toolchain.cmake` stay unprefixed
because their logic is idempotent regardless of which sibling's copy loads
first.

---

## 5. What's not here yet

Everything past the scaffolding above is **AI-1 onward** and does not exist
in this tree: no tensor/graph API of SushiAI's own, no autograd, no layers,
no optimizers, no loss functions, no training loop, no `sa demo mlp`
implementation. The retired 2025-06 prototype (`core/tensor.h`, eager
autograd, CUDA-only) that used to occupy this repository is **not** a
migration source for any of this — it predates the SushiStack integration
entirely and was deleted, not ported, at AI-0. See the master plan for the
actual sequencing.

## Project layout

See the top-level [README.md § Project layout](../README.md#project-layout)
for the full directory tree.
