# Integrating SushiAI

How to consume SushiAI from another project today, at AI-1.

---

## 1. The fact that shapes everything

**SushiAI defines no kernels and writes no SYCL, yet every consumer needs a
SYCL compiler anyway.** The library's own surface is real — a graph IR, the
differentiation transform, layers, losses, optimizers, a `Dataset` seam and a
`Trainer` — but all compute is drawn from `SushiBLAS::sushiblas`, which
SushiAI links `PUBLIC`.

So the SYCL requirement a consumer inherits is **transitive, not direct**:
linking `SushiAI::sushiai` links `SushiBLAS::sushiblas` transitively, which
carries `-fsycl` (on the intel-llvm toolchain) as a `PUBLIC` compile/link
option, so a translation unit linking `SushiAI::sushiai` needs a SYCL-capable
compiler even if it only calls `SushiAI::version_string()`. See SushiBLAS's own
[`INTEGRATION.md` §1](https://github.com/SushiSystems/SushiBLAS/blob/main/docs/INTEGRATION.md#1-the-fact-that-shapes-everything)
for exactly why the flag has to be `PUBLIC` there — SushiAI inherits it
unchanged.

One consequence worth planning around: `modules/layers/umbrella/include/SushiAI/SushiAI.hpp` pulls in
both lanes. The **host lane** — `core/common.hpp`, `core/ops.hpp`,
`core/shape.hpp`, `core/check.hpp`, `core/status.hpp`, `core/result.hpp`,
`graph/ir.hpp`, `graph/builder.hpp`, `graph/memory_plan.hpp`,
`graph/fusion.hpp`, `graph/precision.hpp`, `graph/cast_insertion.hpp`,
`autograd/`, `nn/module.hpp`, `nn/layers.hpp`, `nn/loss.hpp` — reaches no SushiBLAS *engine*
and no SYCL (only sibling *vocabulary* headers: the dtype/layout enums, the
message formatter, `SushiRuntime::span` and `OpID`), so a translation unit
that only builds or inspects a graph can include those headers directly and
stay off the device compiler. Everything else, `core/allocation.hpp` and
`core/seam.hpp` included, is the device lane. See
[ARCHITECTURE.md §3](../architecture/ARCHITECTURE.md#3-the-library-target).

---

## 2. Consuming the package

### From an installed prefix

```bash
# In SushiAI's tree
sa build
cmake --install build --prefix /opt/sushiai

# In your project — CMAKE_PREFIX_PATH needs ALL THREE install prefixes:
# SushiAI, SushiBLAS, and SushiRuntime.
cmake -S . -B build \
    -DCMAKE_PREFIX_PATH="/opt/sushiai;/opt/sushiblas;/opt/sushiruntime" \
    -DCMAKE_CXX_COMPILER=<the same SYCL compiler SushiAI was built with>
```

```cmake
find_package(SushiAI REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE SushiAI::sushiai)
```

`tests/package/` in this repository is a complete, working example of
exactly this.

### From a sibling checkout

`add_subdirectory(path/to/sushiai)` also works and gives you the same
`SushiAI::sushiai` target — the same
[D3 layered resolution](../architecture/ARCHITECTURE.md#2-the-sibling-dependency-chain-d3)
SushiAI itself uses for its own two siblings applies one level up.

Set `SA_BUILD_TESTS=OFF` before that `add_subdirectory` if you do not want
SushiAI's suite compiled, enumerated by `gtest_discover_tests`, and
registered into your own CTest run. It defaults to `ON`.

### Version compatibility

`SameMinorVersion` compatibility while pre-1.0:
`find_package(SushiAI 0.1 REQUIRED)` accepts 0.1.x and rejects 0.2.0.

---

## 3. What you inherit, and what you must supply

Linking `SushiAI::sushiai` gives you:

| Inherited | Why |
|---|---|
| Include directories, the library itself | Obviously. |
| Everything `SushiBLAS::sushiblas` exports `PUBLIC` (its include paths, `-fsycl`, and SushiRuntime's floating-point contract) | SushiAI links SushiBLAS `PUBLIC` — see §1. |

You do **not** inherit SushiAI's own warnings/sanitizer/architecture policy
targets (`-Wall -Wextra -Werror -pedantic` is a rule for this tree, not for
yours) — see `cmake/SushiAIBuildPolicy.cmake`
and SushiBLAS's `ARCHITECTURE.md §10` for why that split matters for a
`STATIC` library's export set.

You must supply:

- **The same SYCL compiler** every library in the chain was built with.
- **C++17 — the same standard, not "or later".** See below.
- **`CMAKE_PREFIX_PATH` naming all three install prefixes** (SushiAI,
  SushiBLAS, SushiRuntime) unless all three share one prefix.
- **SushiRuntime's shared library on the loader path** at run time.

### Why the standard has to match, and does not just have to be "at least"

SushiAI is built at **C++17**, matching SushiBLAS and SushiRuntime — see
[ARCHITECTURE.md §1](../architecture/ARCHITECTURE.md#the-whole-stack-is-c17-and-that-is-not-a-style-choice)
for why the three have to agree at all.

The consequence for you is narrow but real. `SushiRuntime::span` is
`std::span` at C++20 and SushiRuntime's own shim at C++17, chosen on
`__cplusplus`, and SushiAI has out-of-line entry points whose parameter type
is that span — `Autograd::differentiate`, `Core::allocate_tensor`,
`Optim::StepScalars::write`, the gradcheck entry points. Compiled at C++20,
your call to one of them mangles differently from the symbol the library
exports:

```
library  (C++17)  ...V?$span@$$CBI@SushiRuntime@@@Z
consumer (C++20)  ...V?$span@$$CBI$0?0@4@@Z        <- std::span
```

That is an unresolved external at link time, not a warning. It is easy to hit
without touching a span yourself, because the header template
`Train::build_classifier` calls `Autograd::differentiate` for you.

Everything else is standard-agnostic: the headers *compile* under C++20
unchanged, and a translation unit that only builds or inspects a graph, or
only reads `SushiAI::version_string()`, links fine either way. If the rest of
your program must be C++20, the usual containment applies — put the code that
calls SushiAI in its own C++17 target behind an interface of your own types
and link that in. That is a workaround, not a supported configuration: it is
your responsibility that nothing standard-library-versioned crosses the
boundary. `tests/package/` in this repository is a working C++17 consumer;
its `CMakeLists.txt` carries the same explanation at the point where you
would otherwise change the standard.

This restriction lifts, without an API change, when the whole stack moves back
to C++20 together.

---

## 4. Checking the binary you actually linked

```cpp
#include <SushiAI/version.hpp>

std::printf("Built against SushiAI headers %s\n", SUSHIAI_VERSION_STRING);
```

Like SushiBLAS, SushiAI is `STATIC`: there is no separately loaded artifact
whose version could drift from `SUSHIAI_VERSION_STRING` at run time, so the
compile-time macro is the whole story.

---

## 5. Troubleshooting

**`find_package(SushiAI)` succeeds but SushiBLAS or SushiRuntime fails to
resolve downstream.** `CMAKE_PREFIX_PATH` must name all three install
prefixes — `SushiAIConfig.cmake` calls `find_dependency(SushiRuntime)` and
`find_dependency(SushiBLAS)` internally.

**A parse error or link error mentioning `sycl`.** Your translation unit (or
the whole target) is being compiled by a non-SYCL compiler — see §1.

**An unresolved external naming `span` — `differentiate`, `allocate_tensor`,
`StepScalars::write`, a gradcheck entry point, or anything reached through
`Train::build_classifier`.** Your translation unit is being compiled at C++20
against a C++17 library. Compile it at C++17 — see §3.

**A warning about an ignored return value.** These device-lane entry points
are `[[nodiscard]]` and return a `Status` or a `Result<T>`: `Trainer::make`,
`AdamW::make`, `SGD::make`, `CompiledStep::compile`, `bind_parameters`,
`NN::initialize`, `Trainer::fit`. Test `is_ok()` or `has_value()` before
`value()`; reading the value of a failed `Result` stops the process. Host-lane
calls such as `Train::build_classifier` and `Autograd::differentiate` report a
bad graph by throwing `SushiAI::Error`.

**Link errors on a BLAS symbol you expected `SushiAI` to provide.** SushiAI
defines no ops of its own — check you're calling `SushiBLAS::sushiblas`'s API
through `engine.blas()`/etc. SushiAI's own surface is the graph, the
differentiation transform, and the `nn`/`optim`/`data`/`train` layers built on
top of it.
