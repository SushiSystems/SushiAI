# Integrating SushiAI

How to consume SushiAI from another project today, at AI-0. This will grow
substantially once SushiAI has an API surface of its own (AI-1 onward); for
now, integrating SushiAI mostly means integrating what it links.

---

## 1. The fact that shapes everything (for now)

**SushiAI's own public header (`include/SushiAI/SushiAI.h`) exposes only a
version accessor today — no SYCL types, no tensor API, nothing to instantiate
a kernel from.** Everything a consumer actually calls comes from
`SushiBLAS::sushiblas`, which SushiAI links `PUBLIC`.

This means the SYCL requirement a consumer inherits is **transitive, not
direct**: linking `SushiAI::sushiai` links `SushiBLAS::sushiblas`
transitively, which carries `-fsycl` (on the intel-llvm toolchain) as a
`PUBLIC` compile/link option, so a translation unit linking `SushiAI::sushiai`
needs a SYCL-capable compiler even if it only calls
`SushiAI::version_string()`. See SushiBLAS's own
[`INTEGRATION.md` §1](https://github.com/SushiSystems/SushiBLAS/blob/main/docs/INTEGRATION.md#1-the-fact-that-shapes-everything)
for exactly why the flag has to be `PUBLIC` there — SushiAI inherits it
unchanged.

This section will be rewritten once AI-1 lands SushiAI's own API surface.

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
[D3 layered resolution](ARCHITECTURE.md#2-the-sibling-dependency-chain-d3)
SushiAI itself uses for its own two siblings applies one level up.

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

You do **not** inherit SushiAI's own (currently minimal) warnings/
sanitizer/architecture policy targets — see `cmake/SushiAIBuildPolicy.cmake`
and SushiBLAS's `ARCHITECTURE.md §10` for why that split matters for a
`STATIC` library's export set.

You must supply:

- **The same SYCL compiler** every library in the chain was built with.
- **C++20 or later.**
- **`CMAKE_PREFIX_PATH` naming all three install prefixes** (SushiAI,
  SushiBLAS, SushiRuntime) unless all three share one prefix.
- **SushiRuntime's shared library on the loader path** at run time.

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

**Link errors on a BLAS symbol you expected `SushiAI` to provide.** Nothing
here is SushiAI's own yet (AI-0) — check you're calling
`SushiBLAS::sushiblas`'s API through `engine.blas()`/etc., not something this
document implies exists on `SushiAI::sushiai` itself.
