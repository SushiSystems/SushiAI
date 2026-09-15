# Agent Directives

These directives apply to every automated agent operating in this repository.

## 1. Maintainability first, performance second

Maintainability is the primary constraint. Code is written once and read hundreds of times across its lifespan. Every subsystem must be modular, readable, and decoupled. Optimize hot paths without sacrificing module boundaries, encapsulation, or clarity.

## 2. SOLID discipline without exception

Every unit has a single responsibility. Classes and components must be detachable and testable in isolation. Sibling components must share uniform structural shapes. Never apply temporary fixes that only pass tests by coincidence.

## 3. The CLI is the only door (`sa`)

Execute all build, test, and runtime operations through the repository's dedicated CLI:

```bash
sa build
sa test
sa run <target>
sa doxygen
```

Never invoke `cmake`, `ninja`, `ctest`, or compiler binaries directly. Only `sa` configures the required runtime environment, compiler flags, and dynamic library search paths.

## 4. C++17 ABI agreement

The entire SushiStack compiles against the C++17 standard (`-std=c++17`). Do not introduce C++20 features (such as `std::span`, concepts, or `std::format`). Downstream consumers and sibling libraries rely on strict C++17 ABI compatibility across shared boundaries.

## 5. D3 layered resolution

Dependencies resolve through the D3 layered strategy:
1. In-tree CMake target if already part of a superproject build.
2. Installed package located via `find_package`.
3. Sibling checkout (`../sushiruntime`, `../sushiblas`).

Never hardcode machine-specific absolute paths in build definitions or scripts.

## 6. Strict quality bar

Work is complete only when all invariants, tests, and formatting checks pass. "Good enough for now" is not acceptable. When a task cannot be completed within scope, report exact progress honestly. Name what was completed and what remains unfinished.
