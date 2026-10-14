# Contributing to SushiAI

SushiAI is currently a personal project developed by Mustafa Garip. Its state is
recorded in [`docs/design/PROGRAMME.md`](design/PROGRAMME.md): stage 1, the
SushiBLAS v0.1.0 migration, is built and passing, and the naming half of stage 2
is done. To maintain a unified vision while the architecture is still being
designed, **external pull requests are not being accepted at this stage**. Bug
reports and design discussion are welcome, and this document is the reference
for how the project is built.

Contributions from outside Sushi Systems are not accepted yet. SushiAI is
source-available, free for non-commercial use under the PolyForm Noncommercial
License 1.0.0 (see [`LICENSE`](../LICENSE)); commercial use needs a licence from
Sushi Systems, see [`COMMERCIAL.md`](../COMMERCIAL.md).

> New to the codebase? Start with [`TUTORIAL.md`](getting_started/TUTORIAL.md) — it walks from
> an empty checkout to a model that trains, and then explains what the library
> did with it. Then [`ARCHITECTURE.md`](architecture/ARCHITECTURE.md) for the design,
> [`docs/design/PROGRAMME.md`](design/PROGRAMME.md) for the stages in the order they
> are planned, and [`REMAINING_WORK.md`](design/REMAINING_WORK.md) for what is still
> open. `sa demo mlp` is the fastest way to see the whole pipeline do something.

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
hub install-cli sushiai   # puts `sa` / `sushiai` on PATH
sa build                 # Release (default)
sa test                  # functional suite
```

See [`CLI_GUIDE.md`](guides/CLI_GUIDE.md) for the full command reference.

---

## 2. Before you start coding

- This is a personal project; if you're not the maintainer, the productive
  path is an issue — a bug report or a design discussion — not a PR.
- **The open design decisions are in [`PROGRAMME.md`](design/PROGRAMME.md).** The
  IR, the differentiation transform, the layer/optimizer/data/training layers, the
  compile-once contract, memory planning, fusion and the precision policy are all
  decided in code now, and so is mixed precision's *numerics* half: loss scaling
  and overflow backoff. Read the design document of a stage before changing the
  subsystem it covers; the open items are in
  [`REMAINING_WORK.md`](design/REMAINING_WORK.md).
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
sa run sushiai_core_graph_tests -- --gtest_brief=1   # one module's program; args after -- go to it

sa demo mlp --epochs 1 --batches 2   # the showcase, in a couple of seconds
```

`sa test` runs the CTest labels that the GTest suite prefixes assign. Each module builds one
test program, `sushiai_<tier>_<module>_tests`, and `sa run` runs one of them in one process.
SushiBLAS's own test suite does not enter this tree: its `SB_BUILD_TESTS` option
defaults to off when SushiBLAS is added as a subdirectory, as this tree does.

Expectations for a mergeable change:

1. **The full test suite passes** (`sa test --suite all`).
2. **A new symbol ships with a test.** The test of a header
   `modules/<tier>/<module>/include/SushiAI/.../<name>.hpp` is
   `tests/<tier>/<module>/.../test_<name>.cpp`, listed in the `SOURCES` of
   `tests/<tier>/<module>/CMakeLists.txt`. A test that uses several modules belongs to the
   highest-tier one, in the test file of the unit whose behaviour it checks;
   `tests/README.md` has the rest of the tree. **The GTest suite-name prefix is the only thing
   that assigns a CTest label**, so a suite named without `Unit_`/`Integration_`/`Regression_`
   is invisible rather than failing. A guarantee whose failure would be silent, such as the
   compile count in `tests/layers/train/test_trainer/test_compile_once.cpp`, takes the
   `Regression_` prefix.
3. **A CMake or CLI change is verified end to end** — configure, build,
   test, and (for a packaging change) `cmake --install` + `tests/package/`
   against the installed prefix.
4. **A change to `sa demo mlp`'s output keeps the compile-count line.** Two
   tests pin it: the `Integration_Demo.MLPRunsAndReportsOneCompile` CTest
   entry and `cli/tests/test_demo.py`.

---

## 4. Coding style

SushiAI follows the same conventions as its siblings (SushiRuntime,
SushiBLAS) — match the file you're editing.

### License header

Every source file opens with the license block of the `source-comments` skill: the file
name, the project line `SushiAI - https://github.com/SushiSystems/SushiAI`, the copyright
line and the two licence lines. `tools/licensing/write_license_block.py` writes it;
nobody types it by hand.

### C++ standard, naming, braces

- **C++17**, matching SushiBLAS and SushiRuntime. The whole stack has to agree
  on one standard because `SushiRuntime::span` picks between `std::span` and
  SushiRuntime's own shim on `__cplusplus`, and that choice reaches the
  compiled ABI — see
  [ARCHITECTURE.md §1](architecture/ARCHITECTURE.md#the-whole-stack-is-c17-and-that-is-not-a-style-choice)
  for the full reasoning and the table of C++17 spellings this implies
  (`SushiRuntime::span`, `SushiBLAS::Core::format`, `constexpr bool` traits in
  place of concepts, `constexpr` in place of `consteval`). 4-space indentation.
- **Classes/structs/namespaces**: `PascalCase` (`CompiledStep`, `TensorPool`, `SushiAI`).
- **Enumerators**: `PascalCase` (`Lifetime::Parameter`, `StatusCode::Ok`). **Constants**
  carry the `K_` prefix (`K_ALL`, `K_LAYER_REGISTRY`).
- **Functions and variables**: `snake_case`.
- **Private members**: trailing underscore.
- **Allman braces** for multi-statement blocks; a single-statement
  `if`/loop omits braces (see SushiBLAS's
  [`CONTRIBUTING.md` §4](https://github.com/SushiSystems/SushiBLAS/blob/main/docs/CONTRIBUTING.md#4-coding-style)
  for the full examples — the rule is identical here).

### Documentation

Source comments follow the `source-comments` skill, which this page does not restate.
`python tools/documentation/check_source_comments.py include src` checks them. A comment
block is at most eight lines, which the checker holds; a `//` comment is one line that
states an invariant. The `cpp-code-style` skill sets the 100-column limit.

### Failures

- A content error, one a caller can cause with the input it supplies, is a `Status` or a
  `Result<T>` (`core/status.hpp`, `core/result.hpp`). `SA_FAIL_IF` makes one; `SA_TRY` and
  `SA_TRY_ASSIGN` pass one on.
- A programmer's misuse stops the process with `SA_FATAL_IF` (`core/check.hpp`).
- No exception crosses the public surface. A sibling's exception becomes a `Status` only in
  the guards of `core/seam.hpp`.
- This is the target of stage 2 of [`PROGRAMME.md`](design/PROGRAMME.md). The tree does not
  meet it yet; [`KNOWN_ISSUES.md`](reference/KNOWN_ISSUES.md) lists where it does not.

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
[ARCHITECTURE.md §6](architecture/ARCHITECTURE.md#6-packaging) for the exact collision this
prevents. `Vcpkg.cmake` and `Toolchain.cmake` are the deliberate exception:
their logic is idempotent regardless of which sibling's copy loads first, so
they stay unprefixed.

---

## 5. Documentation is part of the change

- **Every change adds one line under `## Unreleased`** in
  [`CHANGELOG.md`](reference/CHANGELOG.md), in the shape
  [`DOCUMENTATION_STYLE_GUIDE.md`](DOCUMENTATION_STYLE_GUIDE.md) §5 gives.
- **Work that opens or closes updates
  [`REMAINING_WORK.md`](design/REMAINING_WORK.md)**, the only backlog; a defect that stays goes
  in [`KNOWN_ISSUES.md`](reference/KNOWN_ISSUES.md).
- **A change under `docs/` passes both documentation checkers**; the commands are in
  [`tools/README.md`](../tools/README.md).
- **A change to the dependency chain, the library target, or the packaging
  shape updates [`ARCHITECTURE.md`](architecture/ARCHITECTURE.md)** — every class, CMake
  function, and file it names must still exist.
- **Any CLI change updates [`CLI_GUIDE.md`](guides/CLI_GUIDE.md)**, and the
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

`sa setup` provisions what the dependency fragments list. SushiAI's own
fragment is
[`cli/sushistack.deps.toml`](../cli/sushistack.deps.toml):

```toml
[module]
depends_on = ["sushiruntime", "sushiblas"]
```

`depends_on` folds both siblings' own toolchain requirements into SushiAI's
readiness picture. If SushiAI ever needs a library neither sibling already
provisions, add a top-level TOML table the way SushiRuntime's own fragment
documents (description/required/linux_apt/windows_vcpkg/check_cmd), then
update `CMakeLists.txt` to find and link it. Verify with `sa setup --dry-run`
and `sa doctor`; `hub install` provisions several checkouts at once.

---

## 8. Adding an operation

An operation is one identity and one row in each table that owns a piece of it. The
headers named below carry `static_assert`s that fail the build when a row is missing or
doubled, so the compiler lists what is left.

1. **Identity.** Declare it in `modules/core/core/include/SushiAI/core/ops.hpp` and add it to `K_ALL`. Use
   `differentiable_op` when a backward rule must exist, `gradient_barrier_op` when its
   result lets no gradient reach its operands, and `op` otherwise. The name is dotted and
   starts with `sushiai.`.

   ```cpp
   // modules/core/core/include/SushiAI/core/ops.hpp, inside namespace Ops
   inline constexpr OpName K_SOFTSIGN = differentiable_op("sushiai.softsign");
   ```

2. **Builder call and shape rule.** Declare the emitter in
   `modules/core/graph/include/SushiAI/graph/builder.hpp` and define it in
   `modules/core/graph/source/builder.cpp`. The body
   is the shape rule: it sets the result's `Shape` and element type from its operands and
   refuses an operand it cannot use. `unary` and `elementwise` do this for the common
   cases.

   ```cpp
   // modules/core/graph/source/builder.cpp
   ValueID GraphBuilder::softsign(ValueID x) const
   {
       return unary(Ops::K_SOFTSIGN.id, x);
   }
   ```

3. **Gradient rule row.** Declare the rule in the header of its domain under
   `modules/core/autograd/include/SushiAI/autograd/rules/`, define it in
   `modules/core/autograd/source/rules/`, and add one row
   to the domain table, raising that table's `std::array` size by one.

   ```cpp
   // modules/core/autograd/include/SushiAI/autograd/rules/elementwise.hpp
   void softsign_backward(const GradientTape& tape);

   inline constexpr std::array<RuleEntry, 13> K_ELEMENTWISE_RULES = {{
       // ... the twelve existing rows ...
       {Ops::K_SOFTSIGN.id, &softsign_backward},
   }};
   ```

   An activation takes one row in `K_ACTIVATIONS` (`core/ops.hpp`) and one in
   `K_ACTIVATION_RULES`, and its gradient is a plain `op` identity with its own lowering row.

4. **Lowering row.** Declare `[[nodiscard]] Status lower_softsign(const LoweringContext& context);`
   in the domain header under `modules/core/graph/include/SushiAI/graph/lowering/`, define it in
   `modules/core/graph/source/lowering/`, and add one row to the domain table, raising its
   `std::array` size.
   The row records its SushiBLAS work on `context.recorder()` and returns a `Status`.

   ```cpp
   // modules/core/graph/source/lowering/elementwise.cpp
   Status lower_div(const LoweringContext& context)
   {
       return record_binary(context, &SushiBLAS::ElementwiseOps::div);
   }
   ```

   `graph/lowering/all.hpp` fails the build when an identity has no row, two rows, or a row
   and a place in `K_UNLOWERED`. `K_UNLOWERED` lists the three operations recorded beside
   the graph (the two optimizer updates and the batch-statistics update), so a graph
   operation takes a row there, never a place.

5. **Test.** Add a unit test of the shape rule to the test file of the header the rule lives
   in, and one `Integration_` case per row that runs the operation through a compiled step.
   `tests/core/graph/test_lowering/test_host_values.cpp` shows the shape: twelve forward cases
   in FLOAT32 and, for two gradient rows, a FLOAT64 case against the closed form. List a new
   test file in the `SOURCES` of its folder's `CMakeLists.txt`. Build and run the change as §3
   says, and add the CHANGELOG line as §5 says.

---

## 9. Reporting bugs and requesting features

Use the GitHub issue templates. A good bug report includes the platform and
toolchain, the build type, the exact `sa` command, and the smallest
reproducer you can manage. For anything that looks like a **security** issue,
do **not** open a public issue — follow [`SECURITY.md`](reference/SECURITY.md).

---

## 10. Questions

Open a GitHub Discussion, join the community Discord (see
[`SECURITY.md`](reference/SECURITY.md) for the current invite), or email
**mustafagarip@sushisystems.io**.

Please be respectful in all project spaces — see
[`CODE_OF_CONDUCT.md`](reference/CODE_OF_CONDUCT.md).
