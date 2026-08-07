# Config-driven training implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Spec:** [`../specs/2026-08-07-config-driven-training-design.md`](../specs/2026-08-07-config-driven-training-design.md)
(design, approved 2026-08-07). Section references below (`§4`, `§7`, …) are to
that document.

**Goal:** Move a model's architecture out of the C++ type system into JSON, and
add `sushiai train` / `sushiai inference` drivers that contain no layer, loss,
dataset or optimizer type name.

**Architecture:** Four constant `{name, factory}` tables replace the type names a
driver would otherwise spell. A runtime-polymorphic `NN::LayerStack` joins the
existing compile-time `NN::Sequential` through one additive `if constexpr` arm in
`for_each_parameter`, and costs nothing in the steady state because `forward()`
runs once at trace time. A versioned binary checkpoint (`SACP`) is split along
the host/device lane so its byte handling is testable with no device.

**Tech stack:** C++17, SushiBLAS/SushiRuntime siblings, nlohmann-json (link
`PRIVATE`, named in no public header), GTest, CTest, the `sa` CLI.

---

## Global constraints

Every task's requirements implicitly include this section.

- **C++17.** No `std::span` (use `SushiRuntime::span`), no `std::format` (use
  `SushiBLAS::Core::format`), no concepts (use `constexpr bool` variable
  templates over `std::void_t` detectors), no `consteval` (use `constexpr`
  called from a `static_assert`). See `docs/ARCHITECTURE.md §1`.
- **License header** on every new file, with `[filename]` replaced — the exact
  block in `docs/CONTRIBUTING.md §4`.
- **Naming:** `PascalCase` types/namespaces, `snake_case` functions/variables,
  trailing underscore on private members, Allman braces for multi-statement
  blocks, single-statement `if`/loop omits braces, 4-space indent.
- **Doxygen** `@brief` / `@param` / `@return` on every public API.
- **Every CMake target, function and macro is prefixed `sushiai_`.**
- **Builds go through `sa`, never ninja/cmake directly.** `sa build`,
  `sa test --suite unit|integration|regression|all`.
- **The GTest suite-name prefix is the only thing that assigns a CTest label.**
  A suite named without `Unit_` / `Integration_` / `Regression_` is invisible
  rather than failing.
- **Nothing self-registers.** `sushiai` is a static library, so a translation
  unit whose only job is to run a registration constructor resolves no symbol
  and is dropped by the linker — silently, and only in a downstream consumer's
  build. Every registry is a file-scope `constexpr` table.
- **An unknown key or an unknown `type` is an error, never an ignored one.**
- **Host lane files reach no SushiBLAS engine and no SYCL**; device lane files
  may. `config/`, `nn/ilayer.hpp`, `nn/layer_stack.hpp`, `nn/layer_registry.hpp`
  and `io/checkpoint_format.hpp` are host lane. `io/checkpoint.hpp` is device
  lane.
- **One logical change per commit**, header `type: short summary`, lowercase,
  imperative, no trailing period. Body is past-tense bullets.

### Recorded deviations from the spec

1. **No `vcpkg.json` manifest** (spec §6 asks for one). Adding a manifest beside
   the top-level `CMakeLists.txt` switches vcpkg into manifest mode, which drops
   the shared classic `installed/` tree off the prefix path — and `sa build`
   passes exactly that tree
   (`cli/sushiai/services/project.py:245`, `-DCMAKE_PREFIX_PATH={vcpkg}/installed/{triplet}`),
   which is where `hwloc`, `gtest` and `opencl` currently resolve from. The
   documented mechanism for a new C++ dependency is a top-level table in
   `cli/sushistack.deps.toml` (`docs/CONTRIBUTING.md §7`); no repository in the
   stack carries a manifest. `nlohmann_json` is already present in that tree.
   Decided with the maintainer on 2026-08-07.
2. **A `PRIVATE` link is not enough to keep the parser out of the export set**
   (spec §6 assumes it is). `sushiai` is a **STATIC** library, so a static
   archive carries none of its dependencies and CMake records even a `PRIVATE`
   link as `$<LINK_ONLY:...>` in the exported `INTERFACE_LINK_LIBRARIES`.
   Measured, not assumed: with `target_link_libraries(sushiai PRIVATE
   nlohmann_json::nlohmann_json)` the generated `SushiAITargets.cmake` carried
   `$<LINK_ONLY:nlohmann_json::nlohmann_json>`, and a consumer that ran
   `find_package(SushiAI)` without also finding `nlohmann_json` would fail to
   configure on a name it should never have heard.

   The fix is the export-set-safe shape `docs/ARCHITECTURE.md §6` already
   documents for the policy targets: pull the include directories and
   definitions on as **values** through `$<TARGET_PROPERTY:...>` rather than
   linking the target. Safe to drop rather than propagate because
   `nlohmann_json` is an INTERFACE (header-only) target — there is no archive
   for anyone to link. Guarded by a new check in `tests/package/CMakeLists.txt`
   beside the existing flag-hygiene ones.

3. **The parser is `nlohmann::ordered_json`, not `nlohmann::json`.** The
   default keeps an object's members in a `std::map` and hands them back
   alphabetically; `Config::Value` promises the order the file spelled, so that
   a "this object carries 'a', 'b', 'c'" diagnostic reads the way the file
   does. Caught by `Unit_Config.KeepsTheKeyOrderTheFileSpelled`.

4. **`Shape` gained a run-time extent constructor** (`explicit
   Shape(SushiRuntime::span<const int64_t>)`). The spec's checkpoint reader
   cannot exist without one: `Shape` was constructible only from an
   `initializer_list`, which a reader whose rank came from a file cannot
   produce short of a switch over every rank. Validation is identical to the
   existing constructor's.

5. **`StepSpec` keeps its `targets` field for this slice.** `ObjectiveLeaves`
   is what generalises it, and `TrainingGraph` carries both — but rewiring
   `Train::Trainer` to feed arbitrary declared leaves is a separate change with
   its own argument. The spec's actual commitment (that `apps/train.cpp` names
   no loss type) is met by the `ObjectiveRegistry`. Recorded as debt in Task 8
   and in `docs/ARCHITECTURE.md`, alongside `Dataset::num_classes()`.

---

## File structure

```
include/SushiAI/config/
  value.hpp             Config::Value, Config::Kind, parse_json, parse_json_file
  registry.hpp          Config::find_entry — the one lookup + diagnostic, shared by all four tables
  model_config.hpp      Config::LayerSpec, Config::ModelSpec, load_model
  train_config.hpp      Config::DataSpec, Config::TrainSpec, load_train, load_data
src/config/
  value.cpp             <- nlohmann::json appears ONLY here
  model_config.cpp
  train_config.cpp

include/SushiAI/nn/
  ilayer.hpp            NN::ILayer, NN::LayerAdapter<L>
  layer_stack.hpp       NN::LayerStack
  layer_registry.hpp    NN::LAYER_REGISTRY, NN::build_layer_stack
src/nn/
  layer_registry.cpp    make_linear, make_gelu

include/SushiAI/train/
  objective.hpp         Train::Metric, Train::ObjectiveLeaves, Train::IObjective,
                        Train::CrossEntropyObjective, Train::OBJECTIVE_REGISTRY
src/train/
  objective.cpp

include/SushiAI/io/
  checkpoint_format.hpp IO::TensorRecord, IO::CheckpointWriter, IO::CheckpointReader   (host lane)
  checkpoint.hpp        IO::save_checkpoint<M>, IO::load_checkpoint<M>                 (device lane)
src/io/
  checkpoint_format.cpp

apps/
  CMakeLists.txt
  train.cpp
  inference.cpp

examples/heavy_mlp/
  model.json  train.json  eval.json  README.md      (the three .cpp/.hpp files are deleted)

tests/functional/unit/
  test_checkpoint_format.cpp
  test_layer_stack.cpp
  test_config.cpp
tests/functional/integration/
  test_checkpoint.cpp
```

**Modified:**

- `include/SushiAI/nn/module.hpp` — `NN::ParameterVisitor`,
  `Detail::HasParameterWalk`, one new `if constexpr` arm in `for_each_parameter`.
- `include/SushiAI/data/dataset.hpp` + `src/data/dataset.cpp` — `Data::Rings`,
  `Data::RingsOptions`, `Data::DATASET_REGISTRY`, `Data::make_dataset`.
- `include/SushiAI/optim/optimizer.hpp` + `src/optim/optimizer.cpp` —
  `Optim::OPTIMIZER_REGISTRY`, `Optim::make_optimizer`.
- `include/SushiAI/train/trainer.hpp` — `build_graph`, `build_classifier`
  becomes a wrapper, `TrainingGraph::leaves`.
- `include/SushiAI/SushiAI.h` — the new public headers.
- `src/CMakeLists.txt` — new sources, `find_package(nlohmann_json)`, PRIVATE link.
- `CMakeLists.txt` — `add_subdirectory(apps)`.
- `tests/CMakeLists.txt` — three new unit sources, one new integration source,
  two app smoke tests.
- `cli/sushistack.deps.toml`, `cli/sushiai/cli.py`, `docs/CLI_GUIDE.md`,
  `docs/ARCHITECTURE.md`, `docs/README.md`, `docs/CHANGELOG.md`, `.gitignore`,
  `examples/CMakeLists.txt`.

---

## Task 0: Preserve the existing example, then branch

**Files:**
- Commit: `examples/heavy_mlp/{model.hpp,train.cpp,infer.cpp}`, `examples/CMakeLists.txt`

`examples/heavy_mlp/` is untracked working code that this plan deletes (§9). It
is committed first so the deletion is a reviewable diff and so the `Rings`
dataset has a source to be moved from rather than retyped.

- [ ] **Step 1: Confirm the baseline is green**

```powershell
sa build
sa test --suite all
```

Expected: build succeeds, every test passes. Record the test count — the
`docs/ARCHITECTURE.md` "153 tests" figure is stale and Task 12 corrects it.

- [ ] **Step 2: Commit the example as it stands**

```bash
git add examples/heavy_mlp examples/CMakeLists.txt
git commit -m "feat(examples): add a heavy MLP trained on a nonlinear rings problem"
```

Body bullets: added a four-hidden-layer GELU MLP over a concentric-rings
problem no linear classifier can separate; added a second binary that loads the
checkpoint and evaluates with no loss and no backward pass; wired both into
`examples/CMakeLists.txt`.

- [ ] **Step 3: Ignore the checkpoint artefact**

Append to `.gitignore`:

```gitignore
# Checkpoints written by the examples and by `sushiai train`.
*.ckpt
```

- [ ] **Step 4: Branch**

```bash
git add .gitignore
git commit -m "chore: ignore checkpoint artefacts"
git switch -c feature/ai-config-driven-training
```

---

## Task 1: The checkpoint file format (host lane)

**Files:**
- Create: `include/SushiAI/io/checkpoint_format.hpp`, `src/io/checkpoint_format.cpp`
- Test: `tests/functional/unit/test_checkpoint_format.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`, `include/SushiAI/SushiAI.h`

**Interfaces:**
- Consumes: `SushiAI::Shape` (`core/shape.hpp`), `SushiAI::DType` and
  `SushiAI::dtype_size` (`core/common.hpp`), `SA_THROW_IF`.
- Produces: `IO::TensorRecord`, `IO::CheckpointWriter`, `IO::CheckpointReader`,
  and the four bound constants. Task 9 is its only in-tree consumer.

The on-disk layout, little-endian, stated rather than assumed (§7):

```
magic     u32       0x50434153, which is the bytes "SACP"
version   u32       1
count     u32       tensor count                     (<= 4096)
--- count times ---
name_len  u32                                        (<= 256)
name      u8 * name_len
dtype     u32       SushiAI::DType
rank      u32                                        (<= Shape::MAX_RANK == 6)
dims      i64 * rank                                 (each > 0)
payload   u8 * (product(dims) * dtype_size(dtype))
```

The reader **never allocates from a file-declared size** (§7's third defect).
`read_payload` copies straight into a destination the caller already owns, whose
byte count the caller has already checked against the record.

- [ ] **Step 1: Write the failing test**

`tests/functional/unit/test_checkpoint_format.cpp`:

```cpp
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/io/checkpoint_format.hpp>
#include "../../common/test_throws.hpp"

namespace
{
    using SushiAI::IO::CheckpointReader;
    using SushiAI::IO::CheckpointWriter;
    using SushiAI::IO::TensorRecord;

    /** @brief A two-tensor checkpoint, as bytes. */
    std::string sample_stream()
    {
        std::ostringstream out(std::ios::binary);
        CheckpointWriter writer(out, 2);

        const std::vector<float> weight{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        const std::vector<float> bias{7.0f, 8.0f};

        writer.write(TensorRecord{"0.weight", SushiAI::DType::FLOAT32, SushiAI::Shape{2, 3}},
                     weight.data());
        writer.write(TensorRecord{"0.bias", SushiAI::DType::FLOAT32, SushiAI::Shape{2}},
                     bias.data());
        writer.finish();
        return out.str();
    }
}

TEST(Unit_CheckpointFormat, RoundTripsNamesShapesAndPayloads)
{
    std::istringstream in(sample_stream(), std::ios::binary);
    CheckpointReader reader(in);

    ASSERT_EQ(reader.count(), 2u);

    const TensorRecord weight = reader.next();
    EXPECT_EQ(weight.name, "0.weight");
    EXPECT_EQ(weight.dtype, SushiAI::DType::FLOAT32);
    EXPECT_EQ(weight.shape, (SushiAI::Shape{2, 3}));

    std::vector<float> weight_payload(6, 0.0f);
    reader.read_payload(weight, weight_payload.data());
    EXPECT_EQ(weight_payload, (std::vector<float>{1, 2, 3, 4, 5, 6}));

    const TensorRecord bias = reader.next();
    EXPECT_EQ(bias.name, "0.bias");
    EXPECT_EQ(bias.shape, (SushiAI::Shape{2}));

    std::vector<float> bias_payload(2, 0.0f);
    reader.read_payload(bias, bias_payload.data());
    EXPECT_EQ(bias_payload, (std::vector<float>{7, 8}));
}
```

- [ ] **Step 2: Run it and watch it fail**

```powershell
sa build
```

Expected: FAIL to compile — `SushiAI/io/checkpoint_format.hpp` does not exist.

- [ ] **Step 3: Write the header**

`include/SushiAI/io/checkpoint_format.hpp`, after the license block:

```cpp
#pragma once

/**
 * @file checkpoint_format.hpp
 * @brief The checkpoint byte stream, and nothing else.
 *
 * Host lane: no SushiBLAS engine, no SYCL, no tensors. It reads and writes a
 * stream of @ref TensorRecord headers and opaque payload bytes, which is what
 * lets the whole format be tested against a @c std::stringstream with no
 * device present. @c io/checkpoint.hpp is the other half: it walks a module's
 * parameters and binds records to the pool's tensors.
 *
 * @section allocation Why the reader allocates nothing
 *
 * A reader that sized a buffer from a file-declared element count would let a
 * corrupt file ask for 16 GB. There is no intermediate buffer here at all:
 * @ref CheckpointReader::next reads a header, the caller checks it against the
 * destination it already owns, and @ref CheckpointReader::read_payload copies
 * into that destination. The file-driven allocation is absent by construction
 * rather than guarded against.
 *
 * @section endianness Little-endian, stated
 *
 * Every scalar is encoded and decoded byte by byte, so the format does not
 * inherit the host's byte order by accident and a big-endian host reads the
 * same file.
 */

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief The first four bytes of every checkpoint: "SACP". */
        inline constexpr std::uint32_t CHECKPOINT_MAGIC = 0x50434153u;

        /** @brief The format version; a mismatch is an error, not a misread. */
        inline constexpr std::uint32_t CHECKPOINT_VERSION = 1u;

        /** @brief The largest tensor count a file may declare. */
        inline constexpr std::uint32_t MAX_TENSOR_COUNT = 4096u;

        /** @brief The longest parameter name a file may declare, in bytes. */
        inline constexpr std::uint32_t MAX_NAME_LENGTH = 256u;

        /**
         * @struct TensorRecord
         * @brief One tensor's identity in a checkpoint: what it is called and what shape it is.
         */
        struct TensorRecord
        {
            /** @brief The dotted parameter name, e.g. "0.weight". */
            std::string name;

            /** @brief The element type. */
            DType dtype = DType::FLOAT32;

            /** @brief The extents. */
            Shape shape{};

            /** @brief How many bytes this record's payload occupies. */
            std::size_t payload_bytes() const noexcept;
        };

        /**
         * @class CheckpointWriter
         * @brief Writes the header and then @p count records, in order.
         */
        class CheckpointWriter
        {
            public:
                /**
                 * @brief Writes the file header immediately.
                 * @param out Where the bytes go; must outlive the writer.
                 * @param count How many records will follow.
                 * @throws Error If @p count exceeds @ref MAX_TENSOR_COUNT, or the write fails.
                 */
                CheckpointWriter(std::ostream& out, std::uint32_t count);

                /**
                 * @brief Writes one record's header and its payload.
                 * @param record What the tensor is called and what shape it is.
                 * @param payload @c record.payload_bytes() bytes to copy out.
                 * @throws Error If @p record is malformed, more records are
                 *         written than declared, or the write fails.
                 */
                void write(const TensorRecord& record, const void* payload);

                /**
                 * @brief Checks that exactly the declared number of records were written.
                 * @throws Error If fewer were written, naming both counts.
                 */
                void finish() const;

            private:
                std::ostream* out_;
                std::uint32_t declared_ = 0;
                std::uint32_t written_ = 0;
        };

        /**
         * @class CheckpointReader
         * @brief Reads the header, then hands out records one at a time.
         */
        class CheckpointReader
        {
            public:
                /**
                 * @brief Reads and validates the file header.
                 * @param in Where the bytes come from; must outlive the reader.
                 * @throws Error If the magic, the version or the count is wrong,
                 *         or the header is truncated.
                 */
                explicit CheckpointReader(std::istream& in);

                /** @brief How many records the file declares. */
                std::uint32_t count() const noexcept { return count_; }

                /** @brief How many records have been read so far. */
                std::uint32_t read_count() const noexcept { return read_; }

                /**
                 * @brief Reads the next record's header, leaving its payload unread.
                 * @return The record.
                 * @throws Error If no records remain, a bound is exceeded, a
                 *         dim is not positive, or the header is truncated.
                 */
                TensorRecord next();

                /**
                 * @brief Reads the payload of the record @ref next just returned.
                 * @param record The record @ref next returned.
                 * @param destination @c record.payload_bytes() writable bytes.
                 * @throws Error If the stream ends early.
                 */
                void read_payload(const TensorRecord& record, void* destination);

            private:
                std::istream* in_;
                std::uint32_t count_ = 0;
                std::uint32_t read_ = 0;
        };

    } // namespace IO
} // namespace SushiAI
```

- [ ] **Step 4: Write the implementation**

`src/io/checkpoint_format.cpp`. Private helpers first, then the two classes:

```cpp
namespace
{
    void put_u32(std::ostream& out, std::uint32_t value)
    {
        char bytes[4];
        for (int i = 0; i < 4; ++i) bytes[i] = static_cast<char>((value >> (8 * i)) & 0xFFu);
        out.write(bytes, 4);
    }

    void put_i64(std::ostream& out, std::int64_t value)
    {
        const std::uint64_t bits = static_cast<std::uint64_t>(value);
        char bytes[8];
        for (int i = 0; i < 8; ++i) bytes[i] = static_cast<char>((bits >> (8 * i)) & 0xFFu);
        out.write(bytes, 8);
    }

    std::uint32_t get_u32(std::istream& in, std::string_view what)
    {
        char bytes[4];
        in.read(bytes, 4);
        SA_THROW_IF(!in, "the checkpoint ended while reading {}", what);
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i)
            value |= static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
        return value;
    }

    std::int64_t get_i64(std::istream& in, std::string_view what);   // same shape, 8 bytes
}
```

`TensorRecord::payload_bytes()` is
`static_cast<std::size_t>(shape.num_elements()) * dtype_size(dtype)`.

`CheckpointWriter`'s constructor throws when `count > MAX_TENSOR_COUNT`, then
writes magic, version, count. `write()` throws when `written_ == declared_`
("this checkpoint declared {} records and a {}th was written"), when
`record.name.size() > MAX_NAME_LENGTH`, when `record.shape.rank() > Shape::MAX_RANK`,
and when any dim is not positive; then it writes `name_len`, the name bytes,
`dtype` as `u32`, `rank`, the dims, and finally `payload_bytes()` raw bytes.
Every write is followed by a stream check that throws naming the record.

`CheckpointReader`'s constructor reads magic (throws
`"not a SushiAI checkpoint: expected magic 'SACP', got 0x{:08X}"`), version
(throws `"this checkpoint is version {}, this build reads version {}"`) and
count (throws when over `MAX_TENSOR_COUNT`). `next()` throws when
`read_ == count_`, reads and bounds-checks `name_len`, reads the name, reads
`dtype` (throws when it is not a value of `DType`), reads and bounds-checks
`rank`, reads the dims (throws when a dim is `<= 0`, naming the record and the
dim), increments `read_`. `read_payload()` reads exactly
`record.payload_bytes()` bytes into `destination` and throws when the stream
ends early.

- [ ] **Step 5: Wire the build**

`src/CMakeLists.txt` — add `io/checkpoint_format.cpp` to `SUSHIAI_SOURCES`,
after the `graph/` block and before `autograd/`, under a comment naming the
lane. `tests/CMakeLists.txt` — add
`functional/unit/test_checkpoint_format.cpp` to the unit block.
`include/SushiAI/SushiAI.h` — add `#include <SushiAI/io/checkpoint_format.hpp>`.

- [ ] **Step 6: Run the test and watch it pass**

```powershell
sa build
sa test --suite unit
```

Expected: `Unit_CheckpointFormat.RoundTripsNamesShapesAndPayloads` PASSES.

- [ ] **Step 7: Add the malformed-file tests**

Same file, each building a byte string from `sample_stream()` and corrupting it.
Use `SA_EXPECT_THROWS_WITH` from `tests/common/test_throws.hpp` (check its exact
spelling first) so each assertion pins the diagnostic, not just the throw:

| Test name | Construction | Message must name |
|---|---|---|
| `RejectsAForeignMagic` | first 4 bytes → `"JUNK"` | `SACP` |
| `RejectsAFutureVersion` | bytes 4–7 → `2` | `version 2` |
| `RejectsACountOverTheBound` | bytes 8–11 → `4097` | `4096` |
| `RejectsATruncatedHeader` | `substr(0, 6)` | `ended while reading` |
| `RejectsATruncatedPayload` | drop the last 4 bytes, read both records | `ended` |
| `RejectsANameOverTheBound` | patch the first `name_len` to `257` | `256` |
| `RejectsARankOverTheBound` | patch the first `rank` to `7` | `6` |
| `RejectsANonPositiveDim` | patch the first dim to `0` | the record's name |
| `RejectsMoreRecordsThanDeclared` | `writer.write()` three times against `count = 2` | `declared` |
| `RejectsFewerRecordsThanDeclared` | `writer.write()` once, then `finish()` | `1` and `2` |

- [ ] **Step 8: Run the whole unit tier**

```powershell
sa build
sa test --suite unit
```

Expected: all ten new cases pass, nothing else regressed.

- [ ] **Step 9: Commit**

```bash
git add include/SushiAI/io/checkpoint_format.hpp src/io/checkpoint_format.cpp \
        tests/functional/unit/test_checkpoint_format.cpp \
        src/CMakeLists.txt tests/CMakeLists.txt include/SushiAI/SushiAI.h
git commit -m "feat(io): add the SACP checkpoint byte format"
```

---

## Task 2: The third arm of the parameter walk

**Files:**
- Modify: `include/SushiAI/nn/module.hpp:197-280` (the `Detail` namespace),
  `include/SushiAI/nn/module.hpp:332-349` (`for_each_parameter`)
- Test: `tests/functional/unit/test_module.cpp`

**Interfaces:**
- Produces: `NN::ParameterVisitor` (a non-owning callable reference) and the
  `walk_parameters(std::string_view, const ParameterVisitor&)` contract. Task 3
  and Task 9 both depend on it.

`for_each_parameter` dispatches on two cases today. Neither can enumerate a
runtime list: `fields()` is static and returns member pointers of one concrete
type. One additive arm fixes it, and no type that compiles today takes it (§3).

A virtual cannot be a template, so the visitor has to be type-erased. It is a
non-owning function reference rather than a `std::function` because the walk is
called from the trace path and from checkpoint I/O, and neither should allocate
to describe a callback it already holds.

- [ ] **Step 1: Write the failing test**

Append to `tests/functional/unit/test_module.cpp`:

```cpp
namespace
{
    /** @brief A leaf whose parameters are enumerated at run time, not by fields(). */
    class WalkableStub
    {
        public:
            WalkableStub()
            {
                weight_.declare(builder_, SushiAI::Shape{2, 3}, SushiAI::NN::ParameterRole::WEIGHT,
                                3, 2, "w");
                bias_.declare(builder_, SushiAI::Shape{2}, SushiAI::NN::ParameterRole::BIAS,
                              0, 0, "b");
            }

            static constexpr auto fields() { return std::tuple<>{}; }

            SushiAI::Graph::ValueId forward(const SushiAI::Graph::GraphBuilder&,
                                            SushiAI::Graph::ValueId x) { return x; }

            void walk_parameters(std::string_view prefix,
                                 const SushiAI::NN::ParameterVisitor& visitor)
            {
                visitor(SushiAI::NN::Detail::join(prefix, "weight"), weight_);
                visitor(SushiAI::NN::Detail::join(prefix, "bias"), bias_);
            }

        private:
            SushiAI::Graph::Graph graph_{};
            SushiAI::Graph::GraphBuilder builder_{graph_};
            SushiAI::NN::Parameter weight_{};
            SushiAI::NN::Parameter bias_{};
    };
}

TEST(Unit_Module, AWalkableLeafIsEnumeratedByItsOwnWalk)
{
    WalkableStub stub;
    std::vector<std::string> names;
    SushiAI::NN::for_each_parameter(stub, [&names](const std::string& name, SushiAI::NN::Parameter&)
    {
        names.push_back(name);
    });

    EXPECT_EQ(names, (std::vector<std::string>{"weight", "bias"}));
    EXPECT_EQ(SushiAI::NN::parameter_count(stub), 2u);
    EXPECT_EQ(SushiAI::NN::parameter_elements(stub), 8);
}
```

- [ ] **Step 2: Run it and watch it fail**

```powershell
sa build
```

Expected: FAIL to compile — `NN::ParameterVisitor` does not exist.

- [ ] **Step 3: Add `ParameterVisitor`**

In `module.hpp`, immediately after `ParameterField` / `field()` and before
`namespace Detail`:

```cpp
        /**
         * @class ParameterVisitor
         * @brief A non-owning reference to a parameter callback, for a virtual walk.
         *
         * @c for_each_parameter takes its visitor as a template parameter, which
         * a virtual function cannot. A runtime layer list therefore needs the
         * callback type-erased — and erased without allocating, because the walk
         * runs on the trace path and inside checkpoint I/O.
         *
         * It refers to the callable rather than owning it, exactly like a
         * @c string_view refers to characters: the referent must outlive the
         * visitor. Every construction site in this library passes a named local,
         * so there is no temporary to dangle.
         */
        class ParameterVisitor
        {
            public:
                /**
                 * @brief Wraps @p callable, which must outlive this visitor.
                 * @tparam Fn Anything callable as (const std::string&, Parameter&).
                 * @param callable The callback; stored by address.
                 */
                template<typename Fn,
                         typename = std::enable_if_t<
                             !std::is_same<std::decay_t<Fn>, ParameterVisitor>::value>>
                ParameterVisitor(Fn& callable) noexcept
                    : target_(static_cast<void*>(std::addressof(callable))),
                      invoke_([](void* target, const std::string& name, Parameter& parameter)
                              {
                                  (*static_cast<Fn*>(target))(name, parameter);
                              })
                {
                }

                /**
                 * @brief Applies the wrapped callback.
                 * @param name The parameter's dotted name.
                 * @param parameter The parameter itself.
                 */
                void operator()(const std::string& name, Parameter& parameter) const
                {
                    invoke_(target_, name, parameter);
                }

            private:
                void* target_ = nullptr;
                void (*invoke_)(void*, const std::string&, Parameter&) = nullptr;
        };
```

`<memory>` joins the include list for `std::addressof`.

- [ ] **Step 4: Add the detector**

Inside `namespace Detail`, beside `IsModuleContainer`:

```cpp
            /**
             * @struct HasParameterWalk
             * @brief Detects a layer that enumerates its own parameters at run time.
             *
             * A @c LayerStack holds @c std::unique_ptr<ILayer>, so neither
             * @c fields() (static, one concrete type) nor @c visit_children
             * (which would hand every child out as the same @c ILayer&) can
             * name what it owns. Such a type publishes @c walk_parameters
             * instead and this detector routes the walk to it.
             */
            template<typename T, typename = void>
            struct HasParameterWalk : std::false_type {};

            /** @brief The satisfied case: T offers walk_parameters(prefix, visitor). */
            template<typename T>
            struct HasParameterWalk<T, std::void_t<
                decltype(std::declval<T&>().walk_parameters(
                    std::declval<std::string_view>(),
                    std::declval<const ParameterVisitor&>()))>>
                : std::true_type {};
```

- [ ] **Step 5: Add the arm**

`for_each_parameter`'s body becomes three arms, the new one **first** so a type
offering both a walk and a `fields()` list is enumerated by the walk it wrote:

```cpp
            if constexpr (Detail::HasParameterWalk<M>::value)
            {
                // ParameterVisitor refers to its callable, so the erased visitor
                // needs a named lvalue to point at rather than the forwarding
                // reference's own name.
                auto& callable = visitor;
                const ParameterVisitor erased(callable);
                module.walk_parameters(prefix, erased);
            }
            else if constexpr (ModuleContainer<M>)
            {
                /* unchanged */
            }
            else
            {
                /* unchanged */
            }
```

Extend the `@brief` above it to name the third case, and add to the file
comment's `@section registration` a sentence saying the walk has a third arm and
why.

- [ ] **Step 6: Run the test and watch it pass**

```powershell
sa build
sa test --suite unit
```

Expected: `Unit_Module.AWalkableLeafIsEnumeratedByItsOwnWalk` PASSES and every
existing `Unit_Module` case still passes — the arm is strictly additive.

- [ ] **Step 7: Prove nothing existing takes the new arm**

Add to `layers.hpp`'s `static_assert` block:

```cpp
        static_assert(!Detail::HasParameterWalk<Linear>::value,
                      "A compile-time leaf must not take the runtime walk arm");
        static_assert(!Detail::HasParameterWalk<Sequential<Linear, ReLU>>::value,
                      "Sequential must stay a container, not a runtime walk");
```

- [ ] **Step 8: Run the whole suite**

```powershell
sa build
sa test --suite all
```

Expected: everything green.

- [ ] **Step 9: Commit**

```bash
git add include/SushiAI/nn/module.hpp include/SushiAI/nn/layers.hpp \
        tests/functional/unit/test_module.cpp
git commit -m "feat(nn): route the parameter walk through a runtime layer's own enumeration"
```

---

## Task 3: `ILayer` and `LayerStack`

**Files:**
- Create: `include/SushiAI/nn/ilayer.hpp`, `include/SushiAI/nn/layer_stack.hpp`,
  `src/nn/layer_stack.cpp`
- Test: `tests/functional/unit/test_layer_stack.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`, `include/SushiAI/SushiAI.h`

**Interfaces:**
- Consumes: `NN::ParameterVisitor`, `NN::for_each_parameter`, `NN::Module`.
- Produces: `NN::ILayer` (pure virtual), `NN::LayerAdapter<L>`,
  `NN::LayerStack`. Task 5 constructs `LayerAdapter`s; Tasks 9 and 10 walk a
  `LayerStack`.

`LayerAdapter<L>` is what makes naming parity structural rather than
conventional (§3): it delegates to the very `for_each_parameter` walk
`Sequential`'s children go through, so `LayerStack` cannot drift from
`Sequential` without one of them failing its own test.

- [ ] **Step 1: Write the failing test**

`tests/functional/unit/test_layer_stack.cpp`:

```cpp
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/ilayer.hpp>
#include <SushiAI/nn/layer_stack.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>

namespace
{
    using namespace SushiAI;

    /** @brief Traces @p model and returns its parameter names, in walk order. */
    template<typename M>
    std::vector<std::string> traced_names(M& model, int64_t batch, int64_t features)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueId x = builder.input(Shape{batch, features}, "inputs");
        model.forward(builder, x);

        std::vector<std::string> names;
        NN::for_each_parameter(model, [&names](const std::string& name, NN::Parameter&)
        {
            names.push_back(name);
        });
        return names;
    }
}

TEST(Unit_LayerStack, NamesItsParametersExactlyAsSequentialDoes)
{
    NN::Sequential compile_time{NN::Linear{6, 8}, NN::GELU{}, NN::Linear{8, 4}};

    NN::LayerStack run_time;
    run_time.push(std::make_unique<NN::LayerAdapter<NN::Linear>>(NN::Linear{6, 8}, 8));
    run_time.push(std::make_unique<NN::LayerAdapter<NN::GELU>>(NN::GELU{}, 8));
    run_time.push(std::make_unique<NN::LayerAdapter<NN::Linear>>(NN::Linear{8, 4}, 4));

    EXPECT_EQ(traced_names(compile_time, 4, 6), traced_names(run_time, 4, 6));
    EXPECT_EQ(traced_names(run_time, 4, 6),
              (std::vector<std::string>{"0.weight", "0.bias", "2.weight", "2.bias"}));
}

TEST(Unit_LayerStack, ReportsTheLastLayersOutputWidth)
{
    NN::LayerStack stack;
    stack.push(std::make_unique<NN::LayerAdapter<NN::Linear>>(NN::Linear{6, 8}, 8));
    stack.push(std::make_unique<NN::LayerAdapter<NN::GELU>>(NN::GELU{}, 8));
    stack.push(std::make_unique<NN::LayerAdapter<NN::Linear>>(NN::Linear{8, 4}, 4));

    EXPECT_EQ(stack.size(), 3u);
    EXPECT_EQ(stack.output_features(), 4);
}

TEST(Unit_LayerStack, RefusesToTraceWhenEmpty)
{
    NN::LayerStack stack;
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{4, 6}, "inputs");
    EXPECT_THROW(stack.forward(builder, x), SushiAI::Error);
}
```

(Check the exact exception type name in `core/common.hpp` before writing
`SushiAI::Error`.)

- [ ] **Step 2: Run it and watch it fail**

```powershell
sa build
```

Expected: FAIL — `SushiAI/nn/ilayer.hpp` does not exist.

- [ ] **Step 3: Write `ilayer.hpp`**

```cpp
#pragma once

/**
 * @file ilayer.hpp
 * @brief A layer whose identity is decided at run time.
 *
 * @section cost Why a virtual call costs nothing here
 *
 * @c NN::Sequential decides its layer list at compile time: it holds a
 * @c std::tuple<Ls...>, its @c fields() is @c static @c constexpr, and the
 * parameter walk is a template walk. A JSON file decides the list at run time.
 * The two cannot be reconciled, so a second, runtime-polymorphic path is
 * required.
 *
 * In an eager framework that would mean a virtual call per layer per step.
 * Here it does not. @c forward() runs **once**, at trace time, to emit IR; the
 * graph is then compiled once and replayed (docs/ARCHITECTURE.md §1). A
 * @c std::vector<std::unique_ptr<ILayer>> therefore costs exactly nothing in
 * the steady state — the graph-native architecture is what pays for this
 * feature.
 *
 * Host lane: no SushiBLAS, no SYCL.
 */

#include <cstdint>
#include <string_view>
#include <utility>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @class ILayer
         * @brief One layer of a runtime-configured model.
         */
        class ILayer
        {
            public:
                virtual ~ILayer() = default;

                /**
                 * @brief Traces the layer, declaring its parameters on first call.
                 * @param builder Where the nodes go.
                 * @param x The input value.
                 * @return The result's id.
                 */
                virtual Graph::ValueId forward(const Graph::GraphBuilder& builder,
                                               Graph::ValueId x) = 0;

                /**
                 * @brief Applies @p visitor to every declared parameter this layer owns.
                 * @param prefix The name prefix this layer's parameters hang under.
                 * @param visitor Called once per declared parameter.
                 */
                virtual void walk_parameters(std::string_view prefix,
                                             const ParameterVisitor& visitor) = 0;

                /** @brief How many features a sample carries after this layer. */
                virtual int64_t output_features() const noexcept = 0;
        };

        /**
         * @class LayerAdapter
         * @brief Presents any compile-time @ref Module as an @ref ILayer.
         *
         * The parameter walk delegates to @c for_each_parameter — the very walk
         * @c Sequential's children go through — so a @c LayerStack's names are
         * @c Sequential's names by construction rather than by convention. That
         * equivalence is what makes a checkpoint written by one loadable by the
         * other, and it is proved by the round-trip test rather than assumed.
         *
         * @tparam L The wrapped layer type; must satisfy @ref Module.
         */
        template<typename L>
        class LayerAdapter final : public ILayer
        {
            public:
                static_assert(Module<L>,
                              "LayerAdapter's argument must satisfy the Module contract");

                /**
                 * @brief Wraps @p layer.
                 * @param layer The layer; moved in.
                 * @param output_features How many features a sample carries after it.
                 */
                LayerAdapter(L layer, int64_t output_features)
                    : layer_(std::move(layer)), output_features_(output_features) {}

                Graph::ValueId forward(const Graph::GraphBuilder& builder,
                                       Graph::ValueId x) override
                {
                    return layer_.forward(builder, x);
                }

                void walk_parameters(std::string_view prefix,
                                     const ParameterVisitor& visitor) override
                {
                    for_each_parameter(layer_, prefix, visitor);
                }

                int64_t output_features() const noexcept override { return output_features_; }

                /** @brief The wrapped layer, for a test that wants to inspect it. */
                L& layer() noexcept { return layer_; }

            private:
                L layer_;
                int64_t output_features_ = 0;
        };

    } // namespace NN
} // namespace SushiAI
```

Note: `for_each_parameter(layer_, prefix, visitor)` takes the leaf arm for a
`Linear`, so there is no recursion back into `LayerAdapter`.

- [ ] **Step 4: Write `layer_stack.hpp`**

```cpp
        /**
         * @class LayerStack
         * @brief Applies a runtime list of layers in order, naming them by index.
         *
         * Satisfies @ref Module, so everything written as
         * @c template<typename @c M> over the parameter walk — @c NN::initialize,
         * @c Train::build_graph, @c IO::save_checkpoint — consumes it unchanged.
         *
         * Names its children by index **including the gaps a parameterless
         * activation leaves**, exactly as @c Sequential does, so a three-layer
         * MLP is "0.weight", "0.bias", "2.weight". That is not cosmetic: it is
         * what lets a checkpoint written by a hand-written @c Sequential load
         * into a @c LayerStack built from the equivalent model.json, and the
         * reverse.
         *
         * Non-copyable, because it owns its layers through @c unique_ptr.
         */
        class LayerStack
        {
            public:
                LayerStack() = default;
                LayerStack(LayerStack&&) noexcept = default;
                LayerStack& operator=(LayerStack&&) noexcept = default;
                LayerStack(const LayerStack&) = delete;
                LayerStack& operator=(const LayerStack&) = delete;

                /**
                 * @brief Appends @p layer to the end of the stack.
                 * @param layer The layer; ownership is taken.
                 * @throws Error If @p layer is null.
                 */
                void push(std::unique_ptr<ILayer> layer);

                /** @brief No parameters of its own; the layers own them all. */
                static constexpr auto fields() { return std::tuple<>{}; }

                /**
                 * @brief Traces every layer in order.
                 * @param builder Where the nodes go.
                 * @param x The input value.
                 * @return The last layer's result.
                 * @throws Error If the stack is empty.
                 */
                Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x);

                /**
                 * @brief Applies @p visitor to every declared parameter, index-prefixed.
                 * @param prefix The name prefix; empty at the root.
                 * @param visitor Called once per declared parameter.
                 */
                void walk_parameters(std::string_view prefix, const ParameterVisitor& visitor);

                /** @brief How many layers the stack holds. */
                std::size_t size() const noexcept { return layers_.size(); }

                /**
                 * @brief How many features a sample carries after the last layer.
                 * @throws Error If the stack is empty.
                 */
                int64_t output_features() const;

            private:
                std::vector<std::unique_ptr<ILayer>> layers_;
        };

        static_assert(Module<LayerStack>, "LayerStack must satisfy the Module contract");
        static_assert(!ModuleContainer<LayerStack>,
                      "LayerStack is walked through walk_parameters, not visit_children");
```

`src/nn/layer_stack.cpp` holds the four out-of-line members.
`walk_parameters` is:

```cpp
        void LayerStack::walk_parameters(std::string_view prefix,
                                         const ParameterVisitor& visitor)
        {
            for (std::size_t i = 0; i < layers_.size(); ++i)
                layers_[i]->walk_parameters(Detail::join(prefix, std::to_string(i)), visitor);
        }
```

- [ ] **Step 5: Wire the build**

`src/CMakeLists.txt` — `nn/layer_stack.cpp` after `nn/init.cpp`.
`tests/CMakeLists.txt` — `functional/unit/test_layer_stack.cpp`.
`SushiAI.h` — both new headers.

- [ ] **Step 6: Run the tests and watch them pass**

```powershell
sa build
sa test --suite unit
```

Expected: all three `Unit_LayerStack` cases PASS.

- [ ] **Step 7: Commit**

```bash
git add include/SushiAI/nn/ilayer.hpp include/SushiAI/nn/layer_stack.hpp \
        src/nn/layer_stack.cpp tests/functional/unit/test_layer_stack.cpp \
        src/CMakeLists.txt tests/CMakeLists.txt include/SushiAI/SushiAI.h
git commit -m "feat(nn): add a runtime layer stack that names parameters as Sequential does"
```

---

## Task 4: `Config::Value` and the JSON parser

**Files:**
- Create: `include/SushiAI/config/value.hpp`, `src/config/value.cpp`
- Test: `tests/functional/unit/test_config.cpp`
- Modify: `cli/sushistack.deps.toml`, `src/CMakeLists.txt`,
  `tests/CMakeLists.txt`, `include/SushiAI/SushiAI.h`

**Interfaces:**
- Produces: `Config::Kind`, `Config::Value`, `Config::parse_json`,
  `Config::parse_json_file`. Tasks 5, 6, 7, 8 all take a `const Config::Value&`.

`nlohmann::json` appears **only** in `src/config/value.cpp`, so the dependency
links `PRIVATE`, `cmake/SushiAIConfig.cmake.in` needs no `find_dependency`, and
the out-of-tree consumer in `tests/package/` never sees it — the same
containment `SushiRuntime::span` and `SushiBLAS::Core::format` apply.

`Value` stores every alternative as a separate member rather than a
`std::variant`, because a variant over `std::vector<Value>` needs `Value`
complete at the point of declaration and `std::vector` of an incomplete type is
what C++17 actually guarantees. Object entries are two parallel vectors, which
keeps insertion order (so a diagnostic lists keys as the file spells them) and
needs no `std::pair` of an incomplete type. A config tree is a few dozen nodes,
so the unused members cost nothing worth a more delicate representation.

- [ ] **Step 1: Declare the dependency**

Append to `cli/sushistack.deps.toml`:

```toml
[nlohmann-json]
description   = "Header-only JSON parser. Used only by src/config/*.cpp to turn a config file into Config::Value; the name appears in no public header, so it links PRIVATE and no consumer inherits it."
required      = true
linux_apt     = ["nlohmann-json3-dev"]
windows_vcpkg = ["nlohmann-json"]
```

Update the fragment's header comment: it currently says "No port entries beyond
that: nothing new to declare here yet" — that is now false.

- [ ] **Step 2: Write the failing test**

`tests/functional/unit/test_config.cpp`:

```cpp
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include "../../common/test_throws.hpp"

using SushiAI::Config::Kind;
using SushiAI::Config::Value;

TEST(Unit_Config, ParsesEveryScalarKind)
{
    const Value root = SushiAI::Config::parse_json(
        R"({"a": 1, "b": 2.5, "c": "text", "d": true, "e": null, "f": [1, 2]})", "<test>");

    ASSERT_EQ(root.kind(), Kind::OBJECT);
    EXPECT_EQ(root.at("a", "<test>").as_int("a"), 1);
    EXPECT_DOUBLE_EQ(root.at("b", "<test>").as_real("b"), 2.5);
    EXPECT_EQ(root.at("c", "<test>").as_string("c"), "text");
    EXPECT_TRUE(root.at("d", "<test>").as_bool("d"));
    EXPECT_TRUE(root.at("e", "<test>").is_null());
    EXPECT_EQ(root.at("f", "<test>").as_array("f").size(), 2u);
}

TEST(Unit_Config, AnIntegerReadsAsAReal)
{
    const Value root = SushiAI::Config::parse_json(R"({"lr": 3})", "<test>");
    EXPECT_DOUBLE_EQ(root.at("lr", "<test>").as_real("lr"), 3.0);
}

TEST(Unit_Config, AWrongKindNamesBothTheKeyAndTheKind)
{
    const Value root = SushiAI::Config::parse_json(R"({"units": "many"})", "<test>");
    SA_EXPECT_THROWS_WITH(root.at("units", "<test>").as_int("units"), "units");
    SA_EXPECT_THROWS_WITH(root.at("units", "<test>").as_int("units"), "string");
}

TEST(Unit_Config, AMissingKeyNamesTheKeyAndTheOrigin)
{
    const Value root = SushiAI::Config::parse_json(R"({"a": 1})", "model.json");
    SA_EXPECT_THROWS_WITH(root.at("b", "model.json"), "model.json");
    SA_EXPECT_THROWS_WITH(root.at("b", "model.json"), "'b'");
}

TEST(Unit_Config, AnUnknownKeyIsAnErrorThatListsWhatIsAllowed)
{
    const Value root = SushiAI::Config::parse_json(R"({"type": "linear", "unitss": 256})",
                                                   "model.json");
    const std::string_view allowed[] = {"type", "units"};
    SA_EXPECT_THROWS_WITH(
        root.reject_unknown_keys("layer 0",
                                 SushiRuntime::span<const std::string_view>(allowed, 2)),
        "unitss");
    SA_EXPECT_THROWS_WITH(
        root.reject_unknown_keys("layer 0",
                                 SushiRuntime::span<const std::string_view>(allowed, 2)),
        "units");
}

TEST(Unit_Config, MalformedJsonNamesTheOrigin)
{
    SA_EXPECT_THROWS_WITH(SushiAI::Config::parse_json("{ not json", "train.json"),
                          "train.json");
}
```

- [ ] **Step 3: Run it and watch it fail**

```powershell
sa build
```

Expected: FAIL — `SushiAI/config/value.hpp` does not exist.

- [ ] **Step 4: Write `value.hpp`**

```cpp
namespace SushiAI
{
    namespace Config
    {
        /** @brief What a @ref Value holds. */
        enum class Kind : std::uint8_t { NUL, BOOL, INT, REAL, STRING, ARRAY, OBJECT };

        /** @brief A printable name for @p kind, for a diagnostic. */
        std::string_view kind_name(Kind kind) noexcept;

        class Value
        {
            public:
                Value() = default;                                    // NUL

                static Value boolean(bool value);
                static Value integer(std::int64_t value);
                static Value real(double value);
                static Value string(std::string value);
                static Value array(std::vector<Value> entries);
                static Value object(std::vector<std::string> keys, std::vector<Value> values);

                Kind kind() const noexcept { return kind_; }
                bool is_null() const noexcept { return kind_ == Kind::NUL; }

                /**
                 * @brief Reads this value as a boolean.
                 * @param context What this value is, for the diagnostic, e.g. "layer 0 'use_bias'".
                 * @return The boolean.
                 * @throws Error If this value is not a boolean; names @p context and the actual kind.
                 */
                bool as_bool(std::string_view context) const;

                /** @brief As @ref as_bool, for a whole number. */
                std::int64_t as_int(std::string_view context) const;

                /** @brief As @ref as_bool, for a real; a whole number is accepted and widened. */
                double as_real(std::string_view context) const;

                /** @brief As @ref as_bool, for a string. */
                const std::string& as_string(std::string_view context) const;

                /** @brief As @ref as_bool, for an array. */
                const std::vector<Value>& as_array(std::string_view context) const;

                /** @brief Whether this object holds @p key; false for any non-object. */
                bool has(std::string_view key) const noexcept;

                /**
                 * @brief This object's value for @p key.
                 * @param key The key to look up.
                 * @param context What this object is, for the diagnostic.
                 * @return The value.
                 * @throws Error If this is not an object, or @p key is absent;
                 *         the message names @p context, @p key, and every key
                 *         that IS present.
                 */
                const Value& at(std::string_view key, std::string_view context) const;

                /**
                 * @brief This object's value for @p key, or @p fallback when absent.
                 * @param key The key to look up.
                 * @param fallback Returned when @p key is absent.
                 * @return The value or @p fallback.
                 */
                const Value& at_or(std::string_view key, const Value& fallback) const noexcept;

                /** @brief This object's keys, in the order the file spelled them. */
                const std::vector<std::string>& keys() const noexcept { return keys_; }

                /**
                 * @brief Refuses any key of this object that is not in @p allowed.
                 *
                 * A typed "unitss": 256 must not train silently against a
                 * default. This is the silent-wrong-answer class the rest of
                 * the codebase is built to refuse.
                 *
                 * @param context What this object is, for the diagnostic.
                 * @param allowed Every key this object may carry.
                 * @throws Error Naming the first offending key and listing @p allowed.
                 */
                void reject_unknown_keys(std::string_view context,
                                         SushiRuntime::span<const std::string_view> allowed) const;

                /**
                 * @brief A copy of this object with @p key removed.
                 * @param key The key to drop; absent is not an error.
                 * @return The reduced object.
                 */
                Value without(std::string_view key) const;

            private:
                Kind kind_ = Kind::NUL;
                bool bool_ = false;
                std::int64_t int_ = 0;
                double real_ = 0.0;
                std::string string_;
                std::vector<Value> array_;
                std::vector<std::string> keys_;
                std::vector<Value> values_;
        };

        /**
         * @brief Parses @p text as JSON.
         * @param text The document.
         * @param origin What to call it in a diagnostic, usually a path.
         * @return The parsed tree.
         * @throws Error If @p text is not valid JSON; names @p origin.
         */
        Value parse_json(std::string_view text, std::string_view origin);

        /**
         * @brief Reads and parses @p path.
         * @param path The file to read.
         * @return The parsed tree.
         * @throws Error If the file cannot be read or is not valid JSON.
         */
        Value parse_json_file(const std::filesystem::path& path);
    }
}
```

- [ ] **Step 5: Write `src/config/value.cpp`**

`#include <nlohmann/json.hpp>` — the only place in the repository. One recursive
`convert(const nlohmann::json&)` walks the parsed document into `Value`,
mapping `number_integer` / `number_unsigned` to `Kind::INT` and
`number_float` to `Kind::REAL`. `parse_json` wraps
`nlohmann::json::parse(..., nullptr, /*allow_exceptions*/ true)` in a
`try`/`catch (const nlohmann::json::exception&)` that rethrows through
`SA_THROW_IF` naming `origin` and the library's own message. `Value::without`
copies `keys_`/`values_` skipping one index.

- [ ] **Step 6: Wire the build**

`src/CMakeLists.txt`: add `config/value.cpp` to `SUSHIAI_SOURCES`, then after
the `target_link_libraries(sushiai PUBLIC ...)` block:

```cmake
# --- The config layer's JSON parser ---
# PRIVATE, and the name `nlohmann` appears in no public header: the parser is
# an implementation detail of src/config/*.cpp. That is what keeps
# cmake/SushiAIConfig.cmake.in free of a find_dependency() and keeps the
# out-of-tree consumer in tests/package/ from ever seeing it - the same
# containment SushiRuntime::span and SushiBLAS::Core::format apply to
# std::span and std::format.
find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(sushiai PRIVATE nlohmann_json::nlohmann_json)
```

Add `functional/unit/test_config.cpp` to `tests/CMakeLists.txt` and
`config/value.hpp` to `SushiAI.h`.

- [ ] **Step 7: Run the tests and watch them pass**

```powershell
sa build
sa test --suite unit
```

Expected: all six `Unit_Config` cases PASS.

- [ ] **Step 8: Prove the dependency did not leak**

```powershell
sa build --clean
```

Then configure and build `tests/package/` against an install prefix, per
`docs/CONTRIBUTING.md §3.3`. Expected: the out-of-tree consumer builds without
`nlohmann_json` on its prefix path.

- [ ] **Step 9: Commit**

```bash
git add include/SushiAI/config/value.hpp src/config/value.cpp \
        tests/functional/unit/test_config.cpp cli/sushistack.deps.toml \
        src/CMakeLists.txt tests/CMakeLists.txt include/SushiAI/SushiAI.h
git commit -m "feat(config): add a typed config tree behind a private JSON parser"
```

---

## Task 5: The registry lookup, and the layer registry

**Files:**
- Create: `include/SushiAI/config/registry.hpp`,
  `include/SushiAI/config/model_config.hpp`, `src/config/model_config.cpp`,
  `include/SushiAI/nn/layer_registry.hpp`, `src/nn/layer_registry.cpp`
- Test: `tests/functional/unit/test_config.cpp` (append)
- Modify: `src/CMakeLists.txt`, `include/SushiAI/SushiAI.h`

**Interfaces:**
- Consumes: `Config::Value`, `NN::ILayer`, `NN::LayerAdapter`, `NN::LayerStack`.
- Produces: `Config::find_entry`, `Config::LayerSpec`, `Config::ModelSpec`,
  `Config::load_model`, `NN::LayerFactory`, `NN::LayerEntry`,
  `NN::LAYER_REGISTRY`, `NN::build_layer_stack`.

`"type"` is the discriminator key in all four registries — one convention, not
four — so the lookup and its diagnostic are written once here and reused by
Tasks 6, 7 and 8.

- [ ] **Step 1: Write `config/registry.hpp`**

```cpp
        /**
         * @brief Finds the entry named @p type, or throws listing what exists.
         *
         * An unknown type is an error whose message says what IS available
         * rather than only what is not, because a driver that knows no type
         * names cannot suggest one and the user has nothing else to go on.
         *
         * @tparam Entry A struct with a @c std::string_view @c type member.
         * @param table The registry, a file-scope constant.
         * @param type The discriminator read from the config file.
         * @param what What kind of thing is being looked up, e.g. "layer type".
         * @return The matching entry.
         * @throws Error Naming @p type, @p what and every entry in @p table.
         */
        template<typename Entry, std::size_t N>
        const Entry& find_entry(const std::array<Entry, N>& table,
                                std::string_view type, std::string_view what)
        {
            for (const Entry& entry : table)
                if (entry.type == type) return entry;

            std::string available;
            for (const Entry& entry : table)
            {
                if (!available.empty()) available += ", ";
                available += entry.type;
            }
            SA_THROW_IF(true, "unknown {} '{}'; this build has: {}", what, type, available);
            return table[0];   // unreachable; SA_THROW_IF(true) always throws
        }
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/functional/unit/test_config.cpp`:

```cpp
TEST(Unit_Config, ChainsLayerWidthsFromTheInputFeatures)
{
    const SushiAI::Config::ModelSpec model = SushiAI::Config::parse_model(
        SushiAI::Config::parse_json(R"({
            "sushiai_model": 1,
            "input": {"features": 6},
            "layers": [{"type": "linear", "units": 8},
                       {"type": "gelu"},
                       {"type": "linear", "units": 4}]
        })", "model.json"), "model.json");

    EXPECT_EQ(model.input_features, 6);
    ASSERT_EQ(model.layers.size(), 3u);
    EXPECT_EQ(model.layers[0].type, "linear");
    EXPECT_EQ(model.layers[1].type, "gelu");

    SushiAI::NN::LayerStack stack = SushiAI::NN::build_layer_stack(model);
    EXPECT_EQ(stack.size(), 3u);
    EXPECT_EQ(stack.output_features(), 4);
}

TEST(Unit_Config, AnUnknownLayerTypeListsTheRegistry)
{
    const SushiAI::Config::ModelSpec model = SushiAI::Config::parse_model(
        SushiAI::Config::parse_json(R"({
            "sushiai_model": 1,
            "input": {"features": 6},
            "layers": [{"type": "conv2d", "units": 8}]
        })", "model.json"), "model.json");

    SA_EXPECT_THROWS_WITH(SushiAI::NN::build_layer_stack(model), "conv2d");
    SA_EXPECT_THROWS_WITH(SushiAI::NN::build_layer_stack(model), "linear");
    SA_EXPECT_THROWS_WITH(SushiAI::NN::build_layer_stack(model), "gelu");
}

TEST(Unit_Config, AnUnknownLayerKeyIsRejectedByTheLayersOwnFactory)
{
    const SushiAI::Config::ModelSpec model = SushiAI::Config::parse_model(
        SushiAI::Config::parse_json(R"({
            "sushiai_model": 1,
            "input": {"features": 6},
            "layers": [{"type": "linear", "unitss": 8}]
        })", "model.json"), "model.json");

    SA_EXPECT_THROWS_WITH(SushiAI::NN::build_layer_stack(model), "unitss");
}

TEST(Unit_Config, AMissingVersionFieldIsAnError)
{
    SA_EXPECT_THROWS_WITH(
        SushiAI::Config::parse_model(
            SushiAI::Config::parse_json(R"({"input": {"features": 6}, "layers": []})",
                                        "model.json"), "model.json"),
        "sushiai_model");
}

TEST(Unit_Config, AFutureModelVersionIsAnError)
{
    SA_EXPECT_THROWS_WITH(
        SushiAI::Config::parse_model(
            SushiAI::Config::parse_json(
                R"({"sushiai_model": 2, "input": {"features": 6}, "layers": []})",
                "model.json"), "model.json"),
        "version 2");
}
```

- [ ] **Step 3: Run and watch them fail**

```powershell
sa build
```

Expected: FAIL — `Config::ModelSpec` does not exist.

- [ ] **Step 4: Write `config/model_config.hpp`**

```cpp
        /** @brief The version this build reads for a model file. */
        inline constexpr std::int64_t MODEL_VERSION = 1;

        /**
         * @struct LayerSpec
         * @brief One layer as the file spelled it: a type, and everything else.
         *
         * @c attributes is the layer object with "type" removed, and it is the
         * layer's own factory that decides which keys it accepts. If this
         * parser validated layer fields, adding "conv2d" would mean editing
         * this parser — which is the schema-side expression of "the driver
         * knows no types".
         */
        struct LayerSpec
        {
            /** @brief The registry key, e.g. "linear". */
            std::string type;

            /** @brief Every other key the layer object carried. */
            Value attributes;
        };

        /**
         * @struct ModelSpec
         * @brief An architecture, and nothing about how it is trained.
         */
        struct ModelSpec
        {
            /** @brief Features per sample entering the first layer. */
            std::int64_t input_features = 0;

            /** @brief The layers, in application order. */
            std::vector<LayerSpec> layers;
        };

        /** @brief Turns a parsed model document into a @ref ModelSpec. */
        ModelSpec parse_model(const Value& root, std::string_view origin);

        /** @brief Reads, parses and validates the model file at @p path. */
        ModelSpec load_model(const std::filesystem::path& path);
```

`parse_model` rejects unknown top-level keys against
`{"sushiai_model", "input", "layers"}`, requires `sushiai_model == MODEL_VERSION`
(message: `"'{}' is a sushiai_model version {} file; this build reads version {}"`),
rejects unknown keys under `input` against `{"features"}`, requires
`input.features > 0`, requires a non-empty `layers` array, and requires each
entry to be an object carrying `"type"`.

- [ ] **Step 5: Write `nn/layer_registry.hpp`**

```cpp
        /**
         * @brief Builds one layer from its spec.
         * @param in_features How many features a sample carries entering it.
         * @param attributes The layer's own keys; the factory rejects any it does not know.
         * @param context What to call this layer in a diagnostic, e.g. "layer 3 'linear'".
         * @return The layer.
         */
        using LayerFactory = std::unique_ptr<ILayer> (*)(int64_t in_features,
                                                         const Config::Value& attributes,
                                                         std::string_view context);

        /** @brief One row of @ref LAYER_REGISTRY. */
        struct LayerEntry
        {
            /** @brief The "type" a model.json spells. */
            std::string_view type;

            /** @brief What builds it. */
            LayerFactory make;
        };

        /** @brief Builds a Linear from "units" and an optional "use_bias". */
        std::unique_ptr<ILayer> make_linear(int64_t in_features, const Config::Value& attributes,
                                            std::string_view context);

        /** @brief Builds a GELU; it takes no keys at all. */
        std::unique_ptr<ILayer> make_gelu(int64_t in_features, const Config::Value& attributes,
                                          std::string_view context);

        /**
         * @brief Every layer type a model.json may name.
         *
         * A file-scope constant table, in the shape of @c Ops::ALL and the
         * fusion CATALOG. **Nothing self-registers**: sushiai is a static
         * library, so a translation unit whose only job is to run a
         * registration constructor resolves no symbol and is dropped by the
         * linker — silently, and only in a downstream consumer's build.
         * Adding a layer is one row here and one factory function.
         */
        inline constexpr std::array<LayerEntry, 2> LAYER_REGISTRY = {
            LayerEntry{"linear", &make_linear},
            LayerEntry{"gelu",   &make_gelu},
        };

        /**
         * @brief Builds the whole stack, chaining each layer's input width.
         *
         * @c units is an output width only; each layer's input is its
         * predecessor's output and @c ModelSpec::input_features seeds the
         * chain, so a dimension mismatch between adjacent layers is not
         * expressible.
         *
         * @param model The parsed architecture.
         * @return The stack.
         * @throws Error If a type is unknown or a factory rejects its attributes.
         */
        LayerStack build_layer_stack(const Config::ModelSpec& model);
```

`make_linear` calls
`attributes.reject_unknown_keys(context, {"units", "use_bias"})`, reads
`units` (must be `> 0`), reads `use_bias` defaulting to `true`, and returns
`std::make_unique<LayerAdapter<Linear>>(Linear{in_features, units, use_bias}, units)`.
`make_gelu` calls `reject_unknown_keys(context, {})` — an empty allowed list,
so any key at all is an error — and returns
`std::make_unique<LayerAdapter<GELU>>(GELU{}, in_features)`.

- [ ] **Step 6: Run the tests and watch them pass**

```powershell
sa build
sa test --suite unit
```

Expected: the five new `Unit_Config` cases PASS.

- [ ] **Step 7: Commit**

```bash
git add include/SushiAI/config/registry.hpp include/SushiAI/config/model_config.hpp \
        src/config/model_config.cpp include/SushiAI/nn/layer_registry.hpp \
        src/nn/layer_registry.cpp tests/functional/unit/test_config.cpp \
        src/CMakeLists.txt include/SushiAI/SushiAI.h
git commit -m "feat(config): build a layer stack from model.json through a constant registry"
```

---

## Task 6: `TrainSpec`, `DataSpec` and relative-path resolution

**Files:**
- Create: `include/SushiAI/config/train_config.hpp`, `src/config/train_config.cpp`
- Test: `tests/functional/unit/test_config.cpp` (append)

**Interfaces:**
- Produces: `Config::DataSpec`, `Config::TrainSpec`, `Config::load_train`,
  `Config::load_data`.

Relative paths resolve against **the config file's own directory**, not the
process working directory, so `sushiai train examples/heavy_mlp/train.json`
works from the repository root.

- [ ] **Step 1: Write the failing test**

```cpp
TEST(Unit_Config, ResolvesRelativePathsAgainstTheConfigsOwnDirectory)
{
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "sushiai_config_test";
    std::filesystem::create_directories(directory);

    {
        std::ofstream out(directory / "train.json");
        out << R"({
            "sushiai_train": 1,
            "model": "model.json",
            "objective": {"type": "cross_entropy"},
            "data": {"type": "rings", "batch_size": 8, "batches": 2, "classes": 4},
            "optimizer": {"type": "adamw", "learning_rate": 0.003},
            "epochs": 3,
            "checkpoint": "out/model.ckpt"
        })";
    }

    const SushiAI::Config::TrainSpec spec =
        SushiAI::Config::load_train(directory / "train.json");

    EXPECT_EQ(spec.model_path, directory / "model.json");
    EXPECT_EQ(spec.checkpoint, directory / "out" / "model.ckpt");
    EXPECT_EQ(spec.objective_type, "cross_entropy");
    EXPECT_EQ(spec.data.type, "rings");
    EXPECT_EQ(spec.optimizer_type, "adamw");
    EXPECT_EQ(spec.epochs, 3);

    std::filesystem::remove_all(directory);
}

TEST(Unit_Config, AnUnknownTopLevelTrainKeyIsAnError)
{
    SA_EXPECT_THROWS_WITH(
        SushiAI::Config::parse_train(
            SushiAI::Config::parse_json(R"({
                "sushiai_train": 1, "model": "m.json",
                "objective": {"type": "cross_entropy"},
                "data": {"type": "rings"},
                "optimizer": {"type": "adamw"},
                "epochz": 3
            })", "train.json"), "train.json", "."),
        "epochz");
}
```

- [ ] **Step 2: Run it and watch it fail**

```powershell
sa build
```

Expected: FAIL — `Config::TrainSpec` does not exist.

- [ ] **Step 3: Write the header and implementation**

```cpp
        /** @brief The version this build reads for a train file. */
        inline constexpr std::int64_t TRAIN_VERSION = 1;

        /** @brief The version this build reads for a standalone data file. */
        inline constexpr std::int64_t DATA_VERSION = 1;

        /**
         * @struct DataSpec
         * @brief Where a run's batches come from.
         */
        struct DataSpec
        {
            /** @brief The @c Data::DATASET_REGISTRY key, e.g. "rings". */
            std::string type;

            /** @brief Every other key; the dataset's own factory validates them. */
            Value attributes;
        };

        /**
         * @struct TrainSpec
         * @brief One training run, end to end.
         */
        struct TrainSpec
        {
            /** @brief The architecture file, resolved against this file's directory. */
            std::filesystem::path model_path;

            /** @brief The @c Train::OBJECTIVE_REGISTRY key. */
            std::string objective_type;

            /** @brief The objective's own keys. */
            Value objective_attributes;

            /** @brief Where the batches come from. */
            DataSpec data;

            /** @brief The @c Optim::OPTIMIZER_REGISTRY key. */
            std::string optimizer_type;

            /** @brief The optimizer's own keys. */
            Value optimizer_attributes;

            /** @brief How many passes over the dataset. */
            std::int64_t epochs = 1;

            /** @brief Where to write the checkpoint; empty means do not write one. */
            std::filesystem::path checkpoint;
        };

        /**
         * @brief Turns a parsed train document into a @ref TrainSpec.
         * @param root The parsed document.
         * @param origin What to call it in a diagnostic.
         * @param base The directory relative paths resolve against.
         * @return The spec.
         * @throws Error If a key is unknown, the version disagrees, or a required key is absent.
         */
        TrainSpec parse_train(const Value& root, std::string_view origin,
                              const std::filesystem::path& base);

        /** @brief Reads, parses and validates the train file at @p path. */
        TrainSpec load_train(const std::filesystem::path& path);

        /** @brief Turns a parsed eval document into a @ref DataSpec. */
        DataSpec parse_data(const Value& root, std::string_view origin);

        /** @brief Reads, parses and validates the data file at @p path. */
        DataSpec load_data(const std::filesystem::path& path);
```

`load_train` computes `base` as `path.parent_path()` (empty → `"."`) and calls
`parse_train`. Path resolution is
`base / std::filesystem::path(text)` followed by
`std::filesystem::weakly_canonical` only when the file exists — the checkpoint's
directory may not exist yet, so it is left lexically joined. Allowed top-level
keys: `{"sushiai_train", "model", "objective", "data", "optimizer", "epochs", "checkpoint"}`.
`epochs` defaults to 1 and must be `> 0`. `checkpoint` is optional.

- [ ] **Step 4: Run the tests and watch them pass**

```powershell
sa build
sa test --suite unit
```

Expected: both new cases PASS.

- [ ] **Step 5: Commit**

```bash
git add include/SushiAI/config/train_config.hpp src/config/train_config.cpp \
        tests/functional/unit/test_config.cpp src/CMakeLists.txt \
        include/SushiAI/SushiAI.h
git commit -m "feat(config): parse train.json and eval.json with directory-relative paths"
```

---

## Task 7: `Data::Rings`, and the dataset and optimizer registries

**Files:**
- Modify: `include/SushiAI/data/dataset.hpp`, `src/data/dataset.cpp`,
  `include/SushiAI/optim/optimizer.hpp`, `src/optim/optimizer.cpp`
- Test: `tests/functional/integration/test_checkpoint.cpp` is Task 9; the
  dataset gets its check here through the existing
  `tests/functional/integration/test_training.cpp` shape.

**Interfaces:**
- Consumes: `Config::Value`, `Config::find_entry`.
- Produces: `Data::RingsOptions`, `Data::Rings`, `Data::DatasetEntry`,
  `Data::DATASET_REGISTRY`, `Data::make_dataset`, `Optim::OptimizerEntry`,
  `Optim::OPTIMIZER_REGISTRY`, `Optim::make_optimizer`.

`Data::Rings` is moved from `examples/heavy_mlp/model.hpp` — the same
host-generated concentric-rings problem, unchanged in its mathematics, with the
class comment's argument for *why* the problem is nonlinear kept intact.

- [ ] **Step 1: Move `Rings` into the library**

`RingsOptions` and `RingsDataset` become `Data::RingsOptions` and `Data::Rings`
in `dataset.hpp` / `dataset.cpp`, declared in the header and defined out of line
(the existing datasets' shape). `sa::` qualifications become plain namespace
members. `name()` returns `"rings"`.

- [ ] **Step 2: Write the failing test**

Append to `tests/functional/integration/test_training.cpp`:

```cpp
TEST(Integration_Data, RingsIsNotLinearlySeparableButAnMlpLearnsIt)
{
    SushiBLAS::Engine& engine = test_engine();      // the fixture this file already uses

    SushiAI::Data::RingsOptions options;
    options.batch_size = 32;
    options.batches = 8;
    const SushiAI::Data::Rings dataset(engine, options);

    EXPECT_EQ(dataset.name(), "rings");
    EXPECT_EQ(dataset.input_dim(), 6);
    EXPECT_EQ(dataset.num_classes(), 4);
    EXPECT_EQ(dataset.batch_count(), 8u);
}

TEST(Integration_Data, TheRegistryBuildsRingsFromAConfigValue)
{
    SushiBLAS::Engine& engine = test_engine();

    const SushiAI::Config::Value attributes = SushiAI::Config::parse_json(
        R"({"batch_size": 16, "batches": 2, "classes": 4, "seed": 7})", "<test>")
        .without("nothing");

    const std::unique_ptr<SushiAI::Data::Dataset> dataset =
        SushiAI::Data::make_dataset(engine, "rings", attributes, "data");

    EXPECT_EQ(dataset->batch_size(), 16);
    EXPECT_EQ(dataset->batch_count(), 2u);
}

TEST(Integration_Data, AnUnknownDatasetTypeListsTheRegistry)
{
    SushiBLAS::Engine& engine = test_engine();
    const SushiAI::Config::Value empty = SushiAI::Config::parse_json("{}", "<test>");
    SA_EXPECT_THROWS_WITH(SushiAI::Data::make_dataset(engine, "coco", empty, "data"), "rings");
}
```

(Check `test_training.cpp` for how it actually reaches an `Engine`; reuse that,
do not invent `test_engine()`.)

- [ ] **Step 3: Add the two registries**

In `dataset.hpp`:

```cpp
        /**
         * @brief Builds one dataset from its config attributes.
         * @param engine The engine that allocates and fills the tensors.
         * @param attributes The dataset's own keys; the factory rejects any it does not know.
         * @param context What to call this dataset in a diagnostic.
         * @return The dataset.
         */
        using DatasetFactory = std::unique_ptr<Dataset> (*)(SushiBLAS::Engine& engine,
                                                            const Config::Value& attributes,
                                                            std::string_view context);

        /** @brief One row of @ref DATASET_REGISTRY. */
        struct DatasetEntry
        {
            /** @brief The "type" a config spells. */
            std::string_view type;

            /** @brief What builds it. */
            DatasetFactory make;
        };

        /** @brief Builds a @ref Rings from batch_size / batches / classes / seed / … */
        std::unique_ptr<Dataset> make_rings(SushiBLAS::Engine& engine,
                                            const Config::Value& attributes,
                                            std::string_view context);

        /** @brief Every dataset type a config may name; see NN::LAYER_REGISTRY on self-registration. */
        inline constexpr std::array<DatasetEntry, 1> DATASET_REGISTRY = {
            DatasetEntry{"rings", &make_rings},
        };

        /**
         * @brief Builds the dataset named @p type.
         * @param engine The engine that allocates and fills the tensors.
         * @param type The registry key.
         * @param attributes The dataset's own keys.
         * @param context What to call it in a diagnostic.
         * @return The dataset.
         * @throws Error If @p type is unknown, listing the registry.
         */
        std::unique_ptr<Dataset> make_dataset(SushiBLAS::Engine& engine, std::string_view type,
                                              const Config::Value& attributes,
                                              std::string_view context);
```

The same shape in `optimizer.hpp`, with

```cpp
        using OptimizerFactory = std::unique_ptr<Optimizer> (*)(
            SushiBLAS::Engine& engine,
            const std::vector<ParameterSlot>& parameters,
            const Config::Value& attributes,
            std::string_view context);
```

and rows `{"adamw", &make_adamw}` and `{"sgd", &make_sgd}`. `make_adamw` accepts
`learning_rate`, `beta1`, `beta2`, `epsilon`, `weight_decay`; `make_sgd` accepts
`learning_rate`, `momentum`, `weight_decay` — check each options struct's actual
members before writing the allowed-key lists, and reject anything else.

- [ ] **Step 4: Run the tests and watch them pass**

```powershell
sa build
sa test --suite integration
```

Expected: the three new `Integration_Data` cases PASS.

- [ ] **Step 5: Commit**

```bash
git add include/SushiAI/data/dataset.hpp src/data/dataset.cpp \
        include/SushiAI/optim/optimizer.hpp src/optim/optimizer.cpp \
        tests/functional/integration/test_training.cpp
git commit -m "feat(data): move the rings problem into the library and add two registries"
```

---

## Task 8: `IObjective` and `build_graph`

**Files:**
- Create: `include/SushiAI/train/objective.hpp`, `src/train/objective.cpp`
- Modify: `include/SushiAI/train/trainer.hpp:220-282` (`build_classifier`)
- Test: `tests/functional/integration/test_training.cpp` (append)

**Interfaces:**
- Consumes: `Graph::GraphBuilder`, `Data::Batch`, `NN::CrossEntropyLoss`.
- Produces: `Train::Metric`, `Train::ObjectiveLeaves`, `Train::IObjective`,
  `Train::CrossEntropyObjective`, `Train::OBJECTIVE_REGISTRY`,
  `Train::make_objective`, `Train::build_graph`. Task 10's driver uses all of
  them.

`build_classifier` hardcodes `NN::CrossEntropyLoss` and `Data::Dataset` demands
a pure-virtual `num_classes()`. A PINN has no classes; YOLO has no `[N, C]`
logits. Without generalising this seam now, `apps/train.cpp` would be rewritten
at the first model that is not a classifier (§5).

`CrossEntropyObjective` reproduces today's behaviour **operation for operation
and in the same order**, so the existing tests are unaffected and any regression
is measurable rather than diffuse.

- [ ] **Step 1: Write `train/objective.hpp`**

```cpp
        /**
         * @struct Metric
         * @brief One named number a step produced, whatever it means.
         *
         * The driver prints "accuracy 92.40%" or "residual 3.1e-4" without
         * knowing which it is.
         */
        struct Metric
        {
            /** @brief What to call it, e.g. "accuracy". */
            std::string_view name;

            /** @brief Its value. */
            double value = 0.0;
        };

        /**
         * @class ObjectiveLeaves
         * @brief The graph values an objective declared, by role.
         *
         * For cross-entropy that is one [N, C] target leaf; for a PINN it would
         * be the collocation points. It is what @c StepSpec::targets
         * generalises into.
         */
        class ObjectiveLeaves
        {
            public:
                /**
                 * @brief Records that @p role is @p id.
                 * @param role What the leaf is for, e.g. "targets".
                 * @param id The declared value.
                 * @throws Error If @p role was already declared.
                 */
                void add(std::string_view role, Graph::ValueId id);

                /** @brief Whether @p role was declared. */
                bool has(std::string_view role) const noexcept;

                /**
                 * @brief The value declared for @p role.
                 * @param role What the leaf is for.
                 * @return Its value id.
                 * @throws Error If @p role was not declared, listing what was.
                 */
                Graph::ValueId at(std::string_view role) const;

                /** @brief Every declared leaf, in declaration order. */
                const std::vector<std::pair<std::string, Graph::ValueId>>& all() const noexcept;

            private:
                std::vector<std::pair<std::string, Graph::ValueId>> leaves_;
        };

        /**
         * @class IObjective
         * @brief What turns a model's output into something to differentiate.
         */
        class IObjective
        {
            public:
                virtual ~IObjective() = default;

                /**
                 * @brief Declares whatever leaves the graph needs beyond the model's input.
                 * @param builder Where the leaves go.
                 * @param leaves Where to record them, by role.
                 */
                virtual void declare(const Graph::GraphBuilder& builder,
                                     ObjectiveLeaves& leaves) const = 0;

                /**
                 * @brief Turns the model's output into the scalar that gets differentiated.
                 * @param builder Where the nodes go.
                 * @param output The model's output value.
                 * @param leaves What @ref declare recorded.
                 * @return The 1-element loss.
                 */
                virtual Graph::ValueId loss(const Graph::GraphBuilder& builder,
                                            Graph::ValueId output,
                                            const ObjectiveLeaves& leaves) const = 0;

                /**
                 * @brief An optional per-step metric: accuracy, mAP, an L2 residual, or none.
                 * @param batch The batch that was just stepped.
                 * @param output The model's output tensor, on the host-visible pool.
                 * @return The metric, or std::nullopt when this objective offers none.
                 */
                virtual std::optional<Metric> score(const Data::Batch& batch,
                                                    const SushiBLAS::Tensor& output) const = 0;

                /** @brief A short name for a run report. */
                virtual std::string_view name() const noexcept = 0;
        };
```

`CrossEntropyObjective` is constructed `{batch, classes}`; `declare` calls
`builder.input(Shape{batch_, classes_}, "targets", dtype_)` and records it under
`"targets"`; `loss` is exactly
`NN::CrossEntropyLoss{}.forward(builder, output, leaves.at("targets"))`; `score`
is the per-row argmax against `batch.labels` that `Trainer::score_accuracy`
already performs — **read that function and reproduce it, do not re-derive it**.

The registry has the same shape as the other three:

```cpp
        using ObjectiveFactory = std::unique_ptr<IObjective> (*)(int64_t batch, int64_t classes,
                                                                 const Config::Value& attributes,
                                                                 std::string_view context);

        struct ObjectiveEntry { std::string_view type; ObjectiveFactory make; };

        std::unique_ptr<IObjective> make_cross_entropy(int64_t batch, int64_t classes,
                                                       const Config::Value& attributes,
                                                       std::string_view context);

        inline constexpr std::array<ObjectiveEntry, 1> OBJECTIVE_REGISTRY = {
            ObjectiveEntry{"cross_entropy", &make_cross_entropy},
        };

        std::unique_ptr<IObjective> make_objective(std::string_view type, int64_t batch,
                                                   int64_t classes,
                                                   const Config::Value& attributes,
                                                   std::string_view context);
```

- [ ] **Step 2: Add `build_graph` and make `build_classifier` a wrapper**

`TrainingGraph` gains one member:

```cpp
            /**
             * @brief The leaves the objective declared, by role.
             *
             * @c StepSpec still names @c targets because @c Train::Trainer
             * feeds exactly that tensor per step; wiring the trainer to feed
             * arbitrary declared leaves is a separate change. Recorded as debt
             * in docs/ARCHITECTURE.md alongside @c Dataset::num_classes().
             */
            ObjectiveLeaves leaves;
```

`build_graph(M& module, const IObjective& objective, int64_t batch, int64_t features, DType, FusionOptions, const IPrecisionPolicy*, bool loss_scaling)`
is `build_classifier`'s body with three substitutions: the `targets` leaf comes
from `objective.declare()`, the loss comes from `objective.loss()`, and the
output-shape check is dropped (an objective that wants one states it itself —
`CrossEntropyObjective::loss` throws when `output` is not `[batch, classes]`,
reproducing today's message). `built.spec.targets` is set from
`built.leaves.at("targets")` when that role exists and left `NO_VALUE` otherwise.

`build_classifier` keeps its exact signature and becomes:

```cpp
        template<typename M>
        TrainingGraph build_classifier(M& module, int64_t batch, int64_t features,
                                       int64_t classes, DType dtype = DType::FLOAT32,
                                       Graph::FusionOptions fusion = {},
                                       const Graph::IPrecisionPolicy* precision = nullptr,
                                       bool loss_scaling = false)
        {
            const CrossEntropyObjective objective(batch, classes, dtype);
            return build_graph(module, objective, batch, features, dtype, fusion,
                               precision, loss_scaling);
        }
```

- [ ] **Step 3: Run the whole suite and watch it stay green**

```powershell
sa build
sa test --suite all
```

Expected: **every existing test passes unchanged.** This is the regression gate
for "operation for operation and in the same order" — in particular
`Regression_LearningCurve`, `Regression_FusionNumerics` and
`Regression_Determinism`, any of which would move if an operation were emitted
in a different order.

- [ ] **Step 4: Add the objective's own tests**

```cpp
TEST(Integration_Objective, CrossEntropyThroughBuildGraphMatchesBuildClassifier)
{
    // Two graphs over the same architecture, one through each entry point.
    // Assert the node counts, the value counts and every node's op id agree.
}

TEST(Integration_Objective, AnUnknownObjectiveTypeListsTheRegistry)
{
    const SushiAI::Config::Value empty = SushiAI::Config::parse_json("{}", "<test>");
    SA_EXPECT_THROWS_WITH(SushiAI::Train::make_objective("yolo", 4, 4, empty, "objective"),
                          "cross_entropy");
}
```

Write the first test's body against `Graph::Graph`'s actual accessors
(`node_count()`, `value_count()`, `node(i)`), which `tests/functional/unit/test_ir.cpp`
already demonstrates.

- [ ] **Step 5: Commit**

```bash
git add include/SushiAI/train/objective.hpp src/train/objective.cpp \
        include/SushiAI/train/trainer.hpp tests/functional/integration/test_training.cpp \
        src/CMakeLists.txt include/SushiAI/SushiAI.h
git commit -m "feat(train): put the loss behind an objective seam and a registry"
```

---

## Task 9: The checkpoint's device lane

**Files:**
- Create: `include/SushiAI/io/checkpoint.hpp`
- Test: `tests/functional/integration/test_checkpoint.cpp`
- Modify: `tests/CMakeLists.txt`, `include/SushiAI/SushiAI.h`

**Interfaces:**
- Consumes: `IO::CheckpointWriter/Reader/TensorRecord`, `NN::for_each_parameter`,
  `Graph::TensorPool`.
- Produces: `IO::save_checkpoint<M>`, `IO::load_checkpoint<M>`.

Each of §7's five defects is closed by making the wrong state unrepresentable
rather than by adding a check beside it. Two of them are already closed by
Task 1 (no file-driven allocation; a version field). The remaining three live
here.

- [ ] **Step 1: Write `io/checkpoint.hpp`**

```cpp
        /**
         * @brief Writes every declared parameter of @p module, by name.
         * @param path Where to write.
         * @param module The trained model; must already have been traced.
         * @param pool Where the parameters' tensors live.
         * @throws Error If the file cannot be opened or a parameter has no tensor.
         */
        template<typename M>
        void save_checkpoint(const std::filesystem::path& path, M& module,
                             Graph::TensorPool& pool);

        /**
         * @brief Loads @p path into @p module's already-materialised parameters.
         *
         * Every parameter the walk yields must be supplied exactly once:
         *
         * - a name the model does not have is an error naming it;
         * - a name supplied twice is an error naming the duplicate;
         * - a parameter the file does not supply is an error naming it;
         * - a dtype or Shape that disagrees is an error printing both.
         *
         * The last is why the format records a Shape and not an element count:
         * a [256, 6] would otherwise load happily into a [6, 256]. And the
         * third is why a @c std::vector<bool> of filled slots replaces a
         * counter — a file with one duplicated name and one missing name has
         * the right count, and the unfilled parameter would then read
         * uninitialised device memory, because @c allocate_tensor does not
         * zero.
         *
         * @param path The checkpoint.
         * @param module A traced model of the same architecture.
         * @param pool Where the parameters' tensors were materialised.
         * @throws Error On any of the four conditions above, or an I/O failure.
         */
        template<typename M>
        void load_checkpoint(const std::filesystem::path& path, M& module,
                             Graph::TensorPool& pool);
```

`load_checkpoint`'s body:

1. Walk `for_each_parameter` into
   `std::vector<std::string> names`, `std::vector<SushiBLAS::Tensor*> targets`,
   `std::vector<bool> filled(names.size(), false)`.
2. `CheckpointReader reader(in);` then `for (i < reader.count())`:
   - `const TensorRecord record = reader.next();`
   - find `record.name` in `names`; absent → throw naming it.
   - `filled[index]` already true → throw `"'{}' supplies '{}' twice"`.
   - dtype disagrees → throw naming both dtypes.
   - `Shape` disagrees → throw printing both via `Shape::to_string()`.
   - `reader.read_payload(record, targets[index]->data());` — straight into the
     destination tensor; **no intermediate buffer**.
   - `filled[index] = true;`
3. After the loop, any `!filled[i]` → throw naming the first such parameter.

Check what `SushiBLAS::Tensor` exposes for a raw byte pointer
(`data()` vs `data_as<T>()`) before writing step 2's payload read, and assert
the tensor's own byte size equals `record.payload_bytes()` before copying.

- [ ] **Step 2: Write the failing tests**

`tests/functional/integration/test_checkpoint.cpp`, five cases:

```cpp
TEST(Integration_Checkpoint, SavesAndLoadsEveryParameterBitForBit)
TEST(Integration_Checkpoint, ADuplicatedNameThrowsAndNamesIt)
TEST(Integration_Checkpoint, AMissingParameterThrowsAndNamesIt)
TEST(Integration_Checkpoint, AShapeMismatchThrowsWithBothShapes)
TEST(Integration_Checkpoint, ASequentialCheckpointLoadsIntoAnEquivalentLayerStack)
```

The last one is **the equivalence test** (§8) and is the point of the whole
task. Its body:

```cpp
    // Two spellings of one model.
    NN::Sequential compile_time{NN::Linear{6, 8}, NN::GELU{}, NN::Linear{8, 4}};

    NN::LayerStack run_time = NN::build_layer_stack(Config::parse_model(
        Config::parse_json(R"({
            "sushiai_model": 1,
            "input": {"features": 6},
            "layers": [{"type": "linear", "units": 8},
                       {"type": "gelu"},
                       {"type": "linear", "units": 4}]
        })", "<test>"), "<test>"));

    // Trace, bind and initialise each against its own pool, then save the
    // compile-time one and load it into the runtime one.
    // Assert: identical parameter names, and every parameter identical bit
    // for bit after the load. Then save the runtime one and load it back into
    // the compile-time one, and assert the same.
```

Compare bit for bit by reading `float` payloads and using `EXPECT_EQ` on the
raw bit patterns (`std::memcpy` into a `std::uint32_t`), not `EXPECT_FLOAT_EQ` —
"identical" is the claim, not "close".

- [ ] **Step 3: Run them and watch them pass**

```powershell
sa build
sa test --suite integration
```

Expected: all five PASS.

- [ ] **Step 4: Commit**

```bash
git add include/SushiAI/io/checkpoint.hpp \
        tests/functional/integration/test_checkpoint.cpp \
        tests/CMakeLists.txt include/SushiAI/SushiAI.h
git commit -m "feat(io): bind checkpoints to a module's parameters through the generic walk"
```

---

## Task 10: `apps/train.cpp` and `apps/inference.cpp`

**Files:**
- Create: `apps/CMakeLists.txt`, `apps/train.cpp`, `apps/inference.cpp`
- Modify: `CMakeLists.txt` (`add_subdirectory(apps)`), `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: everything from Tasks 4–9.
- Produces: the `sushiai_train` and `sushiai_inference` executables.

**The gate for this task is a grep.** Neither file may contain `Linear`, `GELU`,
`CrossEntropy`, `AdamW`, `SGD`, `Rings`, `Sequential` or `LayerStack` as a
constructed type — only the four registry lookups and `NN::LayerStack` as the
variable's type. Run it and paste the output into the commit body:

```powershell
Select-String -Path apps/*.cpp -Pattern 'Linear|GELU|CrossEntropy|AdamW|SGD|Rings|Sequential'
```

Expected: no matches.

- [ ] **Step 1: Write `apps/train.cpp`**

Structure, modelled on `examples/heavy_mlp/train.cpp` with every type name
replaced by a registry lookup:

```cpp
int main(int argc, char** argv)
{
    try
    {
        SA_THROW_IF(argc != 2, "usage: sushiai train <train.json>");

        const Config::TrainSpec spec = Config::load_train(argv[1]);
        const Config::ModelSpec model_spec = Config::load_model(spec.model_path);

        sr::Execution::RuntimeContext context;
        sb::Engine engine(context);

        const std::unique_ptr<Data::Dataset> dataset =
            Data::make_dataset(engine, spec.data.type, spec.data.attributes, "data");

        SA_THROW_IF(dataset->input_dim() != model_spec.input_features,
                    "the model expects {} input features, '{}' supplies {}",
                    model_spec.input_features, dataset->name(), dataset->input_dim());

        NN::LayerStack model = NN::build_layer_stack(model_spec);

        const std::unique_ptr<Train::IObjective> objective =
            Train::make_objective(spec.objective_type, dataset->batch_size(),
                                  dataset->num_classes(), spec.objective_attributes,
                                  "objective");

        Train::TrainingGraph built = Train::build_graph(model, *objective,
                                                        dataset->batch_size(),
                                                        dataset->input_dim());

        Graph::TensorPool pool(engine);
        std::vector<Optim::ParameterSlot> parameters = Train::bind_parameters(built, pool);
        NN::initialize(engine, model, pool);

        const std::unique_ptr<Optim::Optimizer> optimizer =
            Optim::make_optimizer(engine, spec.optimizer_type, parameters,
                                  spec.optimizer_attributes, "optimizer");

        Train::TrainerOptions options;
        options.epochs = 1;
        options.track_accuracy = false;   // the objective owns the metric now
        Train::Trainer trainer(engine, built.graph, pool, built.spec, *optimizer, options);

        // ... the configuration banner, then the epoch loop driving run_step
        //     directly so the objective's own score() is what gets printed ...

        if (!spec.checkpoint.empty())
            IO::save_checkpoint(spec.checkpoint, model, pool);

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "sushiai train failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
```

The epoch loop drives `trainer.run_step(dataset->batch(i))` itself rather than
`trainer.fit()`, so it can call `objective->score(batch, pool.tensor(built.spec.logits))`
and print whatever the objective named — `std::nullopt` prints the loss alone.
That is the property the whole design exists for, and it is worth the ten lines.

The banner **must print a `compiles: N` line** in the same shape as
`sa demo mlp`, because Step 4's smoke test pins it.

- [ ] **Step 2: Write `apps/inference.cpp`**

```
sushiai inference  model.json  model.ckpt  eval.json
```

Three positional arguments; inference takes its data as the third because the
architecture file carries no data block and an optional flag that always errors
when omitted would be a lie about optionality. It builds the stack from
`model.json`, traces a **forward-only** graph (no `build_graph`, no
differentiation — trace `model.forward()` into a bare `Graph::Graph`, plan, bind,
lower through `Graph::CompiledStep` exactly as `examples/heavy_mlp/infer.cpp`
does — read that file and follow it), loads the checkpoint, replays over every
batch and prints the objective-free summary: batch count, sample count and the
predicted-class distribution. No loss and no backward pass.

- [ ] **Step 3: Write `apps/CMakeLists.txt`**

Mirror `examples/CMakeLists.txt` exactly — the target list variable, the
`foreach` applying `target_link_libraries(... PRIVATE sushiai)`,
`sushiai_apply_build_policy`, and the Windows runtime-DLL copy. Add
`add_subdirectory(apps)` to the root `CMakeLists.txt` after
`add_subdirectory(examples)`.

- [ ] **Step 4: Add the two smoke tests**

`tests/CMakeLists.txt`, in the shape of `Integration_Demo.MlpRunsAndReportsOneCompile`.
**Two separate tests, not a chained round trip** (§8):

```cmake
add_test(NAME Integration_App.TrainRunsAndReportsOneCompile
    COMMAND sushiai_train "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny_train.json"
)
set_tests_properties(Integration_App.TrainRunsAndReportsOneCompile PROPERTIES
    LABELS "integration"
    PASS_REGULAR_EXPRESSION "compiles[ \t]*:[ \t]*1[^0-9]"
)
```

The trailing `[^0-9]` is load-bearing: CMake's regex has no multiline mode, so
`$` would anchor to the end of the whole output and `compiles : 12` would
satisfy a bare `...: 1`.

Create `tests/fixtures/{tiny_model.json,tiny_train.json,tiny_eval.json}` — a
2-layer model, batch 8, 2 batches, 2 epochs. The inference smoke test needs a
checkpoint that exists before CTest runs, so `tiny_train.json`'s `checkpoint`
points into `${CMAKE_CURRENT_BINARY_DIR}` and the inference test is declared
with `DEPENDS Integration_App.TrainRunsAndReportsOneCompile` — a CTest ordering
dependency, which is not the same thing as a chained assertion.

- [ ] **Step 5: Run them and watch them pass**

```powershell
sa build
sa test --suite integration
```

Expected: both new `Integration_App` tests PASS.

- [ ] **Step 6: Run the grep gate**

```powershell
Select-String -Path apps/*.cpp -Pattern 'Linear|GELU|CrossEntropy|AdamW|SGD|Rings|Sequential'
```

Expected: no matches. If there are any, the driver is not yet what the design
promises and this task is not done.

- [ ] **Step 7: Commit**

```bash
git add apps CMakeLists.txt tests/CMakeLists.txt tests/fixtures
git commit -m "feat(apps): add config-driven train and inference drivers"
```

Body must include the grep's empty output as evidence.

---

## Task 11: The two CLI subcommands

**Files:**
- Modify: `cli/sushiai/cli.py`, `cli/sushiai/services/project.py` (if the run
  helper lives there), `docs/CLI_GUIDE.md`
- Test: `cli/tests/test_demo.py`'s shape, in a new `cli/tests/test_train.py`

`sushiai` is already a second entry point for the developer CLI
(`cli/pyproject.toml`'s `[project.scripts]` binds both `sa` and `sushiai`), so
these are two new subcommands on an existing app, forwarding their arguments
verbatim to the C++ binary the way `demo mlp` already does.

- [ ] **Step 1: Read how `demo mlp` is wired**

```powershell
Select-String -Path cli/sushiai/cli.py -Pattern 'demo' -Context 3
```

Follow that shape exactly; do not invent a second way to locate a built binary.

- [ ] **Step 2: Add `train` and `inference`**

```python
@app.command()
def train(config: Path = typer.Argument(..., help="A train.json.")) -> None:
    """Train a model described by a JSON configuration."""
```

and

```python
@app.command()
def inference(
    model: Path = typer.Argument(..., help="The model.json the checkpoint was trained under."),
    checkpoint: Path = typer.Argument(..., help="A .ckpt written by `sushiai train`."),
    data: Path = typer.Argument(..., help="An eval.json naming the evaluation data."),
) -> None:
    """Evaluate a trained checkpoint over a dataset."""
```

Both forward verbatim and propagate the binary's exit code.

- [ ] **Step 3: Write the CLI test**

`cli/tests/test_train.py`, in `test_demo.py`'s shape: assert the two commands
appear in `--help`, and that a missing config file exits non-zero with a message
naming the path.

- [ ] **Step 4: Run the CLI suite**

```powershell
python -m pytest cli/tests -q
```

Expected: green. (`sa test` runs CTest only; the pytest suite is CI's, per §8 —
so it is run by hand here.)

- [ ] **Step 5: Update `docs/CLI_GUIDE.md`**

Add both commands with their arguments, an example invocation against
`examples/heavy_mlp/`, and their exit codes, in the file's existing shape.

- [ ] **Step 6: Commit**

```bash
git add cli docs/CLI_GUIDE.md
git commit -m "feat(cli): add the train and inference subcommands"
```

---

## Task 12: The example becomes configuration, and the docs catch up

**Files:**
- Create: `examples/heavy_mlp/{model.json,train.json,eval.json,README.md}`
- Delete: `examples/heavy_mlp/{model.hpp,train.cpp,infer.cpp}`
- Modify: `examples/CMakeLists.txt`, `docs/ARCHITECTURE.md`, `docs/README.md`,
  `docs/CHANGELOG.md`

- [ ] **Step 1: Write the three JSON files**

`model.json`, `train.json` and `eval.json` exactly as §6 spells them. `eval.json`'s
seed differs from training's, so the evaluation set is genuinely unseen — the
property the deleted `infer.cpp` already got right.

- [ ] **Step 2: Write `examples/heavy_mlp/README.md`**

What the rings problem is and why no linear classifier separates it (carried
over from the deleted `model.hpp`'s header comment — that argument is the
example's whole point and must not be lost), the two commands to run, and the
expected accuracy.

- [ ] **Step 3: Delete the three source files and revert `examples/CMakeLists.txt`**

`examples/CMakeLists.txt` returns to its two original targets — the diff should
be exactly the inverse of Task 0's commit.

- [ ] **Step 4: Verify the example end to end**

```powershell
sa build
sushiai train examples/heavy_mlp/train.json
sushiai inference examples/heavy_mlp/model.json heavy_mlp.ckpt examples/heavy_mlp/eval.json
```

Expected: training reports `compiles: 1` and a falling loss; inference loads the
checkpoint and reports a distribution. **Record the actual accuracy printed** —
it goes into the README and into `docs/CHANGELOG.md`, and it must be the number
that ran, not one carried over from the deleted example.

- [ ] **Step 5: Update `docs/ARCHITECTURE.md`**

Four edits, all of which §9 names:

1. Strike **"No checkpointing, no serialization"** from §7 and describe the
   `SACP` format in its place.
2. Add a section on the runtime layer path — `ILayer` / `LayerStack` / the four
   registries — including the argument that a virtual call costs nothing because
   `forward()` runs once at trace time.
3. Correct the stale **"153 tests"** figure in §6 to the count Task 0 Step 1
   recorded plus everything this plan added. Run
   `ctest --test-dir build -N | Select-Object -Last 1` and use that number.
4. Record the two pieces of debt: `Dataset::num_classes()` stays
   classification-shaped, and `Train::Trainer` still feeds `StepSpec::targets`
   rather than arbitrary `ObjectiveLeaves`.

- [ ] **Step 6: Update `docs/README.md` and `docs/CHANGELOG.md`**

`README.md` gains the two commands and a worked `model.json`. `CHANGELOG.md`'s
`[Unreleased]` section gains an entry naming the checkpoint format, the runtime
layer path, the four registries and the two drivers.

- [ ] **Step 7: Run everything one last time**

```powershell
sa build --clean
sa test --suite all
python -m pytest cli/tests -q
```

Expected: all green. Paste the counts into the commit body.

- [ ] **Step 8: Commit**

```bash
git add examples docs
git commit -m "docs: turn the heavy MLP example into configuration"
```

- [ ] **Step 9: Finish the branch**

Announce: "I'm using the finishing-a-development-branch skill to complete this
work." Then follow `superpowers:finishing-a-development-branch`.

---

## Self-review against the spec

| Spec section | Where it lands |
|---|---|
| §1 In: `Config::Value` + JSON `PRIVATE` | Task 4 |
| §1 In: `ModelSpec` / `TrainSpec` / `DataSpec` | Tasks 5, 6 |
| §1 In: `ILayer` / `LayerStack` / `LayerRegistry` | Tasks 3, 5 |
| §1 In: `IObjective` / `ObjectiveRegistry` | Task 8 |
| §1 In: `DatasetRegistry` + `Data::Rings` moved in | Task 7 |
| §1 In: `OptimizerRegistry` | Task 7 |
| §1 In: `checkpoint_format` + `checkpoint` | Tasks 1, 9 |
| §1 In: `module.hpp` +1 detector, +1 arm | Task 2 |
| §1 In: apps with no type names | Task 10 (grep gate) |
| §1 In: CLI subcommands | Task 11 |
| §1 In: example becomes three JSON files | Task 12 |
| §2 Two commands, inference's third positional | Tasks 10, 11 |
| §3 Architecture / file layout | The File structure section above |
| §3 `LayerStack` and `Sequential` name identically | Task 3 (`LayerAdapter`), proved in Task 9 |
| §4 Four constant tables, `"type"` discriminator, unknown type lists the table | Task 5 (`find_entry`), reused by 7 and 8 |
| §4 Layer fields validated by the factory | Task 5 (`reject_unknown_keys` in each factory) |
| §5 `IObjective`, `ObjectiveLeaves`, `Metric`, `build_classifier` wrapper | Task 8 |
| §5 `num_classes()` debt recorded | Task 12 Step 5 |
| §6 `Config::Value`, nlohmann PRIVATE | Task 4 |
| §6 vcpkg manifest | **Deviation 1** — deps.toml instead, with evidence |
| §6 `model.json` / `train.json` / `eval.json` schemas | Tasks 5, 6, 12 |
| §6 Version fields, unknown keys, chained widths, relative paths | Tasks 5, 6 |
| §7 Format, bounds, little-endian, five defects | Tasks 1, 9 |
| §8 Every unit case | Tasks 1, 3, 5, 6 |
| §8 Every integration case, incl. the equivalence test | Tasks 7, 8, 9, 10 |
| §9 Consequences: example, .gitignore, three docs | Tasks 0, 12 |
