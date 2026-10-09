# Domain modules: implementation plan

**Status:** Planned. No task is written. Section 6 holds eight questions the owner has not ruled.

The design is `docs/design/DOMAIN_MODULES.md`. This plan covers its four phases and was measured
on 2026-10-09 against HEAD `7ab5c9e`, which builds and passes 1874 tests. Nothing here was built
or run; section 7 lists what was not read. Line numbers are of the working tree.

## 1. What exists (measured)

### 1.1 The three tables and who reads them

| Table | Where | Rows |
| --- | --- | --- |
| `K_NODE_REGISTRY` | `include/SushiAI/nn/node_registry.hpp:51-59` | `batchnorm2d`, `concat` (owned by `nn`); `conv`, `bottleneck`, `c3k`, `c3k2`, `sppf` (the blocks) |
| `K_LAYER_REGISTRY` | `include/SushiAI/nn/layer_registry.hpp:185-200` | 14 layer types, all owned by `nn` |
| `K_DATASET_REGISTRY` | `include/SushiAI/data/dataset.hpp:363-366` | `rings`, `mnist` |

`node_registry.hpp:37-41` includes the five block headers, so every includer of the registry
compiles the blocks. `NodeEntry` is in `nn/node.hpp:114-126`, `LayerEntry` in
`nn/layer_registry.hpp:51-66`, `DatasetEntry` in `data/dataset.hpp:318-333`. A layer factory
throws; a node factory returns a `Result`. Readers, by search of `*.hpp`, `*.cpp` and `*.h`:

| Symbol | Library | Callers that need a registry parameter |
| --- | --- | --- |
| `build_model` | `src/nn/node_registry.cpp:230` | `apps/train.cpp:93`, `apps/inference.cpp:102`, 26 calls in `tests/functional/unit/test_node_registry.cpp` (`:573` to `:891`) |
| `make_node` | `node_registry.cpp:209`, called at `:165` | 10 calls in `test_node_registry.cpp` (`:193` to `:551`) |
| `build_layer_stack` | `src/nn/layer_registry.cpp:430` | `test_checkpoint.cpp:65`; 8 in `test_config.cpp:210-277`; 8 in `test_layer_registry.cpp:192-794`; 4 in `test_layer_stack.cpp:240-275` |
| `make_dataset` | `src/data/dataset.cpp:641` | `apps/train.cpp:83`, `apps/inference.cpp:91`, 12 calls in `tests/functional/integration/test_registries.cpp:121-245` |
| `K_NODE_REGISTRY` | `node_registry.cpp:213`, `:226` | `test_node_registry.cpp:410`, `:419`, `:429-432`, `:557` |
| `K_LAYER_REGISTRY` | `node_registry.cpp:217`, `:226`; `layer_registry.cpp:452` | `test_layer_registry.cpp:127`, `:217-220`, `:233`; `test_node_registry.cpp:421`, `:503-508`, `:561` |
| `K_DATASET_REGISTRY` | `dataset.cpp:646` | `test_registries.cpp:177` |

`Config` reads no table: it mentions them in two comments, `config/model_config.hpp:78` and
`config/train_config.hpp:53`. Neither example calls any of the four functions.
`Config::find_entry` (`config/registry.hpp:48-66`) throws; it serves layers
(`layer_registry.cpp:452`), datasets (`dataset.cpp:646`), optimizers
(`src/optim/optimizer_registry.cpp:153`) and objectives (`src/train/objective.cpp:148`); see
question 5 for the last two. `K_C3K2_INNER` (`nn/blocks/c3k2.hpp:171`) moves with the blocks.

### 1.2 What moves to `vision/`

| From | Files | Namespace today |
| --- | --- | --- |
| `include/SushiAI/detect/`, `src/detect/` | 17 headers, `README.md`; 14 sources | `namespace SushiAI { namespace Detect` |
| `include/SushiAI/imaging/`, `src/imaging/` | 19 headers, `README.md`; 19 sources and the private `bilinear.hpp` | `namespace SushiAI::Imaging` |
| `include/SushiAI/nn/blocks/`, `src/nn/blocks/` | 5 headers; 5 sources and the private `block_attributes.hpp`, `.cpp` | `namespace SushiAI { namespace NN` |
| `data/dataset.hpp:253-309`, `:345-355`; `src/data/dataset.cpp:57-84`, `:176-244`, `:372-499`, `:610-639` | MNIST | `SushiAI::Data` |
| `tests/functional/unit/` | `test_detect_*.cpp` (5), `test_imaging_*.cpp` (17), `test_block_*.cpp` (5) | |

Includes of the three folders from outside them: 61 lines in 29 code files, which are
`nn/node_registry.hpp` (5), the 27 tests (55) and `test_node_registry.cpp:48` (1).
`SushiAI.h:119-123` and `:134-170` add 41 quoted lines, and `tests/functional/unit/README.md`
holds 38 such paths. No file of `detect` or `imaging` includes `nn`, `graph`, `data`, `train`,
`config`, `optim`, `io`, `autograd`, `eager` or the other of the two. The blocks include `core`,
`graph` and `nn` (`nn/blocks/c3k2.hpp:38-49`). `Detect::` and `Imaging::` occur 318 times in 14
files; 13 imaging tests hold `using namespace SushiAI::Imaging`.

MNIST shares three file-local helpers with the other datasets: `allocate_batch`
(`dataset.cpp:132-162`), `write_one_hot` (`:165-174`) and `MaterialisedDataset` (`:250-282`).
Outside `data` it is used by `examples/example.cpp:51-66`, `examples/demo_mlp.cpp:446-481`,
`test_training.cpp:562-570` and eight cases of `test_registries.cpp:174-275`.
`tools/contract/site_classes.tsv` names a moved file in 131 of its 400 rows.

### 1.3 Build, tests and the CLI

- `src/CMakeLists.txt:14-146` is one source list for one static target, `sushiai` (`:148`);
  lines `80-85` and `96-129` are the vision sources. `install(DIRECTORY)` ships every header
  (`:232-234`), and the completeness check fails on one `SushiAI.h` does not quote (`:261-286`).
- `tests/CMakeLists.txt:37-150` is one binary, `sushiai_tests`. The suite prefix alone assigns
  the CTest label (`:168-182`). Three more tests run the demo and the two apps (`:199-256`).
- The root file's one option, `SA_BUILD_TESTS`, is `ON` with no top-level test
  (`CMakeLists.txt:54`). SushiBLAS compares `CMAKE_SOURCE_DIR` with `CMAKE_CURRENT_SOURCE_DIR`
  and makes the result the option's default (`sushiblas/CMakeLists.txt:94-98`, `:137`).
- Both apps, both examples, `tests/common/test_utils.hpp:41` and `tests/package/main.cpp:30`
  include `SushiAI/SushiAI.h`; the four programs link `sushiai` (`apps/CMakeLists.txt:18`,
  `examples/CMakeLists.txt:18`). `cmake/SushiAIConfig.cmake.in:32` loads one targets file.
- `sa run <name>` finds an executable by name in `build/` through `sushicore`
  (`cli/sushiai/services/discovery.py:20-24`, `project.py:490`). The CLI names four targets,
  none a test binary (`cli/sushiai/cli.py:36-37`, `project.py:58-60`, `cli/config.toml:19`), so
  `sa run sushiai_vision_tests` needs no CLI change as far as this repository shows. `sa test`
  selects by label (`project.py:292-295`); `sa build -D` reaches configure (`:153-164`).

### 1.4 The layering checker

`tools/layering/check_layering.py` finds modules under `<root>/modules/<tier>/<module>/`
(`:52-59`) and returns no subject for a repository without that folder (`:62-67`). SushiAI has
none, so the three rules run over nothing and CI passes (`.github/workflows/ci.yml:220-221`).
`K_TIER_ORDER` is empty (`:23`). `tools/tests/` holds six test files, none for this checker.
Against the design's tier table, `train` and `io` include `nn` today: 24 includes in 14 files,
`train/steps.hpp` and `io/checkpoint.hpp` among them. No file of `nn` includes either.

## 2. The contract every task follows

1. New code reports a content failure as a `Status` or a `Result<T>` and a misuse with
   `SA_FATAL_IF`, holds no `try`, `catch` or `throw`, and propagates with the `SA_TRY` family.
   A host allocation that can fail runs inside `Detail::guard` (`dataset.cpp:155-159`).
2. Code that throws today and is only moved keeps throwing: the layer factories,
   `build_layer_stack`, `Config::Value`'s readers. A status function reaches a layer factory
   through `NN::Detail::bridge_throwing`, as `node_registry.cpp:100-103` does.
3. A registry is a value passed by reference. No task adds a global registry, a function-local
   static registry, or an object whose constructor registers a row.
4. Names follow `cpp-code-style`: `K_UPPER_SNAKE` constants, `[[nodiscard]]` on every status,
   one primary type per header. Comments follow `source-comments`: the licence box and the
   `@file` block of `nn/node_registry.hpp:1-26` on every new file, a `@brief` on every symbol.
5. A subagent never builds. It checks each source it wrote with `clang -fsyntax-only` through
   `compile_commands.json` and pastes the command and the output. A moved file has no entry
   there until the owner's next configure; the task uses the flags of its old path and says so.
6. No task edits a `CMakeLists.txt`, either umbrella header, a file under `cmake/`,
   `docs/design/`, the manual or the changelog; section 5 lists what the orchestrator adds.
   Nobody edits `docs/guides/CLI_GUIDE.md`; what this work makes false on its 12 matching lines
   is reported to the owner.
7. The orchestrator moves files with `git mv` before a wave starts. A task edits a moved file in
   place and creates no copy of it.

## 3. Phases

### Phase 1: registries become values

```cpp
namespace SushiAI::NN                                  // nn/node_registry.hpp
{
    class NodeRegistry final
    {
        public:
            [[nodiscard]] Status add(const NodeEntry& row);    // InvalidDescription on a held type
            [[nodiscard]] Status add(const LayerEntry& row);
            [[nodiscard]] const NodeEntry* find_node(std::string_view type) const noexcept;
            [[nodiscard]] const LayerEntry* find_layer(std::string_view type) const noexcept;
            [[nodiscard]] SushiRuntime::span<const NodeEntry> nodes() const noexcept;
            [[nodiscard]] SushiRuntime::span<const LayerEntry> layers() const noexcept;
    };
    [[nodiscard]] Result<std::unique_ptr<Node>> make_node(
        const NodeRegistry& registry, std::string_view type, ShapeSpan inputs,
        const Config::Value& attributes, std::string_view context);
    [[nodiscard]] Result<ModelGraph> build_model(const NodeRegistry& registry,
                                                 const Config::ModelSpec& model);
    // nn/layer_registry.hpp, which forward-declares NodeRegistry: 14 layers, batchnorm2d, concat
    [[nodiscard]] Status register_layers(NodeRegistry& registry);
    LayerStack build_layer_stack(const NodeRegistry& registry, const Config::ModelSpec& model);
    // nn/blocks/registration.hpp until phase 2: conv, bottleneck, c3k, c3k2, sppf
    [[nodiscard]] Status register_blocks(NodeRegistry& registry);
}
namespace SushiAI::Data                                // data/dataset_registry.hpp
{
    class DatasetRegistry final
    {
        public:
            [[nodiscard]] Status add(const DatasetEntry& row);
            [[nodiscard]] const DatasetEntry* find(std::string_view type) const noexcept;
            [[nodiscard]] SushiRuntime::span<const DatasetEntry> rows() const noexcept;
    };
    [[nodiscard]] Status register_datasets(DatasetRegistry& registry);    // rings
    [[nodiscard]] Result<std::unique_ptr<Dataset>> make_dataset(
        const DatasetRegistry& registry, SushiBLAS::Engine& engine, std::string_view type,
        const Config::Value& attributes, std::string_view context);
    [[nodiscard]] Status register_mnist(DatasetRegistry& registry);       // in dataset.hpp
}
```

Both registries store rows in a `std::vector`, keep insertion order and follow the rule of zero.
A row with an empty type or a null factory is fatal. A row's `type` is a `std::string_view`, so
its text must outlive the registry, as a literal does.

**T1.1 `NodeRegistry` and the three builders.** Files: `include/SushiAI/nn/node_registry.hpp`,
`src/nn/node_registry.cpp`, `include/SushiAI/nn/layer_registry.hpp`, `src/nn/layer_registry.cpp`.
Deletes both tables. `node_registry.hpp` drops its includes of `batch_norm2d.hpp`, `concat.hpp`
and the five block headers and includes `layer_registry.hpp` for `LayerEntry`. `make_node` asks
`find_node`, then `find_layer`, and keeps today's unknown-type text (`node_registry.cpp:224-226`)
over `nodes()` and `layers()`. `add_row` (`:143`) and `build_model` pass the registry down.
`build_layer_stack` calls `find_layer` and throws the text of `config/registry.hpp:64` for an
absent type. `register_layers` is defined in `layer_registry.cpp`. Tests: T1.4, T1.5.
Acceptance: both sources pass the syntax check, and `K_NODE_REGISTRY` and `K_LAYER_REGISTRY` are
gone from `include/` and `src/`. Dispatch: `opus low`.

**T1.2 `register_blocks`.** Files: `include/SushiAI/nn/blocks/registration.hpp`,
`src/nn/blocks/registration.cpp` (new). Five `SA_TRY(registry.add(NodeEntry{...}))` lines in the
order of `node_registry.hpp:54-58`. Tests: T1.4. Acceptance: syntax check once T1.1 is on disk.
Dispatch: `haiku medium`.

**T1.3 `DatasetRegistry`.** Files: `include/SushiAI/data/dataset_registry.hpp`,
`src/data/dataset_registry.cpp` (new), `include/SushiAI/data/dataset.hpp`,
`src/data/dataset.cpp`, and the two comments at `config/model_config.hpp:78` and
`config/train_config.hpp:53`. `DatasetFactory`, `DatasetEntry` and `make_dataset` leave
`dataset.hpp:311-333`, `:357-378`. `make_dataset` returns `InvalidDescription` with
`unknown dataset type '{}'; this build has: {}` when `find` returns null. Tests: T1.6.
Acceptance: both sources pass the syntax check; `K_DATASET_REGISTRY` is gone from the tree.
Dispatch: `haiku high`.

**T1.4 The node registry's tests.** Files: `tests/functional/unit/test_node_registry.cpp`. A
file-local `composed()` builds a registry from `register_layers` and `register_blocks`, and the
36 calls take it first. Cases that read a table read `nodes()` and `layers()`.

| Case | Pins |
| --- | --- |
| `RegisterLayersAddsFourteenLayersAndTwoNodes` (was `TheTypeStringsAreTheOnesModelFilesSpell`) | The 14 layer types in order, then `batchnorm2d`, `concat` |
| `RegisterBlocksAddsTheFiveBlocks` | `conv`, `bottleneck`, `c3k`, `c3k2`, `sppf`, in order |
| `ATypeAddedTwiceIsRefused` (was `NoTypeIsInBothTables`) | `InvalidDescription` naming the type, node over node and node over layer |
| `AnEmptyRegistryKnowsNoType` | `make_node` over an empty registry fails and lists no type |
| `ARegistryWithoutTheBlocksRefusesConv` | `build_model` of `K_BRANCHING_MODEL` over `register_layers` alone fails at `layer 0 'conv'` |
| Acceptance | The file passes the syntax check and the five cases exist. Dispatch: `opus low` |

**T1.5 The callers of `build_layer_stack`.** Files: `test_layer_registry.cpp`,
`test_layer_stack.cpp`, `test_config.cpp` (unit), `test_checkpoint.cpp` (integration). The 21
calls take a registry built by `register_layers`; `test_layer_registry.cpp:127` calls
`find_layer`, and `:217-233` read `layers()`. No case is added or removed. Acceptance: the four
files pass the syntax check. Dispatch: `haiku high`.

**T1.6 The dataset registry's tests.** Files: `tests/functional/integration/test_registries.cpp`,
`tests/functional/unit/test_dataset_registry.cpp` (new). The 12 calls take a registry holding
`rings` and `mnist`. `AnUnknownDatasetTypeListsEveryTypeThisBuildHas` and the second half of
`TheDatasetRegistryListsMnist` become `SA_EXPECT_FAILURE(..., InvalidDescription, ...)` over the
same texts. New suite `Unit_DatasetRegistry`: `AddKeepsInsertionOrder`,
`FindReturnsNullForAnUnknownType`, `ATypeAddedTwiceIsRefused`, `RegisterDatasetsAddsRingsOnly`.
Acceptance: both files pass the syntax check. Dispatch: `haiku medium`.

**T1.7 The apps compose their registries.** Files: `apps/registries.hpp`, `apps/registries.cpp`
(new), `apps/train.cpp`, `apps/inference.cpp`. The header declares, in `SushiAI::Apps`, the
struct `Registries` with the members `NN::NodeRegistry nodes` and `Data::DatasetRegistry
datasets`, and `[[nodiscard]] Result<Registries> compose_registries()`, which calls the four
`register_*` functions above. Each app calls it once before `make_dataset` (`train.cpp:83`,
`inference.cpp:91`) and passes `registries.datasets` and `registries.nodes`. The printed lines
keep their text. Tests: the two CTest cases at `tests/CMakeLists.txt:235-256`. Acceptance: the
three sources pass the syntax check. Dispatch: `opus low`.

### Phase 2: vision moves

After the move a module is `include/SushiAI/vision/<module>/` and `src/vision/<module>/`, for
`detect`, `imaging`, `blocks` and `datasets` (`mnist.hpp`, `mnist.cpp`), in the namespaces
`SushiAI::Vision::Detect`, `Imaging`, `Blocks` and `Datasets`. `vision/registration.hpp` and
`src/vision/registration.cpp` are in `SushiAI::Vision`. The tests are in `tests/vision/`.
```cpp
namespace SushiAI::Vision                              // vision/registration.hpp
{
    [[nodiscard]] Status register_nodes(NN::NodeRegistry& registry);          // the five blocks
    [[nodiscard]] Status register_datasets(Data::DatasetRegistry& registry);  // mnist
}
namespace SushiAI::Data                                // data/materialised_dataset.hpp, question 3
{
    class MaterialisedDataset final : public Dataset;  // as dataset.cpp:250-282
    [[nodiscard]] Result<Batch> allocate_batch(SushiBLAS::Engine& engine, int64_t rows,
                                               const Shape& sample, int64_t classes);
    void write_one_hot(SushiBLAS::Tensor& targets, const std::vector<int32_t>& labels,
                       int64_t classes);
}
```

The six MNIST names of `dataset.hpp:257-355` keep their signatures in `Vision::Datasets`, and
`K_C3K2_INNER` stays a `constexpr` table. Detection and the blocks keep their two-level form,
written `namespace SushiAI { namespace Vision::Detect {`, so no body line is indented again;
imaging keeps its one-line form. Making the forms one is a formatting commit outside this plan.

**T2.1 Detection.** Files: the 17 headers and `README.md` of `include/SushiAI/vision/detect/`,
the 14 sources of `src/vision/detect/`. `<SushiAI/detect/` becomes `<SushiAI/vision/detect/`, and
`namespace Detect` becomes `namespace Vision::Detect` with its closing comment. Acceptance: the
14 sources pass the syntax check. Dispatch: `haiku high`.

**T2.2 Imaging.** Files: the 19 headers and `README.md` of `include/SushiAI/vision/imaging/`,
the 20 files of `src/vision/imaging/`. The same two edits with `SushiAI::Vision::Imaging`.
Acceptance: the 19 sources pass the syntax check. Dispatch: `haiku high`.

**T2.3 Blocks.** Files: the 5 headers of `include/SushiAI/vision/blocks/`, the 7 files of
`src/vision/blocks/`. `namespace NN` becomes `namespace Vision::Blocks`, and every name of `nn`
the files use gains `NN::` (`Conv2d`, `BatchNorm2d`, `SiLU`, `Identity`, `MaxPool2d`, `Node`,
`ShapeSpan`, `Trace`, `TraceModule`, `ModuleList`, `NormOptions`, `Detail::`). No header holds a
`using` directive, and no diagnostic text changes. Acceptance: the 6 sources pass the syntax
check. Dispatch: `opus low`.

**T2.4 MNIST leaves `data`.** Files: `include/SushiAI/vision/datasets/mnist.hpp`,
`src/vision/datasets/mnist.cpp`, `include/SushiAI/data/materialised_dataset.hpp`,
`src/data/materialised_dataset.cpp` (all new), `include/SushiAI/data/dataset.hpp`,
`src/data/dataset.cpp`. The MNIST ranges of section 1.2 and `register_mnist` leave `data`;
`mnist.cpp` reads its two integers with `attributes.at_or`, as `dataset.cpp:101-107` does. The
three shared helpers move to the new `data` pair unchanged, and `SyntheticClassification` and
`Rings` call them there. Acceptance: the three sources pass the syntax check, and a search for
`mnist` in any case under `include/SushiAI/data/` and `src/data/` finds nothing. Dispatch:
`opus low`.

**T2.5 Registration.** Files: `include/SushiAI/vision/registration.hpp`,
`src/vision/registration.cpp`, moved from `nn/blocks/registration.*`. `register_nodes` is the
body of `register_blocks` over the `Blocks::` factories; `register_datasets` adds `mnist` with
`Datasets::make_mnist`. Acceptance: syntax check once T2.3 and T2.4 are on disk. Dispatch:
`haiku medium`.

**T2.6 Detection and imaging tests.** Files: `tests/vision/test_detect_*.cpp` (5) and
`tests/vision/test_imaging_*.cpp` (17), two dispatches. Include paths gain `vision/`,
`"../../common/` becomes `"../common/`, `using namespace SushiAI::Imaging` becomes
`SushiAI::Vision::Imaging`, and `Detect::` and `Imaging::` after `using namespace SushiAI` gain
`Vision::`. Suite and case names do not change. Acceptance: the 22 files pass the syntax check.
Dispatch: `haiku high`, twice.

**T2.7 Block tests.** Files: `tests/vision/test_block_*.cpp` (5). The path edits of T2.6, and
`NN::` becomes `Vision::Blocks::` on the names the five block headers declare, among them
`K_C3K2_INNER` and its two factories (`test_block_c3k2.cpp:410-414`). Acceptance: the five files
pass the syntax check. Dispatch: `haiku high`.

**T2.8 The registry tests split.** Files: `tests/functional/unit/test_node_registry.cpp`,
`tests/vision/test_registration.cpp` (new). The core file keeps what `nn` alone can show and
names no vision type: its two model texts (`:138-160`) use `conv2d`, `batchnorm2d` and `silu` in
place of `conv`, `ANodeTypeBuildsItsOwnNode` (`:475-486`) uses `concat`, and `K_MINIMAL_CASES`
keeps two rows. The new file, suite `Unit_VisionRegistration`, takes the five block rows of
`K_MINIMAL_CASES`, `RegisterNodesAddsTheFiveBlocks`, `ARegistryWithoutTheBlocksRefusesConv`,
every case that loads `examples/yolo26/model.json` (`:746` to the end of the file), and a new
`RegisterDatasetsAddsMnist`. Acceptance: both files pass the syntax check; the core file
includes no vision header; the report lists every case renamed. Dispatch: `opus low`.

**T2.9 MNIST tests.** Files: `tests/vision/test_datasets_mnist.cpp` (new), and
`test_registries.cpp` and `test_training.cpp` under `tests/functional/integration/`. The eight
cases of `test_registries.cpp:174-275` and the case at `test_training.cpp:562-571` move to the
suite `Integration_VisionMnist`, over a registry built by `Vision::register_datasets`.
`test_registries.cpp` keeps a registry of `register_datasets` alone. Acceptance: the three files
pass the syntax check and the two core files name no `Mnist`. Dispatch: `haiku high`.

**T2.10 The programs.** Files: `apps/registries.cpp`, `examples/example.cpp`,
`examples/demo_mlp.cpp`. `compose_registries` calls `Vision::register_nodes` and
`Vision::register_datasets`. The examples include `<SushiAI/vision/datasets/mnist.hpp>` and
spell `sa::Vision::Datasets::` at `example.cpp:51-66` and `demo_mlp.cpp:446-481`. Acceptance:
the three files pass the syntax check. Dispatch: `haiku medium`.

### Phase 3: the second target

The orchestrator writes the build files (section 5). Two tasks remain, around one signature in
`apps/domains.hpp`: `[[nodiscard]] Status SushiAI::Apps::register_domains(NN::NodeRegistry&
nodes, Data::DatasetRegistry& datasets);`.

**T3.1 The apps choose their domains at link time.** Files: `apps/domains.hpp`,
`apps/domains_vision.cpp`, `apps/domains_none.cpp` (new), `apps/registries.cpp`.
`domains_vision.cpp` calls the two `Vision::register_*` functions; `domains_none.cpp` returns
success. `registries.cpp` calls `register_layers`, `register_datasets` and `register_domains`
and includes no vision header. CMake compiles one of the two sources, so no file holds an `#if`.
Acceptance: the three sources pass the syntax check. Dispatch: `haiku high`.

**T3.2 The package consumer of vision.** Files: `tests/package/vision_main.cpp` (new). It
includes `<SushiAI/vision/vision.h>`, calls `register_layers`, `Vision::register_nodes` and
`Vision::register_datasets`, and returns 1 unless `find_node("c3k2")` and `find("mnist")` are
both non-null. Acceptance: syntax check once `vision.h` is on disk. Dispatch: `haiku medium`.

### Phase 4: the layering check

```python
K_TIER_ORDER: tuple[str, ...] = ("core", "layers", "domains", "programs")
K_MODULE_TIERS: dict[str, str] = {
    "core": "core", "graph": "core", "autograd": "core", "optim": "core", "config": "core",
    "eager": "core", "data": "core", "vision": "domains",
    "nn": "layers", "train": "layers", "io": "layers",      # question 2
}
K_LIBRARY_ROOTS = ("include/SushiAI", "src")     # a module is <root>/<module>/
K_SCOPED_ROOTS = {"tests/functional": "layers", "tests/vision": "domains",
                  "apps": "programs", "examples": "programs"}
```

**T4.1 The checker for this tree.** Files: `tools/layering/check_layering.py`,
`tools/tests/test_check_layering.py` (new). `_discover` reads the two library roots in place of
`modules/`; a file directly under a root (`SushiAI.h`, `version.cpp`) belongs to no module. A
subject's tier comes from `K_MODULE_TIERS` or `K_SCOPED_ROOTS`. Four rules in `K_CHECKER.rules`:

| Rule | Fails on |
| --- | --- |
| `rule_declared_tiers` | A module folder the table lacks, a table entry with no folder, a tier outside `K_TIER_ORDER` |
| `rule_upward_includes` | `#include <SushiAI/<module>/...>` whose module ranks above the includer's tier |
| `rule_cross_domain_includes` | A file of one domain that includes another domain |
| `rule_private_reach` | A quoted include that leaves its module's folder under a library root |

Tests, `unittest` over temporary trees as `tools/tests/test_check_contract.py:59-70` builds
them, each named for what it pins: `test_a_downward_include_is_clean`,
`test_an_upward_include_is_reported` (one issue, with file and line),
`test_a_domain_including_another_domain_is_reported`,
`test_a_core_test_including_a_domain_is_reported`, `test_an_undeclared_module_folder_is_reported`,
`test_a_root_header_belongs_to_no_module`, `test_this_repository_is_clean` (exit code 0).
Acceptance: `python -m unittest tools/tests/test_check_layering.py` passes and
`python tools/layering/check_layering.py .` prints nothing; the report pastes both outputs.
Dispatch: `opus low`.

## 4. Waves

| Wave | Tasks | Tree parses after it | Owner's build |
| --- | --- | --- | --- |
| 1 | T1.1, T1.2, T1.3 | Library yes; the callers of section 1.1 no | |
| 2 | T1.4, T1.5, T1.6, T1.7 | Yes | B1: `sa build`, `sa test` |
| 3, step a | `git mv` by the orchestrator; T2.1, T2.2, T2.3, T2.4; then T2.5 | Library yes; tests and programs no | |
| 3, step b | T2.6, T2.7, T2.8, T2.9, T2.10 | Yes | B2: `sa build`, `sa test` |
| 4 | T3.1, T3.2, the orchestrator's build files | Yes | B3: `sa build`, `sa test`, `sa test --suite package`, `sa run sushiai_vision_tests`. B4: `sa build --clean -D SA_BUILD_VISION=OFF`, `sa test` |
| 5 | T4.1 | Yes | None; the task runs Python only |

File sets within a wave are disjoint, and a wave starts when the one before it is on disk. Phase
2 is one wave closed by one build. In B2 the vision sources and tests are still in `sushiai` and
`sushiai_tests` under their new paths, so B2 separates a wrong move from a wrong target. B4 is
the exit condition of phase 3 and a second configure, so it cannot join B3.

## 5. What the orchestrator adds

| Wave | Build lists | Umbrella | Other |
| --- | --- | --- | --- |
| 2 | `src/CMakeLists.txt`: `nn/blocks/registration.cpp`, `data/dataset_registry.cpp`. `tests/CMakeLists.txt`: `functional/unit/test_dataset_registry.cpp`. `apps/CMakeLists.txt`: `registries.cpp` in both targets | `SushiAI.h`: `nn/blocks/registration.hpp`, `data/dataset_registry.hpp` | Changelog; the READMEs of `nn` and `apps` |
| 3 | `src/CMakeLists.txt`: lines `80-85` and `96-129` under `vision/`, plus `vision/registration.cpp`, `vision/datasets/mnist.cpp`, `data/materialised_dataset.cpp`. `tests/CMakeLists.txt`: 27 paths under `vision/`, plus `vision/test_registration.cpp`, `vision/test_datasets_mnist.cpp` | `SushiAI.h`: the 41 lines of `:119-123`, `:134-170` under `vision/`, plus `data/materialised_dataset.hpp`, `vision/datasets/mnist.hpp`, `vision/registration.hpp` | The `file` column of 131 rows of `site_classes.tsv`; `tests/vision/README.md` from the 38 paths of `tests/functional/unit/README.md`; `include/SushiAI/nn/README.md`; `docs/architecture/ARCHITECTURE.md`, `docs/guides/MODEL_FILE.md`; changelog; the design's status line |
| 4 | Root: `SA_IS_TOP_LEVEL` as `sushiblas/CMakeLists.txt:94-98`, then `option(SA_BUILD_VISION ... ${SA_IS_TOP_LEVEL})`. `src/vision/CMakeLists.txt`: `sushiai_vision`, alias `SushiAI::vision`, `EXPORT_NAME vision`, `PUBLIC SushiAI::sushiai`, `sushiai_apply_build_policy`, `sushiai_apply_sycl`, C++17, in the `SushiAITargets` export set, its own completeness check. `src/CMakeLists.txt`: the vision sources leave, the glob at `:261-264` excludes `vision/`, `install(DIRECTORY)` excludes `vision` when the option is off. `tests/vision/CMakeLists.txt`: `sushiai_vision_tests` with `../common/test_main.cpp`, the three discovery blocks, the fixture definition, the DLL copy. `apps/CMakeLists.txt`: `domains_vision.cpp` and `SushiAI::vision`, or `domains_none.cpp`. `examples/CMakeLists.txt` and the demo test at `tests/CMakeLists.txt:199-207`: behind the option. `tests/package/CMakeLists.txt`: a second consumer when `SushiAI::vision` is a target | New `vision/vision.h`, which quotes `../SushiAI.h` and every header under `vision/`; `SushiAI.h` loses its `vision/` lines | Changelog; the design's status line |
| 5 | None | None | `tools/README.md:12`, `:35`; `docs/design/REMAINING_WORK.md:123`, `:235`; changelog; the design's status line to `Shipped` |

## 6. Questions for the owner

Each is a boundary, an interface or an ownership the design leaves open. The plan is written
under the recommendation.

**Decided by the owner on 2026-10-09:** every question takes its recommendation.

1. **The shape of `NN::NodeRegistry`.** The design gives it `add(row)` and `find(name)`; there
   are two row types whose factories differ in signature and in error contract. (a) One
   registry, two `add` overloads, `find_node` and `find_layer`. (b) Two registries, both passed
   to `build_model`. (c) One row type: each layer gets a node factory and `LayerEntry` goes.
   Recommendation: a. `build_layer_stack` needs layers, and c rewrites 14 factories.
2. **The tier of `train` and `io`.** The design puts both in tier 1, below `nn`; both include
   `nn` today, in 14 files. (a) Declare them in tier 2 with `nn`. (b) Keep the table and remove
   the 24 includes, which moves `ModelState`, `ModelOutputs` and `Trace` below `nn`.
   Recommendation: a. No tier below a domain names it either way.
3. **The helpers MNIST shares with `data`.** (a) `allocate_batch`, `write_one_hot` and
   `MaterialisedDataset` become public in `data/materialised_dataset.hpp`. (b) Vision gets its
   own copy. Recommendation: a; the next domain's dataset needs the same three.
4. **Programs when `SA_BUILD_VISION` is off.** The apps name no type; both examples call
   `try_load_mnist`. (a) The apps link `domains_vision.cpp` or `domains_none.cpp`; the examples
   and `sa demo mlp` are built only with vision. (b) Apps and examples all require vision.
   (c) One `#if` in the apps. Recommendation: a.
5. **The optimizer and objective tables** (`optim/optimizer_registry.hpp:101`,
   `train/objective.hpp:240`) stay `constexpr` behind `Config::find_entry`, or become values
   now. Recommendation: leave them until the design's first open item settles the objective.
6. **Where the layering checker changes.** Its discovery assumes a `modules/` layout this
   repository lacks, and the file reads as shared with other repositories. (a) Rewrite it in
   this repository's copy. (b) Change it in SushiSkills first and take the new copy.
   Recommendation: a now, b as a backlog line, so phase 4 waits on no other repository.
7. **How the package exports vision.** (a) `SushiAI::vision` in the one `SushiAITargets` export
   set, present when built. (b) A `vision` component. (c) A second package. Recommendation: a.
8. **`sa test` and the vision suite.** (a) No CLI change: the existing suites run the vision
   cases by their prefixes, and `sa run sushiai_vision_tests` runs them alone. (b) A `vision`
   entry in `Suite`. Recommendation: a; b edits the CLI and its guide, which another session holds.

## 7. Not verified

- **`sushicore`** is outside this repository and was not read. That `sa run
  sushiai_vision_tests` works unchanged rests on `sa run sushiai_tests` going through the same
  call. B3 settles it.
- **The second target's build.** `cmake/SushiAISyclTarget.cmake:30-43` was read; the toolchain
  function it calls was not. `tests/package/CMakeLists.txt:28-81` checks the interface of
  `SushiAI::sushiai` only; what `SushiAI::vision` exports was not worked out. B3 settles both.
- **Not read:** `src/config/model_config.cpp`, `src/nn/layer_stack.cpp`, the bodies of the block
  sources, `apps/README.md`, both examples beyond the search results, `tests/package/main.cpp`,
  the READMEs under `detect/` and `imaging/`, `tools/naming/rename_map.tsv` (18 matching lines),
  `Doxyfile` and `docs/guides/CLI_GUIDE.md`.
- **The list of `nn` names in T2.3** comes from the block headers' includes and from search
  results. The syntax check of the six sources finds a missed one.
- **Whether a test outside the 27 reaches a vision type through `SushiAI.h`.** The block names
  were found in the five block tests and `test_node_registry.cpp` only; `Detect::` and
  `Imaging::` were searched in qualified form only. B4 finds a missed one. Every count here is
  a text search of the working tree.
