# Domain modules: implementation plan

**Status:** Planned. Phases 1 and 2 are built; the move to the `repository-layout` tree is not started. Section 8 holds seven questions.

The owner ruled on 2026-10-09 that the whole repository follows the `repository-layout` skill.
This plan replaces the four-phase plan: it takes the tree from HEAD `bb86983` plus the
uncommitted phase 3 and 4 work to `modules/<tier>/<module>/`, and finishes the second target and
the layering check in that layout. `docs/design/DOMAIN_MODULES.md` still holds the tiers and the
rulings; its paths are superseded. Nothing here was built or run. Counts were measured on the
working tree with two scratch scripts that are deleted.

## 1. Done

- Phase 1, commit `cf9fe05`: `NN::NodeRegistry` and `Data::DatasetRegistry` are values a caller
  builds and passes; the static tables are gone. Build B1 passed 1881 tests.
- Phase 2, commit `bb86983`: `detect`, `imaging`, `blocks` and `datasets` sit under
  `include/SushiAI/vision/` and `src/vision/` in `SushiAI::Vision::*`; their 29 tests sit under
  `tests/vision/`.
- Uncommitted and unbuilt (19 modified files, 10 untracked paths): target `sushiai_vision` in
  `src/vision/CMakeLists.txt`, `include/SushiAI/vision/vision.h`, `SA_BUILD_VISION`,
  `tests/vision/CMakeLists.txt`, `apps/domains*.cpp`, `tests/package/vision_main.cpp`,
  `tests/common/model_text.hpp`, a rewritten `tools/layering/check_layering.py` with
  `tools/tests/test_check_layering.py`. The C++ of this work survives the move; its three CMake
  files and the checker are rewritten by this plan.

## 2. Measured

### 2.1 Module include graph

Counted: `#include <SushiAI/<module>/...>` in `include/SushiAI` and `src`, by module pair, as
header includes / source includes. Quoted includes cross a module in the two umbrellas only.

| Module | Includes |
| --- | --- |
| `core` | none |
| `graph` | core 40/30 |
| `config` | core 5/0 |
| `autograd` | core 8/0, graph 5/0 |
| `optim` | core 10/21, graph 1/0, config 1/1 |
| `data` | core 8/8, config 1/0 |
| `eager` | core 1/5, graph 4/0, autograd 1/0 |
| `nn` | core 61/26, graph 47/4, config 5/2 |
| `io` | core 9/9, nn 2/2 |
| `train` | core 15/23, graph 27/15, nn 8/12, config 1/1, data 2/0, optim 2/0, autograd 0/1 |
| `vision/imaging` | core 0/16 |
| `vision/detect` | core 0/14 |
| `vision/blocks` | core 10/5, graph 6/3, nn 26/1, config 0/5 |
| `vision/datasets` | core 2/2, data 1/1, config 1/0 |

Cycles between modules: none (Tarjan over the 14 nodes finds no component larger than one).
Edges up the tier order: none. No vision module includes another vision module.

What this forces: nothing. Static libraries per module would link, and no boundary has to move.
The choice between STATIC and OBJECT is made on the export set, question Q1. The graph is an
include graph; a symbol used through a hand-written `extern` would not show, and build B1 is
what confirms the link edges.

Third parties: `nlohmann` is included by `src/config/value.cpp` alone. No file includes SYCL
directly. `SushiBLAS` headers are included by 10 modules, `SushiRuntime` by 13 (not `imaging`);
14 `detect` headers include `SushiRuntime`.

### 2.2 Files that belong to no module

| File | Today | Includes |
| --- | --- | --- |
| `include/SushiAI/SushiAI.h` | umbrella, 112 quoted includes | every core and layers header |
| `include/SushiAI/version.hpp`, `src/version.cpp` | read by the root `CMakeLists.txt` version check | nothing |
| `include/SushiAI/vision/vision.h` | umbrella, 43 quoted includes | every vision header |
| `include/SushiAI/vision/registration.hpp`, `src/vision/registration.cpp` | `register_nodes`, `register_datasets` | core, data, nn, blocks, datasets |

Private headers: `src/autograd/differentiator.hpp`, `src/vision/blocks/block_attributes.hpp`,
`src/vision/imaging/bilinear.hpp`; each is included only by sources of its own folder. READMEs
exist for 4 of 14 modules, under `include/SushiAI/{eager,nn,vision/detect,vision/imaging}/`.
`tests/common` holds 7 files: `test_utils.hpp` includes the umbrella, `reference_fixture.hpp`
includes `io`, `model_text.hpp` includes `config`, `test_status.hpp` and `test_throws.hpp`
include `core`. `tests/fixtures` holds 3 files and `yolo26/` (6 files).

### 2.3 Tests

111 test files: 55 in `tests/functional/unit`, 17 in `integration`, 10 in `regression`, 29 in
`tests/vision`. The rule: a test moves into module M when it is a unit or `tests/vision` file,
its direct `<SushiAI/...>` includes have exactly one maximal module M under the dependencies of
2.1, and no helper it includes reaches above M's tier.

| Destination | Count |
| --- | --- |
| core 4, graph 7, autograd 7, eager 3, data 1, io 1, nn 22, train 2, version 1 | 48 |
| vision: imaging 17, detect 5, blocks 5, registration 2 | 29 |
| root `tests/unit` | 7 |
| root `tests/integration` 17, `tests/regression` 10 | 27 |

The 7 that stay in `tests/unit`: `test_backward`, `test_conv2d`, `test_diagnostics`,
`test_layer_registry`, `test_loss_roots` (maxima `autograd` and `nn`), `test_loss_scaling` (no
direct module include), `test_seam` (a `core` test that includes `test_utils.hpp`, which
includes the layers umbrella). 26 of the 27 integration and regression files reach the umbrella
directly or through `test_utils.hpp`, so their includes do not bound what they exercise and all
27 stay. `optim`, `config` and `vision/datasets` own no test file; `test_config.cpp` builds
layer stacks and lands in `nn`, `test_datasets_mnist.cpp` calls `register_datasets` and lands in
`registration`.

Test binaries, three shapes:

| Shape | Cost |
| --- | --- |
| Two binaries as today, sources collected from the module `tests/` folders | No change to `sa run sushiai_tests`, the CLI, CI or the 35 s run; 2 links. A module's tests cannot be built alone |
| One binary per tier plus root | 4 or more links, 4 DLL copies, 4 discoveries; `sa run` names change; `test_main.cpp` compiled 4 times |
| One binary per module | 14 or more links and discoveries; each needs the objects of its dependency closure, which the two-archive shape of Q1 does not hand out |

Recommended: the first. A module lists its test files in its own `CMakeLists.txt`; the function
appends them to the binary of its library.

### 2.4 What names a path

| Place | What it names |
| --- | --- |
| `CMakeLists.txt` | `add_subdirectory(src, examples, apps, tests)`, the version check on `include/SushiAI/version.hpp` |
| `src/CMakeLists.txt` (277 lines) | 83 sources, `${PROJECT_SOURCE_DIR}/include`, `install(DIRECTORY include/)`, the umbrella check as `"<header>"` strings |
| `src/vision/CMakeLists.txt` | 40 sources, the vision umbrella check |
| `tests/CMakeLists.txt`, `tests/vision/CMakeLists.txt` | 82 and 29 file paths, `fixtures`, `../common/test_main.cpp` |
| `tests/package/CMakeLists.txt` | messages naming `src/config/` and `src/CMakeLists.txt` |
| `cli/sushiai/services/project.py` | `root / "tests" / "package"` (:427), `root / "Doxyfile"` (:564), two messages naming `src/CMakeLists.txt` and `include/SushiAI/SushiAI.h` (:452) |
| `cli/sushiai/cli.py` | three epilogs with `examples/heavy_mlp/...`; targets are found by name in the build tree, not by path |
| `cli/sushistack.deps.toml` | a description naming `src/config/*.cpp` |
| `Doxyfile` | `INPUT = docs/reference/API_REFERENCE.md include/SushiAI`, patterns `*.hpp *.h *.md` |
| `.github/workflows/ci.yml` | 5 comment lines; the checker commands take `.` and stay valid |
| `tools/contract/check_contract.py` | `K_SOURCE_ROOTS = ("include", "src", "apps", "examples")`, `K_LIBRARY_ROOTS = ("include", "src")` |
| `tools/contract/site_classes.tsv` | 401 rows whose first column is a path under `include/` or `src/` |
| `tools/layering/check_layering.py` | `K_LIBRARY_ROOTS`, `K_SCOPED_ROOTS`, one module per folder under `include/SushiAI` |
| `tools/documentation/check_docs_layout.py` | `rule_module_readmes` wakes when `modules/` exists and asks a `README.md` of every `modules/<tier>/<module>/` |
| `check_source_comments.py`, `check_changelog.py`, `write_license_block.py` | no source path; nothing to change |
| `tools/naming/rename_map.tsv` | 323 old paths; the record of a finished rename, left as it is |

Path mentions in prose, lines per file: `ARCHITECTURE.md` 28, `REMAINING_WORK.md` 18,
`CONTRIBUTING.md` 16, `CLI_GUIDE.md` 16, `CHANGELOG.md` 15, `KNOWN_ISSUES.md` 10, `ROADMAP.md` 9,
`FAQ.md` 5, `TUTORIAL.md` 5, `SECURITY.md` 4, `MODEL_FILE.md` 4, `GLOSSARY.md` 2,
`docs/README.md` 2, `INTEGRATION.md` 1, `DOMAIN_MODULES.md` 1; root `README.md` 8, `NOTICE.md` 2;
`tests/functional/unit/README.md` 108, the other test, app and example READMEs 38 together.
`docs/agent` and `docs/archive` hold 1397 and are records; no task edits them. Changelog
entries are records too. Source comments: 70 lines in 26 headers, 34 in 22 sources, 411 in 47
files under `tests`, 25 in 5 files under `apps` and `examples`.

`docs/guides/CLI_GUIDE.md` belongs to another session and no task edits it. What becomes false
in it: line 162 (`src/CMakeLists.txt`, `include/SushiAI/**/*.hpp`), 305 and 317
(`examples/demo_mlp.cpp`), 328 to 331, 338 and 397 (`examples/heavy_mlp`, `examples/mnist_cnn`),
345 (`apps/train.cpp`), 405 and 433 (`include/SushiAI/nn/layer_registry.hpp`,
`include/SushiAI/data/dataset.hpp`), 483 to 485 (root `Doxyfile`, if Q5 moves it), 494 (scope
`include/SushiAI`). Line 224, `sa run sushiai_tests`, stays true under 2.3.

Root against the skill's list: `Doxyfile` is forbidden there; `.gitattributes` is missing;
`sushi-module.toml` is not on the list and both siblings carry it, so it stays.

### 2.5 SushiEngine's `Module.cmake`

`sushiengine_add_module(NAME LAYER TYPE SOURCES PUBLIC_DEPENDS PRIVATE_DEPENDS EXTERNAL_PUBLIC
EXTERNAL_PRIVATE CONFIGURED_HEADERS)` checks the module against the table in
`EngineLayers.cmake`, refuses an upward edge unless a dated exception lists it, creates
`sushiengine_<name>` as STATIC (INTERFACE without sources), adds `include/` PUBLIC and `source/`
PRIVATE, and links the dependencies. It installs and exports nothing: SushiEngine ships
applications, not a package. Its tests all sit in root `tests/`, and its umbrella sits in
`engine/include/`, a folder the `modules/` checkers would read as a tier.

SushiAI takes the argument checks, the table lookup, the edge check and the include roots. It
adds what a package needs:

- `sushiai_apply_sycl(<target> SOURCES ...)` and `sushiai_apply_build_policy` on every module
  that compiles, and the SYCL link options on the two libraries.
- `BUILD_INTERFACE` on every module include root, one `install(DIRECTORY include/)` per module,
  skipped for a tier whose library is not built.
- The two libraries in export set `SushiAITargets` under the names they have today.
- The umbrella check per library: every header under the include roots of its modules appears
  in the umbrella as `<SushiAI/...>`.
- `nlohmann_json` on `config` only, as include directories and definitions taken as values, so
  the name stays out of the exported link interface as `tests/package` asserts.

## 3. The contract every task follows

- No include path, namespace, public symbol or target name a consumer sees changes.
- The orchestrator does every `git mv` and writes every CMake file. A task edits inside files.
- A task writes only its listed files and returns its report as its reply.
- Nobody but the owner builds. A task proves its edit with the check its row names.
- No task edits `docs/guides/CLI_GUIDE.md`, `docs/agent/**` or `docs/archive/**`.
- A file is staged by path. `CLI_GUIDE.md` is never staged.

## 4. Target tree

```
README.md LICENSE NOTICE.md COMMERCIAL.md CMakeLists.txt AGENTS.md CLAUDE.md sushi-module.toml
.clang-format .editorconfig .gitignore .gitattributes
cmake/  cli/  tools/  modules/  tests/  applications/  docs/
modules/CMakeLists.txt                  the two libraries, the tier folders
modules/<tier>/CMakeLists.txt           one add_subdirectory line per module
modules/<tier>/<module>/{README.md, CMakeLists.txt, include/SushiAI/..., source/, tests/}
tests/{CMakeLists.txt, common/, fixtures/, package/, unit/, integration/, regression/}
applications/{CMakeLists.txt, programs/, examples/}          (Q4)
```

Files per module: public headers / sources / private headers / tests. Four modules are new and
hold the files of 2.2 (Q2).

| Module | Files | Public include root holds |
| --- | --- | --- |
| `core/core` | 9 / 4 / 0 / 4 | `SushiAI/core/` |
| `core/version` (new) | 1 / 1 / 0 / 1 | `SushiAI/version.hpp` |
| `core/graph` | 27 / 20 / 0 / 7 | `SushiAI/graph/` with `lowering/` |
| `core/config` | 4 / 3 / 0 / 0 | `SushiAI/config/` |
| `core/autograd` | 13 / 10 / 1 / 7 | `SushiAI/autograd/` with `rules/` |
| `core/optim` | 6 / 6 / 0 / 0 | `SushiAI/optim/` |
| `core/data` | 3 / 3 / 0 / 1 | `SushiAI/data/` |
| `core/eager` | 7 / 3 / 0 / 3 | `SushiAI/eager/` |
| `layers/nn` | 31 / 23 / 0 / 22 | `SushiAI/nn/` |
| `layers/io` | 3 / 3 / 0 / 1 | `SushiAI/io/` |
| `layers/train` | 8 / 7 / 0 / 2 | `SushiAI/train/` |
| `layers/umbrella` (new) | 1 / 0 / 0 / 0 | `SushiAI/SushiAI.h` |
| `vision/imaging` | 19 / 19 / 1 / 17 | `SushiAI/vision/imaging/` |
| `vision/detect` | 17 / 14 / 0 / 5 | `SushiAI/vision/detect/` |
| `vision/blocks` | 5 / 5 / 1 / 5 | `SushiAI/vision/blocks/` |
| `vision/datasets` | 1 / 1 / 0 / 0 | `SushiAI/vision/datasets/` |
| `vision/registration` (new) | 1 / 1 / 0 / 2 | `SushiAI/vision/registration.hpp` |
| `vision/umbrella` (new) | 1 / 0 / 0 / 0 | `SushiAI/vision/vision.h` |

## 5. CMake interface

`cmake/SushiAILayers.cmake`, the tier table. A module of a domain tier is keyed
`<tier>/<module>`, so the next domain may have its own `blocks`.

```cmake
set(SUSHIAI_TIER_ORDER    core layers vision)
set(SUSHIAI_DOMAIN_TIERS  vision)
set(SUSHIAI_TIER_LIBRARY  core sushiai  layers sushiai  vision sushiai_vision)
set(SUSHIAI_MODULE_TIERS
    core core  version core  graph core  config core  autograd core  optim core  data core
    eager core  nn layers  io layers  train layers  umbrella layers
    vision/imaging vision  vision/detect vision  vision/blocks vision  vision/datasets vision
    vision/registration vision  vision/umbrella vision)
set(SUSHIAI_TIER_EXCEPTIONS "")   # consumer dependency date reason
```

Every domain tier ranks one above `layers`; an edge between two different domain tiers is
refused whatever the order says. A new domain is one word in the first three lists and its
module rows.

`cmake/SushiAIModule.cmake`:

```cmake
sushiai_add_module(
    NAME <module>  TIER <tier>
    [TYPE OBJECT|INTERFACE]            # OBJECT when SOURCES are given, else INTERFACE
    [SOURCES source/<file>.cpp ...]
    [PUBLIC_DEPENDS <key> ...]  [PRIVATE_DEPENDS <key> ...]     # keys of SUSHIAI_MODULE_TIERS
    [EXTERNAL_PUBLIC <target> ...]
    [EXTERNAL_PRIVATE_INCLUDES <target> ...]   # include directories and definitions as values
    [TESTS tests/<file>.cpp ...])

sushiai_add_library(
    NAME <target>  EXPORT_NAME <name>  TIERS <tier> ...
    UMBRELLA <module key> <header below its include root>
    [DEPENDS <library target> ...])
```

`sushiai_add_module` makes target `sushiai_<key with _ for />`, checks NAME and TIER against
the table and each dependency edge against the order, sets the include roots, applies the build
policy and SYCL, installs the headers, and appends the objects, the include root and the test
files to global properties of the tier's library. `sushiai_add_library` makes the STATIC archive
from those objects, links `SushiRuntime::SushiRuntime` and `SushiBLAS::sushiblas` PUBLIC, runs
the umbrella check, and installs into `SushiAITargets`. `modules/CMakeLists.txt` calls it twice:
`sushiai` (EXPORT_NAME `sushiai`, tiers `core layers`) and, under `SA_BUILD_VISION`,
`sushiai_vision` (EXPORT_NAME `vision`, tier `vision`, DEPENDS `sushiai`).

Declared dependencies, from 2.1 (header include: PUBLIC; source only: PRIVATE):

| Module | PUBLIC | PRIVATE or external |
| --- | --- | --- |
| `core` | | EXTERNAL_PUBLIC SushiRuntime, SushiBLAS |
| `version` | | |
| `graph`, `config` | core | `config`: EXTERNAL_PRIVATE_INCLUDES `nlohmann_json::nlohmann_json` |
| `autograd` | core graph | |
| `optim` | core graph config | |
| `data` | core config | |
| `eager` | core graph autograd | |
| `nn` | core graph config | |
| `io` | core nn | |
| `train` | core graph config data optim nn | PRIVATE autograd |
| `umbrella` | every core and layers module | INTERFACE |
| `vision/imaging` | | PRIVATE core |
| `vision/detect` | | PRIVATE core; EXTERNAL_PUBLIC SushiRuntime |
| `vision/blocks` | core graph nn | PRIVATE config |
| `vision/datasets` | core config data | |
| `vision/registration` | core | PRIVATE data nn vision/blocks vision/datasets |
| `vision/umbrella` | the five other vision modules | INTERFACE |

A module can include only what it declares, because each module has its own include root. The
test binaries link the libraries and see every root; the layering checker covers them.

## 6. Waves

Dispatch rows: `haiku medium` for a spec-complete edit of one or two files, `haiku high` for a
spec-complete edit of many files, `opus low` for integration or judgment. Every dispatch opens
with the seven lines of the owner's "Delegation" section and sets model and effort.

### Wave 0: the starting point (orchestrator)

- O0.1 Create branch `refactor/repository-layout` from `main`.
- O0.2 Stage by path and commit the phase 3 and 4 work as it stands, marked unbuilt in the
  message (Q6). `docs/guides/CLI_GUIDE.md` and `docs/agent/2026_10_05_ESTATE_AUDIT/` stay out.

### Wave 1: the tree and the build files (orchestrator)

O1.1 Write `cmake/SushiAILayers.cmake` and `cmake/SushiAIModule.cmake` as in section 5.

O1.2 Move with `git mv`, one script in the scratchpad, rules applied in this order. `<t>` is the
tier of module `<m>` in section 4.

| # | From | To |
| --- | --- | --- |
| 1 | `include/SushiAI/vision/<m>/README.md` | `modules/vision/<m>/README.md` |
| 2 | `include/SushiAI/vision/<m>/**` | `modules/vision/<m>/include/SushiAI/vision/<m>/**` |
| 3 | `src/vision/<m>/**` | `modules/vision/<m>/source/**` |
| 4 | `include/SushiAI/vision/registration.hpp`, `src/vision/registration.cpp` | `modules/vision/registration/include/SushiAI/vision/registration.hpp`, `.../source/registration.cpp` |
| 5 | `include/SushiAI/vision/vision.h` | `modules/vision/umbrella/include/SushiAI/vision/vision.h` |
| 6 | `include/SushiAI/<m>/README.md` | `modules/<t>/<m>/README.md` |
| 7 | `include/SushiAI/<m>/**` | `modules/<t>/<m>/include/SushiAI/<m>/**` |
| 8 | `src/<m>/**` | `modules/<t>/<m>/source/**` |
| 9 | `include/SushiAI/version.hpp`, `src/version.cpp` | `modules/core/version/include/SushiAI/version.hpp`, `.../source/version.cpp` |
| 10 | `include/SushiAI/SushiAI.h` | `modules/layers/umbrella/include/SushiAI/SushiAI.h` |
| 11 | `tests/vision/test_imaging_*`, `test_detect_*`, `test_block_*` | `modules/vision/{imaging,detect,blocks}/tests/` |
| 12 | `tests/vision/test_registration.cpp`, `test_datasets_mnist.cpp` | `modules/vision/registration/tests/` |
| 13 | each of the 48 unit files of 2.3 | `modules/<t>/<m>/tests/` |
| 14 | `tests/functional/{unit,integration,regression}/**` that remain | `tests/{unit,integration,regression}/**` |
| 15 | `apps/**` | `applications/programs/**`, sources under `source/` |
| 16 | `examples/**` | `applications/examples/**`, the two `.cpp` under `source/` |
| 17 | `Doxyfile` | `.config/doxygen/Doxyfile` (Q5) |

File names do not change. Rule 13 takes its list from the rule of 2.3, printed by the script
and pasted into the report. After the moves `include/`, `src/`, `apps/`, `examples/`,
`tests/functional/` and `tests/vision/` do not exist.

O1.3 Write the build files: root `CMakeLists.txt` (subdirectories `modules`, `applications`,
`tests`; version check path), `modules/CMakeLists.txt`, three tier lists, 18 module files,
`applications/CMakeLists.txt` with `programs/` and `examples/`, `tests/CMakeLists.txt` (two
binaries from the properties of section 5 plus the 34 root files, include root `tests/`,
`SA_TEST_FIXTURE_DIR`), the messages of `tests/package/CMakeLists.txt`. Delete
`src/CMakeLists.txt`, `src/vision/CMakeLists.txt`, `tests/vision/CMakeLists.txt`,
`apps/CMakeLists.txt`, `examples/CMakeLists.txt`. Each comment of the old files that states a
reason moves to the function or file that now does the thing.

O1.4 Run the move script over column 1 of `tools/contract/site_classes.tsv` (401 rows). Set
`INPUT` in the Doxyfile to `modules` with `EXCLUDE_PATTERNS` `*/source/* */tests/*`. Add
`.gitattributes` after SushiBLAS's.

### Wave 2: the includes that the move breaks (3 tasks, parallel, after wave 1)

| Id | Files | Does | Acceptance | Dispatch |
| --- | --- | --- | --- | --- |
| T2.1 | the two umbrella headers | Rewrites each `#include "<path>"` as `#include <SushiAI/<path>>` (`<SushiAI/vision/<path>>` in `vision.h`) | 112 and 43 angle includes, 0 quoted; a pasted loop shows each resolves under one `modules/*/*/include` | haiku medium |
| T2.2 | every `.cpp` and `.hpp` under `tests/` and `modules/*/*/tests/` that includes `"../common/` or `"../../common/` | Rewrites those includes as `<common/<name>.hpp>`; includes between files of `tests/common` stay | `grep -rn '"\.\./' tests modules` prints nothing; the count of rewritten lines is pasted | haiku high |
| T2.3 | `cli/sushiai/cli.py`, `cli/sushiai/services/project.py`, `cli/sushistack.deps.toml`, `cli/README.md`, the `cli/tests` files that assert those strings | Doxyfile path, the epilogs, the two messages, per the move table (Q7) | `python -m pytest cli/tests -q` pasted, with any test that starts a build named and skipped | haiku high |

Rule 6 of the dispatch block cannot run here: `compile_commands.json` names the old paths until
the owner configures. The greps above stand in, and build B1 is the compile.

**Build B1 (owner).** `sa build`, `sa run sushiai_tests`, `sa run sushiai_vision_tests`,
`sa test`, `sa test --suite package`, then one configure with `SA_BUILD_VISION=OFF`. It
confirms: OBJECT modules take `add_sycl_to_target` and `-fsycl`; every declared dependency of
section 5 is enough to compile its module; the test counts match the last run (1881 and the
vision count); the export set names `SushiAI::sushiai` and `SushiAI::vision` and neither
`nlohmann_json` nor a `sushiai_<module>` target; an install holds no `source/` or `tests/` file.

### Wave 3: checkers, READMEs, comments (9 tasks, after B1 passes)

| Id | Files | Does | Acceptance | Dispatch |
| --- | --- | --- | --- | --- |
| T3.1 | `tools/layering/check_layering.py`, `tools/tests/test_check_layering.py` | Discovers `modules/<tier>/<module>/` as the shared checker does; resolves `<SushiAI/<m>/..>`, `<SushiAI/<domain>/<m>/..>` and the four root headers to modules; rules `declared_tiers`, `upward_includes`, `cross_domain_includes`, `private_reach`; root `tests/` and `applications/` as scopes above every tier | unittest output pasted; the checker on `.` reports 0 findings or each finding is listed | opus low |
| T3.2 | `tools/contract/check_contract.py`, `tools/tests/test_check_contract.py`, `tools/contract/README.md` | Roots become `modules` and `applications`; a module's `tests/` is not a library site | unittest output pasted; the checker on `.` matches all 401 rows | opus low |
| T3.3 | `README.md` of the 8 core-tier modules | Writes 7, updates `eager`: owns, depends on (from its `CMakeLists.txt`), public entry points | `check_docs_layout.py --rule module_readmes` pasted | haiku high |
| T3.4 | `README.md` of the 4 layers-tier modules | Writes 3, updates `nn` | as T3.3 | haiku high |
| T3.5 | `README.md` of the 6 vision modules | Writes 4, updates `imaging` and `detect` | as T3.3 | haiku high |
| T3.6 | comments in `modules/core/**` and `modules/layers/**` | Applies the move table to every path a comment names; a comment about a deleted CMake file names its successor from the list in the dispatch | grep for the old roots prints nothing; `check_source_comments.py` pasted | haiku high |
| T3.7 | comments in `modules/vision/**`, `applications/**` and their READMEs | as T3.6 | as T3.6 | haiku high |
| T3.8 | comments in `tests/**`, `tests/*/README.md`, comments of `tests/CMakeLists.txt` excluded | as T3.6; `tests/unit/README.md` loses the rows of the 48 moved files | as T3.6 | haiku high |
| T3.9 | reads everything waves 1 to 3 wrote | Reviews SOLID shape (one function per concern in `SushiAIModule.cmake`, sibling module files shaped the same) and humanizer register of the READMEs, as named items; runs the four checkers and the tool unittests | findings listed with file and line; outputs pasted | opus low |

T3.1 to T3.8 hold disjoint files and run together; T3.9 runs after them.

Orchestrator after the tasks: O3.1 the manual and design (`ARCHITECTURE.md`, `CONTRIBUTING.md`,
`KNOWN_ISSUES.md`, `ROADMAP.md`, `FAQ.md`, `TUTORIAL.md`, `SECURITY.md`, `MODEL_FILE.md`,
`GLOSSARY.md`, `INTEGRATION.md`, `docs/README.md`, root `README.md`, `NOTICE.md`), the status
and paths of `DOMAIN_MODULES.md`, `REMAINING_WORK.md`; O3.2 the comments of `ci.yml` and
`.gitignore`; O3.3 changelog lines, one per scope; O3.4 `REPORT.md`; O3.5 a note to the session
that owns `CLI_GUIDE.md` with the lines of 2.4.

**Build B2 (owner).** `sa build`, `sa test`, `sa test --suite package`, `sa doxygen`. It closes
the work. Two builds in all.

## 7. Commits

One commit per step that builds on its own: the wave 0 commit; wave 1 and wave 2 together after
B1 (the renames with the build files and the include edits, since no part compiles alone);
wave 3 with the documents after B2.

## 8. Questions for the owner

| # | Question | Recommendation |
| --- | --- | --- |
| Q1 | Module target type. STATIC per module exports 16 archives and makes `SushiAI::sushiai` an INTERFACE over them. OBJECT per module keeps two archives and the export set as it is | OBJECT, aggregated by `sushiai_add_library`. If B1 shows `add_sycl_to_target` refuses an OBJECT library, fall back to STATIC and ask again |
| Q2 | Home of the six files of 2.2. An umbrella cannot live in a module it includes above, and `modules/include/` would be read as a tier by two checkers | Four small modules: `core/version`, `layers/umbrella`, `vision/registration`, `vision/umbrella`; 18 modules in all, include paths unchanged |
| Q3 | Shared test support. A quoted `"../../common/x.hpp"` from a module's `tests/` leaves the module, which the shared `private_reach` rule refuses | Keep `tests/common/`, put `tests/` on the include path of both binaries, include as `<common/x.hpp>` |
| Q4 | Shape inside `applications/` | `applications/programs/` (train, inference, the registration files) and `applications/examples/` (the two examples and the three configuration folders), as SushiBLAS has `applications/examples/` |
| Q5 | Home of `Doxyfile`. The skill forbids the root; both siblings keep `.config/doxygen/Doxyfile`; `.gitignore` here says `docs/Doxyfile`; `.config/` is not on the skill's root list | `.config/doxygen/Doxyfile`, with the CLI path changed in T2.3 |
| Q6 | Commit the unbuilt phase 3 and 4 work before the move | Yes, as its own commit on the branch, so the move commit is renames and `git log --follow` holds |
| Q7 | May T2.3 edit `cli/sushiai/*.py`, given another session holds `CLI_GUIDE.md` | Yes, for the Doxyfile path, three epilogs and two messages only |

**Decided on 2026-10-09** by the engineer the owner delegated these to, after his ruling that
the whole repository follows the skill: every question takes its recommendation, except that
the work stays on `main` (another session shares the working tree), so O0.1 is dropped.

Already ruled and not asked: the local `check_layering.py` diverges from the shared copy, whose
include pattern reads `<SushiAI/vision/detect/x.hpp>` as module `vision` and whose module map is
keyed by bare folder name; carrying the change to SushiSkills stays a backlog line.

## 9. Not verified

- Nothing was configured, built or run. The dependency table of section 5 comes from includes.
- Whether `add_sycl_to_target` and `-fsycl` work on OBJECT libraries with either SYCL toolchain.
- Whether an OBJECT module with `$<BUILD_INTERFACE:...>` include roots and PUBLIC links to other
  OBJECT modules leaves the export set clean; B1's package suite decides.
- The `SushiRuntime` DLL copy, `gtest_discover_tests` and the three `add_test` application
  cases were read, not rerun against the new paths.
- `cli/tests` were listed, not read; which of them assert the strings T2.3 changes is unknown.
- The uncommitted phase 3 and 4 C++ was not read line by line; `git diff --stat` and the CMake
  files were.
- Configuration files under `examples/` were not opened for paths that leave their folder.
- `tools/tests/test_check_contract.py` (65 path lines) and `test_naming.py` (36) were counted,
  not read.
- SushiBLAS's `.gitattributes` exists; its content was not read. SushiEngine has none.
- Whether `check_docs_layout.py` accepts links from the manual into `modules/`.
- Test counts per module follow the include rule of 2.3; no test body was read to confirm what
  it exercises, apart from `test_datasets_mnist.cpp`.
