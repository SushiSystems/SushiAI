# Domain modules

**Status:** Open — the module tree is built and passing; the split of `vision/detect` and the module `vision/training` are ruled and being written. Plan and report: `docs/agent/2026_10_09_DOMAIN_MODULES/`

How SushiAI keeps what belongs to one field of use (vision today; language, physics and
reinforcement learning later) apart from the library every field shares, so that a field can be
added, built and removed without an edit to another. The tree follows the `repository-layout`
skill of SushiSkills, as SushiEngine does.

## Rulings

Each was made by the owner on 2026-10-09.

| Subject | Ruling |
| --- | --- |
| Layout | The whole repository follows the `repository-layout` skill: every module is a tree under `modules/<tier>/<module>/` |
| Domains | A domain is a tier. Vision is `modules/vision/`, and the modules it holds are separate modules, as `detect` and `imaging` are |
| Layers | `Conv2d`, `BatchNorm2d`, pooling and the other layers stay in `nn`. A layer is not a field of use |
| Build | A domain is its own library. A consumer that does not use it does not compile it |
| Durability | The split is chosen so that it is not split again when the next domain arrives |
| Tier of `train` and `io` | The layers tier, beside `nn`: both include it, for `ModelState`, `ModelOutputs` and `Trace` |
| `NodeRegistry` | One registry with two `add` overloads, `find_node` and `find_layer` |
| Vision switched off | The programs link a registration file with or without vision; the examples are built only with it |
| Detection | `vision/detect` stays one module, split inside itself: the root holds what both families use (boxes, IoU, suppression, top-k, average precision), `free/` the anchor-free path of YOLO26 (`Vision::Detect::Free`) and `anchor/` the anchor-box path of the Darknet-era models (`Vision::Detect::Anchor`) |
| Detection training | The detection objectives, the host stage and the training step live in a module `vision/training`, which depends on `train` and `vision/detect`; `vision/detect` keeps depending on `core` alone |
| Anchoring | Anchors are derived, assigned to heads and checked against a dataset by the unit, as automatically as sound engineering allows; an imported model keeps the anchors of its file |
| Reference implementations | Nothing is quoted from one. Darknet and Ultralytics are run outside the repository to produce tensors; SushiAI's own implementation is written from the behaviour |

Decided by the engineer the owner delegated them to:

| Subject | Decision |
| --- | --- |
| Include path of a domain's module | `<SushiAI/vision/<module>/x.hpp>`. Module names such as `blocks` and `datasets` repeat in every domain, and the domain in the path keeps them apart. The skill's `<Project>/<module>/` holds for the other tiers |
| CMake target of a module | One target per module, declared by `sushiai_add_module`, as SushiEngine's `sushiengine_add_module` does. The modules are object libraries archived into two libraries, so the package exports two names |
| Files of no module | `version`, the two umbrella headers and vision's registration are small modules of their own |

## The tree

```
modules/<tier>/<module>/
    README.md
    CMakeLists.txt
    include/SushiAI/...      the module's public surface
    source/                  private to the module
    tests/                   the tests of this module alone
tests/                       tests that cross module boundaries; common/, fixtures/, package/
applications/                programs/ (train, inference), examples/
```

## Tiers

A dependency points down the order or sideways inside its own tier, never up and never from one
domain to another.

| Tier | Modules | Knows |
| --- | --- | --- |
| `core` | `core`, `version`, `graph`, `config`, `autograd`, `optim`, `data`, `eager` | No layer type, no domain |
| `layers` | `nn`, `io`, `train`, `umbrella` | The core tier. Layer types, `ModelGraph`, `ModelState`, the steps and the checkpoints over them, the registries' types. No domain |
| `vision` | `imaging`, `detect`, `blocks`, `datasets`, `registration`, `umbrella` | The core and layers tiers. No other domain |

The programs under `applications/` and the tests under root `tests/` may use whatever they link.
The next domain is a sibling folder of `modules/vision/` and one word in the tier table.

Two places hold the tier table and must agree: `cmake/SushiAILayers.cmake`, which the build
reads, and `tools/layering/check_layering.py`, which reads the includes.

## The shape of a domain

Every domain has the same modules, and a module that would be empty is not created. Siblings
that do the same kind of thing are shaped the same, so the fourth domain is laid out by copying
the names of the first.

| Module of `modules/<domain>/` | Holds | Vision's content |
| --- | --- | --- |
| One per kind of data | The data type, its transforms and its file readers | `imaging` |
| One per task | Geometry, assignment, decoding and metrics of one task | `detect`, with `free/` and `anchor/` inside it |
| `training` | The objectives of the domain's tasks and the steps that train them | Not created yet |
| `blocks` | Model blocks built from `nn` layers that one family of models uses | `ConvNorm`, `Bottleneck`, `C3k`, `C3k2`, `SPPF` |
| `datasets` | Dataset types of the domain | MNIST |
| `importers` | Readers of other frameworks' model and weight files | Not created yet; the Darknet importer |
| `registration` | The functions that add the domain's rows to a registry | `register_nodes`, `register_datasets` |
| `umbrella` | The header that includes the library's umbrella and every header of the domain | `SushiAI/vision/vision.h` |

Namespaces: `SushiAI::Vision::Detect`, `SushiAI::Vision::Imaging`, `SushiAI::Vision::Blocks`,
`SushiAI::Vision::Datasets`. A second task (segmentation, pose) is a sibling of `detect`, not a
growth of it.

## Registration

A tier that names what a higher tier owns cannot be built without it, so no table in `nn` or
`data` lists a type of a domain.

- A registry is a value: `NN::NodeRegistry`, with `add`, `find_node` and `find_layer`, and
  `Data::DatasetRegistry`, with `add` and `find`. `build_model` and the dataset factory take
  the registry as a parameter.
- Each module states its own rows. `NN::register_layers(registry)` adds the layer types,
  `Data::register_datasets(registry)` the synthetic sets, `Vision::register_nodes(registry)`
  the blocks and `Vision::register_datasets(registry)` MNIST.
- A program composes them: it builds a registry, calls the functions of what it links, and
  passes it on. There is no global registry and no static initialiser, since a static library's
  unreferenced object files are dropped by the linker and a registration in one would vanish
  without an error.
- A model file that names a type no registered module has fails with a status that lists the
  names the registry holds.

## Build

| Library | Alias | Tiers | Links | Umbrella |
| --- | --- | --- | --- | --- |
| `sushiai` | `SushiAI::sushiai` | `core`, `layers` | SushiBLAS, SushiRuntime | `SushiAI/SushiAI.h` |
| `sushiai_vision` | `SushiAI::vision` | `vision` | `SushiAI::sushiai` | `SushiAI/vision/vision.h` |

`sushiai_add_module(NAME TIER SOURCES PUBLIC_DEPENDS PRIVATE_DEPENDS ... TESTS)` in
`cmake/SushiAIModule.cmake` declares a module. A dependency is a key of the tier table, and an
edge that points up the order or from one domain to another stops the configure, so the rule is
kept by the build and not only by the checker. Each module has its own include root, so a
module can include only what it declares. `sushiai_add_library` archives the objects of its
tiers, checks that the umbrella names every header under their include roots, and installs the
library into the export set `SushiAITargets`.

`SA_BUILD_VISION` defaults to `ON` at the top level and to `OFF` under a consumer, as
SushiBLAS's `SB_BUILD_TESTS` does. Each library has its own test binary, `sushiai_tests` and
`sushiai_vision_tests`; a module lists its test files in its own `CMakeLists.txt`, and the
binary of its library collects them.

## Open

- What a domain may add to the core tier. An operation, a gradient rule or a lowering a domain
  needs is written in the core tier, in its operation's domain file, because the IR is one; a
  domain adds layers and algorithms, not operations.
- The layering checker diverges from the shared copy in SushiSkills, which reads
  `<SushiAI/vision/detect/x.hpp>` as a module named `vision`. Carrying the domain-tier key to
  the shared copy is a line of [`REMAINING_WORK.md`](REMAINING_WORK.md).
