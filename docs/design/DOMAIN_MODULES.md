# Domain modules

**Status:** Open — phase 2 of 4, phase 1 built and passing; the plan is `docs/agent/2026_10_09_DOMAIN_MODULES/PLAN.md`

How SushiAI keeps what belongs to one field of use (vision today; language, physics and
reinforcement learning later) apart from the library every field shares, so that a field can be
added, built and removed without an edit to another.

## Rulings

Each was made by the owner on 2026-10-09.

| Subject | Ruling |
| --- | --- |
| Top folder | Vision gets a top folder, `vision/`. The modules it holds stay separate modules inside it, as `detect` and `imaging` are separate today |
| Layers | `Conv2d`, `BatchNorm2d`, pooling and the other layers stay in `nn`. A layer is not a field of use |
| Build | A domain is its own library target. A consumer that does not use it does not compile it |
| Durability | The split is chosen so that it is not split again when the next domain arrives |
| Tier of `train` and `io` | Tier 2, beside `nn`: both include it in 14 files, for `ModelState`, `ModelOutputs` and `Trace` |
| `NodeRegistry` | One registry with two `add` overloads, `find_node` and `find_layer` |
| Vision switched off | The applications link a registration file with or without vision; the examples are built only with it |
| Layering checker | Rewritten in this repository's copy; moving it to SushiSkills is a backlog line |

## Tiers

An include points down or sideways inside its own tier's module, never up and never across two
domains.

| Tier | Modules | Knows |
| --- | --- | --- |
| 1. Core | `core`, `graph`, `autograd`, `optim`, `config`, `data`, `eager` | No layer type, no domain |
| 2. Layers | `nn`, `train`, `io` | Tier 1. Layer types, `ModelGraph`, `ModelState`, the steps and the checkpoints over them, the registries' types. No domain |
| 3. Domains | `vision`, later `language`, `physics`, `reinforcement` | Tiers 1 and 2. No other domain |
| 4. Programs | `apps`, `examples` | Whatever they link |

`data` is tier 1 for what is not a domain's (`Dataset`, the synthetic sets) and loses what is
(MNIST goes to vision).

## The shape of a domain

Every domain has the same folders, and a folder that would be empty is not created. Siblings that
do the same kind of thing are shaped the same, so the fourth domain is laid out by copying the
names of the first.

| Folder under `<domain>/` | Holds | Vision's content today |
| --- | --- | --- |
| A module per kind of data | The data type, its transforms and its file readers | `imaging/` |
| A module per task | Geometry, assignment, loss terms, decoding and metrics of one task | `detect/` |
| `blocks/` | Model blocks built from `nn` layers that one family of models uses | `ConvNorm`, `Bottleneck`, `C3k`, `C3k2`, `SPPF`, now under `nn/blocks/` |
| `datasets/` | Dataset types of the domain | MNIST, now in `data/dataset.hpp` |
| `importers/` | Readers of other frameworks' model and weight files | Nothing yet; the Darknet importer |
| `registration.hpp` | The one function that adds the domain's rows to a registry | Nothing yet |

Paths: `include/SushiAI/vision/<module>/`, `src/vision/<module>/`, `tests/vision/`. Namespaces:
`SushiAI::Vision::Detect`, `SushiAI::Vision::Imaging`, `SushiAI::Vision::Blocks`. A second task
(segmentation, pose) is a sibling of `detect/`, not a growth of it.

## Registration

Today `nn/node_registry.hpp` and `nn/layer_registry.hpp` hold `constexpr` tables that name every
type, the YOLO blocks among them, and `data/dataset.hpp` holds one that names MNIST. A tier that
names what a higher tier owns cannot be built without it, so the tables are inverted.

- A registry is a value: `NN::NodeRegistry`, with `add`, `find_node` and `find_layer`, and
  `Data::DatasetRegistry`, with `add` and `find`. `build_model` and the dataset factory take the registry as a parameter.
- Each module states its own rows. `NN::register_layers(registry)` adds the layer types;
  `Vision::register_nodes(registry)` adds the blocks; `Vision::register_datasets(registry)` adds
  MNIST.
- A program composes them: it builds a registry, calls the functions of what it links, and
  passes it on. There is no global registry and no static initialiser, since a static library's
  unreferenced object files are dropped by the linker and a registration in one would vanish
  without an error.
- A model file that names a type no registered module has fails with the status it fails with
  today, which lists the names the registry holds.

## Build

| Target | Alias | Holds | Links |
| --- | --- | --- | --- |
| `sushiai` | `SushiAI::sushiai` | Tiers 1 and 2 | SushiBLAS, SushiRuntime |
| `sushiai_vision` | `SushiAI::vision` | `vision/` | `SushiAI::sushiai` |

`SA_BUILD_VISION` defaults to `ON` at the top level and to `OFF` under a consumer, as SushiBLAS's
`SB_BUILD_TESTS` does. Each target has its own umbrella header (`SushiAI/SushiAI.h`,
`SushiAI/vision/vision.h`), its own completeness check at configure time and its own test binary
(`sushiai_tests`, `sushiai_vision_tests`), so a domain's suite runs alone.

`tools/layering/check_layering.py` is given the tiers above and fails on an include that points
up or across domains. It is the check that keeps the split; today it declares nothing.

## Phases

| # | Phase | Ends when |
| --- | --- | --- |
| 1 | Registries become values | `build_model` and the dataset factory take a registry; the `constexpr` tables are gone; the suite passes |
| 2 | Vision moves | `detect`, `imaging`, the blocks and MNIST are under `vision/` with their namespaces; every include and test follows |
| 3 | The second target | `sushiai_vision` builds and links alone, with its umbrella and its test binary; `sushiai` builds with `SA_BUILD_VISION=OFF` |
| 4 | The layering check | `check_layering.py` declares the tiers and passes |

Phase 2 breaks every include path and namespace of `detect` and `imaging`; no tag exists, so no
consumer is broken. YOLO26's remaining units and the Darknet-era work are written after phase 3,
into `vision/`.

## Open

- Whether the detection objective and the detection trainer step live in `vision/detect/` or in
  a `vision/training/` module beside it is decided when stage 5 of
  [`PROGRAMME.md`](PROGRAMME.md) designs them.
- What a domain may add to tier 1. An operation, a gradient rule or a lowering a domain needs is
  written in tier 1, in its operation's domain file, because the IR is one; a domain adds layers
  and algorithms, not operations.
