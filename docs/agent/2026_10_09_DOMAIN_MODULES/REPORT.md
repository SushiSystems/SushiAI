# Domain modules: report

Covers [`PLAN.md`](PLAN.md), phase by phase.

## Phase 1: registries become values

Written on 2026-10-09 by the plan's author and workflow `wf_c016951e-52f`.

```
dispatch: plan of the four phases — opus low, passed, rung 1
dispatch: T1.2 — haiku medium, 16 turns, passed, rung 1
dispatch: T1.1 — opus low, 8 turns, passed, rung 1
dispatch: T1.3 — haiku high, 21 turns, passed, rung 1
dispatch: review wave 1 — opus low, 9 turns, passed, rung 1
dispatch: T1.7 — opus low, 8 turns, passed, rung 1
dispatch: T1.6 — haiku medium, 17 turns, passed, rung 1
dispatch: T1.4 — opus low, 12 turns, passed, rung 1
dispatch: T1.5 — haiku high, 50 turns, passed, rung 1
dispatch: review wave 2 — opus low, 12 turns, passed, rung 1
```

The reviewers repaired `NodeRegistry::add`, which appended outside `Detail::guard`, two licence
boxes that named another file, and missing `@brief` lines. The wave-2 reviewer searched
`include`, `src`, `tests`, `apps` and `examples` and found no caller without a registry.

The orchestrator added `nn/blocks/registration.cpp` and `data/dataset_registry.cpp` to
`src/CMakeLists.txt`, their headers to `SushiAI.h`, `test_dataset_registry.cpp` to the test
list, `registries.cpp` to both application targets, and the row of the `SA_THROW_IF` site
`build_layer_stack` gained for an unknown layer type.

Build B1, 2026-10-09, SushiBLAS `494ff8e`, RTX 3080 Ti: no build error;
`sa run sushiai_tests -- --gtest_brief=1` passed 1881 tests and skipped 1 on the first run;
`sushiai_train` and `sushiai_inference` each printed `compiles: 1` on the tiny fixtures.

Open after phase 1:

- `make_layer_registry` is copied in four test files; it belongs under `tests/common`.
- The join that lists type names exists in `node_registry.cpp`, `layer_registry.cpp` and
  `dataset_registry.cpp`.
- The READMEs of `apps`, `nn` and two test folders name the deleted tables.

## Phase 2: vision moves

The orchestrator moved the folders with `git mv` and repathed the build lists, the umbrella
header and 131 rows of `site_classes.tsv`; workflow `wf_0c6bf947-1e0` edited the moved files.

```
dispatch: T2.1 — haiku high, 14 turns, passed, rung 1
dispatch: T2.2 — haiku high, 24 turns, passed, rung 1
dispatch: T2.3 — opus low, 12 turns, passed, rung 1
dispatch: T2.4 — opus low, 10 turns, passed, rung 1
dispatch: T2.5 — haiku medium, 17 turns, passed, rung 1
dispatch: review step a — opus low, 16 turns, passed, rung 1
dispatch: T2.10 — haiku medium, 12 turns, passed, rung 1
dispatch: T2.7 — haiku high, 22 turns, passed, rung 1
dispatch: T2.9 — haiku high, 22 turns, passed, rung 1
dispatch: T2.6b — haiku high, 25 turns, passed, rung 1
dispatch: T2.6a — haiku high, 36 turns, passed, rung 1
dispatch: T2.8 — opus low, 19 turns, passed, rung 1
dispatch: review step b — opus low, 23 turns, passed, rung 1
```

The step-b reviewer found that `src/nn/batch_norm2d.cpp` included the blocks' private
`block_attributes.hpp`, which would have made `nn` depend on `vision`. The orchestrator moved
the row readers both use to `include/SushiAI/nn/node_attributes.hpp` and
`src/nn/node_attributes.cpp` (`NN::Detail`); `src/vision/blocks/block_attributes.hpp` keeps the
three block defaults and names the readers with using-declarations.

Build B2, 2026-10-09, same machine and SushiBLAS commit: no build error; `sushiai_tests`
passed 1885 tests and skipped 1 on the first run; `sushiai_train`, `sushiai_inference` and
`sushiai_demo_mlp` each printed `compiles: 1`.

Open after phase 2:

- Nine helpers are copied between `test_node_registry.cpp` and `tests/vision/test_registration.cpp`.
- The READMEs and three documents name the old paths; `tests/vision/` has no README.

## Phases 3 and 4, and the move to the repository layout

The second target and the layering check were first written over `include/SushiAI/vision/` and
`src/vision/` and committed unbuilt (`fbd2c2a`). The owner then ruled that the whole repository
follows the `repository-layout` skill, as SushiEngine does; the plan was rewritten and the tree
moved the same day.

```
dispatch: domains at link time and the package consumer — haiku high, passed, rung 1
dispatch: layering checker over the first tree — opus low, passed, rung 1
dispatch: one home for the registry test helpers — opus low, passed, rung 1
dispatch: plan of the repository layout — opus low, passed, rung 1
dispatch: T3.2 — opus low, 10 turns, passed, rung 1
dispatch: T3.6 — haiku high, 20 turns, passed, rung 1
dispatch: T2.3 — haiku high, 16 turns, passed, rung 1
dispatch: T3.1 — opus low, 9 turns, passed, rung 1
dispatch: T3.4 — haiku high, 29 turns, passed, rung 1
dispatch: T3.7 — haiku high, 24 turns, passed, rung 1
dispatch: T3.5 — haiku high, 28 turns, passed, rung 1
dispatch: T3.3 — haiku high, 48 turns, passed, rung 1
dispatch: T3.8 — haiku high, 41 turns, passed, rung 1
dispatch: T3.9 review — opus low, 8 turns, passed, rung 1
dispatch: manual and root documents after the move — haiku high, passed, rung 1
```

The orchestrator moved 423 files with `git mv` (421 show as renames), wrote
`cmake/SushiAILayers.cmake`, `cmake/SushiAIModule.cmake` and the 26 `CMakeLists.txt` files of
`modules/`, `applications/` and `tests/`, turned the two umbrellas to angle includes (112 and 44
lines), the 146 quoted `common/` includes of the tests to `<common/...>`, and repathed the 401
rows of `site_classes.tsv`.

Measured before the move: no cycle between the 14 modules, no edge up the tier order, no vision
module that includes another. No module boundary moved.

Builds, 2026-10-09, RTX 3080 Ti, SushiBLAS working tree at `83aaacf`:

| Run | Result | Repair |
| --- | --- | --- |
| 1 | Configure stopped: the module target `sushiai_train` had the name of the program | Module targets are `sushiai_module_<key>` |
| 2 | Built. `sushiai_tests` 1274 passed; `sushiai_vision_tests` 611 passed, 1 skipped; the three programs printed `compiles: 1` | |
| 3 | `sa test --suite package`: SushiAI installed both libraries and its headers; the consumer failed on `SushiBLAS/core/common.hpp`, which SushiBLAS's install does not ship since its own move to `modules/` | None here; reported to the SushiBLAS session |
| 4 | `SA_BUILD_VISION=OFF`: built, `sushiai_tests` 1274 passed, `sushiai_train` ran; `build.ninja` names no vision file | |

1274 and 611 sum to the 1885 of the run before the move. Object-library modules take
`add_sycl_to_target` and `-fsycl`, and the exported link interface of `SushiAI::sushiai` names
SushiRuntime and SushiBLAS alone.

The wave-3 reviewer put the three headings of the 14 new module READMEs in one order. The
orchestrator restored the unit notes the rewrite of `eager`'s README had dropped, moved the
notes on the blocks from `nn`'s README to `vision/blocks`, and removed two dependencies `io`
declared and did not use.

Run 5, the same day: SushiBLAS moved its own tree to modules and renamed its public include
paths (`tools/naming/sushiblas_include_paths_0_1_0.tsv` in that repository, 70 rows). SushiAI's
89 include lines in 57 files follow the table. Against that working tree, which SushiBLAS had
not committed, the build passed, `sushiai_tests` passed 1274, `sushiai_vision_tests` passed 611
and skipped 1, both programs printed `compiles: 1`, and `sa test --suite package` built and ran
both consumers. That run also compiled the last edits of wave 3.

Open:

- `sushiai_add_module` derives the key, makes the target, links, registers the tests and
  installs in one function, and looks the domain tier up in three places; the reviewer asks
  for one function per concern.
- `tests/unit/README.md` has 110 prose lines over 100 columns, 26 of them older than this work,
  and keeps the sections of test files that now sit in their modules.
- `docs/guides/CLI_GUIDE.md` belongs to another session; its lines 162, 305, 317, 328 to 331,
  338, 345, 397, 405, 433, 483 to 485 and 494 name paths of the old tree.
- The type-name join exists in three registry sources.
