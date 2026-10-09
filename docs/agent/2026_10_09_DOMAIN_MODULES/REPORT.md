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
