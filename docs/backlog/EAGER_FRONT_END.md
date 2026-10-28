# Eager front end

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-15 | P2 | unmeasured | Build the eager units and run their tests | the 19 test files of the eager module pass, seven of them on an engine |  | `docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md` | 2026-10-19 |
| RW-16 | P2 | unmeasured | Read the host cost of an eager step off a run | the `Medium_EagerStepCost` cases of `tests/core/eager/test_session.cpp` have printed the cost of an unchanged step for two and fifty layers, and of a scope… |  |  | 2026-10-19 |
| RW-17 | P2 | open | Design the phases of an eager session | the design says which values of a closed phase stay readable, and how `Autograd::differentiate` treats an earlier phase's values as constants while… |  | `docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md` | 2026-10-19 |
| RW-18 | P2 | blocked | Add the debug option that flushes after every node | an eager session can flush after each node | RW-17 |  | 2026-10-19 |
| RW-19 | P2 | open | Compare an eager loop with `Train::Trainer` | a test under `tests/` compares the two |  |  | 2026-10-19 |
| RW-20 | P2 | open | Sign every region in one pass | `signature_of` costs one walk of the graph per step, not one per region |  |  | 2026-10-19 |
| RW-21 | P2 | open | Update buffers on the eager path | a running statistic is blended across eager steps |  |  | 2026-10-19 |
| RW-22 | P2 | open | Run a session whose two states share one gradient value | `Session::run` handles it instead of returning `Unsupported` |  |  | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
