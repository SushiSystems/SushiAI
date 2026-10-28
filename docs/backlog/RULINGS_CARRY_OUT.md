# Rulings carry out

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-2 | P2 | open | Make `BatchNorm2d` and a nested `ModelGraph` refuse a misfit with a `Status` | `NN::ModuleNode::create` returns `InvalidDescription` for a `BatchNorm2d` whose input is not `[H, W, channels]` and for a nested `ModelGraph` whose input does… |  |  | 2026-10-20 |
| RW-5 | P2 | open | Design value-dependent control flow in autograd | a design says how `Autograd::differentiate` treats a branch or loop that depends on a value; the owner took it into scope on 2026-10-19 |  |  | 2026-10-20 |
| RW-6 | P2 | open | Test a model file with a missing block | a test under `tests/vision/registration/` loads a model file with a block missing and checks that the `Status` names the block |  |  | 2026-10-20 |
| RW-130 | P2 | open | Carry out layered execution across the stack | tasks A0 to A7 of `docs/design/LAYERED_EXECUTION/README.md` are closed, and with them `sushiruntime RW-65`, SushiBLAS's B1 to B13 and SushiEngine's E1 to E17 |  | `docs/design/LAYERED_EXECUTION/README.md` | 2026-10-20 |
| RW-145 | P2 | open | Measure SushiAI on SushiProfiler | phases P1 to P6 of `docs/design/PROFILING/README.md` are built and the YOLOv5n write-up names the share of each group of a step and the kernels that bound it |  | `docs/design/PROFILING/README.md` | 2026-10-26 |
| RW-146 | P2 | open | Move the project service of the CLI onto injected collaborators | `cli/sushiai/services/project.py` holds a `ProjectService` that takes its console, runner, driver and configuration through its constructor, as SushiBLAS's… |  | `cli/sushiai/services/project.py` | 2026-10-26 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
