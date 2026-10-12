# Plan-packed scratch: plan

Each task has one acceptance criterion and at most one build. A build runs with 6 jobs and only
while the owner's machine is free; no run raises the memory limit of `sa`.

| # | Task | Files | Acceptance |
| --- | --- | --- | --- |
| 1 | Split the pool's byte counter by kind and print it for one step of YOLOv4-tiny | `tensor_pool.hpp`, `tensor_pool.cpp`, `tests/vision_support/darknet_training.cpp` | The run prints arenas, own values, ones and scratch; the share of scratch is written into `REPORT.md` |
| 2 | Owner's word on section 5 of the SPEC, with the share of task 1 beside it | none | The three choices are settled |
| 3 | `ScratchRequest`, `ScratchRule`, the second member of `LoweringEntry`, `scratch_of` | `lowering/scratch.hpp`, `lowering/rule.hpp`, `lowering.hpp`, `lowering.cpp` | `Unit_Lowering` lists the requests of a graph with one batch norm and one linear layer |
| 4 | One scratch rule per operation that asks for scratch, marking each request `Transient` or `Held` | `lowering/norm.cpp`, `loss.cpp`, `linalg.cpp`, `random.cpp`, their headers | Every `scratch()` call of the four files has a request with the same slot, shape and type |
| 5 | `MemoryPlan::build` places `Transient` requests; `scratch_placement` | `memory_plan.hpp`, `memory_plan.cpp`, `test_memory_plan.cpp` | The three new cases of SPEC section 7.3 pass |
| 6 | `TensorPool::scratch` returns the placed view; `LoweringContext::scratch(slot)` | `tensor_pool.*`, `arena.*`, `lowering/context.*`, the four lowering files | The whole suite passes with unchanged bounds |
| 7 | `CompiledStep`, `Trainer` and `compile_region` pass the requests to the planner | `compiled_step.cpp`, `region_program.cpp`, the training steps that build a plan | One step of YOLOv4-tiny reports scratch bytes that are `Held` buffers alone |
| 8 | Measure and close: the tiny models before and after, the full-size suites under the default limit, the manual | `REPORT.md`, `modules/core/graph/README.md`, `ROADMAP.md` section 18, `REMAINING_WORK.md`, `KNOWN_ISSUES.md`, `CHANGELOG.md` | The numbers are in `REPORT.md` and no page states the old behaviour |

Tasks 3 to 5 touch no file that tasks 6 and 7 need first, so 3, 4 and 5 can be written before
one build. Task 6 changes every lowering row at once and gets a build of its own.

## Risks

- **A request marked `Transient` that is `Held`.** A buffer the host fills once would be
  overwritten by another node. Task 4 reads each row for a host write or a prelude before it
  marks a request, and task 6 runs the whole suite.
- **The node limit.** Above `PlanOptions::reuse_node_limit` (8192 nodes) the planner overlaps
  nothing. The node count of the full-size training graphs is not known; task 1 prints it.
- **A small share.** If task 1 shows scratch is a small part of the total, the repair of the
  full-size suites lies elsewhere, most likely in what the step keeps for the backward pass.
