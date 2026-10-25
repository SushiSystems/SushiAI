# Compile time and include hygiene: report

**Status:** Open — written 2026-10-26; tasks are written as workers report, nothing built.

## Dispatches

dispatch: header include inventory — haiku high, 22 tool calls, passed, rung 1
dispatch: C1 program shared library — haiku high, 27 tool calls, passed, rung 1
dispatch: C2 and C7 device sources — opus low, running

## 1. Findings

Section 2 of `PLAN.md` holds what was read on 2026-10-26.

## 2. C1, written and not built

The program sources form the static library `sushiai_program_shared`; the four programs and the
program tests link it, and each source compiles once instead of five times. It links `sushiai`
and the domain libraries PUBLIC, because the headers beside the sources name their types.
Registration is by explicit call, so a static library drops no row. Not verified: the link of the
program tests in a distributed build.

## 3. Header inventory

83 headers under `modules/**/include` and `applications/programs/source` include `engine.hpp`,
`tensor.hpp`, `task_recorder.hpp`, `blas.hpp`, `SushiBLAS.hpp`, `sushiruntime.hpp` or
`random.hpp`.

| Verdict | Headers |
| --- | --- |
| Needs the full header | 56, of which 17 name only `SushiRuntime::span` from it |
| A forward declaration would do | 27 |
| Includes it for nothing | `train/objective.hpp` and `engine_fixture/test_environment.hpp` include `engine.hpp` and use none of it |

The 17 span-only headers: `lowering.hpp`, `cached_generator.hpp`, `position_feed.hpp`,
`ring_position_feed.hpp`, `table_position_feed.hpp`, `window_generator.hpp`, `step_view.hpp`,
`darknet_forward.hpp`, `darknet_import.hpp`, `darknet_training.hpp`, `batch_inputs.hpp`,
`task_table.hpp`, `anchor_objective.hpp`, `anchor_training_step.hpp`, `evaluation_step.hpp`,
`inference_run.hpp`, `training_run.hpp`.

SushiBLAS ships no forward header; `Tensor` is a `struct`, `Engine` a `class`.

Umbrellas: `SushiAI/SushiAI.hpp` reaches 201 headers and is included by 107 test files, 8 module
files (the three Darknet case modules' headers and sources, and the language and vision
umbrellas) and 7 application files. `SushiAI/vision.hpp` reaches 144 headers and two tests.
