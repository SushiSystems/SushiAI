# Compile time and include hygiene: report

**Status:** Open — written 2026-10-26; tasks are written as workers report, nothing built.

## Dispatches

dispatch: header include inventory — haiku high, 22 tool calls, passed, rung 1
dispatch: C1 program shared library — haiku high, 27 tool calls, passed, rung 1
dispatch: C2 and C7 device sources — opus low, 3 tool calls, stopped on DISPATCH rule 3 (no build files); done inline
dispatch: C4 fusion selector — sonnet low, 33 tool calls, sent back once for rules 1 and 2, passed
dispatch: C5 span-only headers — haiku high, 56 tool calls, passed, rung 1
dispatch: C8a umbrella out of tests/core — haiku high, passed, rung 1
dispatch: C3 device sources checker — sonnet medium, 21 tool calls, passed, rung 1
dispatch: C8b umbrella out of tests/layers — haiku high, running
dispatch: C8c umbrella out of tests/vision and tests/language — haiku high, running

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
| Needs the full header | 56 |
| A forward declaration would do | 27 |
| Includes it for nothing | `train/objective.hpp` and `engine_fixture/test_environment.hpp` include `engine.hpp` and use none of it |

The inventory said 17 of these named only `SushiRuntime::span`. C5 read each one: 16 name
`SushiBLAS::Engine` or `Tensor`, so the claim was wrong and the inventory's verdicts are not to be
used without reading the header. The "forward declaration would do" count of 27 is unchecked for
the same reason. `test_environment.hpp` was the one header that included `engine.hpp` for nothing;
`objective.hpp` uses `Tensor`. `anchor_objective.hpp` includes `tensor.hpp` and names no `Tensor`.

SushiBLAS ships no forward header; `Tensor` is a `struct`, `Engine` a `class`.

Umbrellas: `SushiAI/SushiAI.hpp` reaches 201 headers and is included by 107 test files, 8 module
files (the three Darknet case modules' headers and sources, and the language and vision
umbrellas) and 7 application files. `SushiAI/vision.hpp` reaches 144 headers and two tests.
