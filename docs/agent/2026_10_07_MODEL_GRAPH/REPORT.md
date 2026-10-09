# Model graph engine units: report

Covers the seven waves of [`PLAN.md`](PLAN.md), written on 2026-10-09 by workflows
`wf_3f76e1ed-de3` (waves 1 to 4) and `wf_9bdc628a-453` (waves 5 to 7). The waves were written without a build; section "Build and run" records the build and the runs
that followed.

## Dispatches

```
dispatch: T14.1 ModelState — haiku medium, 32 turns, passed, rung 1
dispatch: review wave 1 — opus low, 9 turns, passed, rung 1
dispatch: T14.2 initialize over a state — haiku medium, 24 turns, passed, rung 1
dispatch: T15.1 import_weights — haiku medium, 22 turns, passed, rung 1
dispatch: T15.2 state checkpoints — haiku medium, 22 turns, passed, rung 1
dispatch: T16.4 BufferUpdateRecorder — haiku medium, 32 turns, passed, rung 1
dispatch: T17.1 Dataset::sample_shape and the MNIST layout — haiku medium, 29 turns, passed, rung 1
dispatch: review wave 2 — opus low, 14 turns, passed, rung 1
dispatch: T16.1 step builders — haiku medium, 12 turns, passed, rung 1
dispatch: T16.2 bind_parameters over a state — opus low, 12 turns, passed, rung 1
dispatch: T16.3 Trainer with buffer updates — opus low, 15 turns, passed, rung 1
dispatch: T16.5 TrainingStep and check_feed — haiku medium, 18 turns, passed, rung 1
dispatch: T16.6 EvaluationStep — opus low, 10 turns, passed, rung 1
dispatch: review wave 3 — opus low, 18 turns, passed, rung 1
dispatch: T16.7 callers of the step builders — haiku high, 45 turns, passed, rung 1
dispatch: tests of T16.1, T16.2 and T16.6 — opus low, 29 turns, passed, rung 1
dispatch: review wave 4 — opus low, 15 turns, passed, rung 1
dispatch: T17.4 — haiku medium, 10 turns, passed, rung 1
dispatch: T17.2 — opus low, 11 turns, passed, rung 1
dispatch: T17.3 — opus low, 11 turns, passed, rung 1
dispatch: review wave 5 — opus low, 8 turns, passed, rung 1
dispatch: T17.5c — haiku high, 34 turns, passed, rung 1
dispatch: T17.5a — haiku high, 42 turns, passed, rung 1
dispatch: T17.5b — haiku high, 45 turns, passed, rung 1
dispatch: review wave 6 — opus low, 12 turns, passed, rung 1
dispatch: T17.6 — haiku high, 53 turns, passed, rung 1
dispatch: review wave 7 — opus low, 12 turns, passed, rung 1
```

"Passed" means the wave's reviewer accepted the unit after its own repairs. The reviewers
repaired 12 of the 13 implementer units; T16.1 alone needed none.

## What the reviewers repaired

| Wave | File | Repair |
| --- | --- | --- |
| 1 | `nn/model_state.hpp` | Included `core/seam.hpp`, which the `create` template uses; stated the fatal `K_NO_STATE` precondition of `from_entries` |
| 1 | `nn/model_state.cpp` | Moved the allocation loop of `from_entries` inside `guard(SeamPhase::Allocate)` |
| 1 | `test_model_state.cpp` | Replaced three bare floating-point `EXPECT_EQ` with `EXPECT_NEAR` and a stated tolerance |
| 2 | `buffer_update_recorder.cpp` | Moved the allocation of `bind_buffer_updates` under `guard(Allocate)`; the mismatch text names both dtypes and shapes |
| 2 | `import_weights.cpp` | The unknown-record text says "entry", not "parameter" |
| 2 | `test_import_weights.cpp` | Replaced a local byte comparison with `SaTest::read_tensor_bytes` |
| 2 | `dataset.cpp`, `init.cpp`, two tests | Placement, missing includes, comment shape, one 101-column line |
| 3 | `trainer.cpp` | Bridged `MemoryPlan::build`, which throws; the state form returns `Incompatible` when a pre-planned pool packs a pinned value instead of ignoring `built.pinned` |
| 3 | `feed.cpp`, `steps.hpp` | Bridged the throwing `Shape` constructor; `check_feed` returns `InvalidDescription` for extents no `Shape` holds |
| 3 | `training_step.cpp` and three headers | Qualified `Detail::`; cut comment blocks to eight lines |
| 4 | `test_training.cpp`, `test_evaluation_step.cpp` | Missing includes and `@brief` lines; an explicit seed on every new dataset |
| 4 | `test_steps.cpp` | Removed a provenance paragraph from the golden's comment |

The wave-3 reviewer confirmed the order of passes of T16.2 and T16.3 against the plan and against
`trainer.cpp` at `7fbb49c`: trace, differentiate, casts, fusion, memory plan, attach, leaves,
gradients, lowering with the optimizer and then the buffer updates.

## Closeout by the orchestrator

- Added the 9 sources, 8 headers and 7 test files of plan section 10 to `src/CMakeLists.txt`,
  `include/SushiAI/SushiAI.h` and `tests/CMakeLists.txt`.
- Cut two comment blocks over eight lines (`import_weights.hpp`, `buffer_update_recorder.hpp`).
- Added the row of the new `SA_THROW_IF` site in `CrossEntropyObjective::loss`
  (`tools/contract/site_classes.tsv`) and moved two rows to their new lines.
- `check_source_comments.py`, `check_contract.py` and `check_docs_layout.py` print no finding.

## Build and run, 2026-10-09

Built against SushiBLAS `494ff8e` and run on the RTX 3080 Ti with
`sa run sushiai_tests -- --gtest_brief=1`. The last of four runs: 1878 tests in 131 suites,
1877 passed, 1 skipped, 0 failed, 35 s. The first build of the four waves had no error. The same
runs were the first for commit `9e9f8a7`.

| Run | Result | Cause | Repair |
| --- | --- | --- | --- |
| 1 | Two cases of `Integration_Training` read an accuracy of 0.31 and 0.125 | `9e9f8a7` removed `Lowering::values_needing_own_allocation`, which had kept the logits out of the arena as a side effect; the planner then packed them under later values | A training graph pins its loss and outputs, the pool form of `bind_parameters` passes `built.pinned` to the planner, and `EvaluationStep` pins its outputs (`steps.cpp`, `trainer.cpp`, `evaluation_step.cpp`) |
| 2 | Fatal in `bind_buffer_updates`: a source was not bound | Nothing materialised an update's source before the step was compiled | The state form of `bind_parameters` materialises the sources (`trainer.cpp`) |
| 3 | `APerStepOptimizerStillStepsWhenTheModelHoldsABatchNorm` saw three parameters stay still | The case's expectation: with a zeroed classifier the first gradient reaches `3.weight` alone | The case checks `3.weight` (`test_training.cpp`) |
| 4 | All passed | | |

Confirmed by the runs: the instantiations of `steps.hpp` compile; the 17-operation golden of
`test_steps.cpp` holds; two compiled plans work over one storage; `ModelState::attach` skips a
buffer no leaf reads; the 14 lowering cases and the tail tolerance of the learning curve pass.
`AnImageLayoutYieldsRankFourBatches` is the skipped case: the MNIST files are absent, so the
`image` layout has not been read from a file.

## Waves 5 to 7

Wave 5 put `sushiai train` and `sushiai inference` on `NN::build_model`, a `ModelState` and the
shape-checked steps, and moved the MNIST example to a version 2 model file with the `image`
layout. Wave 6 moved 22 test and example files onto the state forms; its reviewer put `state`
before `pool` at 15 regression sites. Wave 7 deleted `Parameter::id`, `ModelSpec::input_features`,
the module forms of `initialize`, `save_checkpoint` and `load_checkpoint`, the pool form of
`bind_parameters`, and four cases the plan marks; its reviewer moved four rows of
`site_classes.tsv` to their new lines.

Built and run on 2026-10-09, same machine and SushiBLAS commit:

| Run | Result | Cause | Repair |
| --- | --- | --- | --- |
| 5 | `AParameterTheModelDoesNotHaveIsRefusedAndNamesIt` failed | The case pinned the wording of the deleted module loader | It pins `import_weights`' wording, "this model has no such entry" |
| 6 | 1874 tests, 1873 passed, 1 skipped, 0 failed | | |
| 7 | 1875 tests, 1874 passed, 1 skipped, 0 failed, after the closing refactor below | | |

No recorded regression reference moved under the state forms, and the counts of
`Regression_StaticMemory` hold. `sushiai_train` on `tiny_train.json`, `sushiai_inference` on the
checkpoint it wrote and `sushiai_demo_mlp` with the arguments of its CTest case each printed
`compiles: 1`. The MNIST example was not run: its IDX files are absent.

## Closing refactor

dispatch: one home for four duplicated helpers — opus low, passed, rung 1
dispatch: README of the unit tests — haiku high, passed, rung 1
dispatch: READMEs of the integration and regression tests — haiku high, passed, rung 1
dispatch: module README, guides and architecture — haiku high, passed, rung 1

`Detail::adopt_plan` (`train/steps.hpp`) replaces the two plan adoptions; `ScratchFile` and
`build_two_output_model` live in `tests/common/test_utils.hpp`; `from_entries` has its death
test. The orchestrator wrote `core/tensor_shape.hpp`, whose `shape_of` replaces three file-local
copies and the inline one of `record_for`; the dispatched agent had stopped there because
`src/core/shape.cpp` does not exist.

## Open

- Eighteen cases that predate this work have no entry in the integration README, and the
  regression README keeps five sentences that tell history.

- `site_classes.tsv:6` carries a note that cites `checkpoint.hpp:96`; the code is at line 61.
- `test_fusion_numerics.cpp` has six lines over 100 columns that predate this work.
- The MNIST example in version 2 has not been loaded or trained.
