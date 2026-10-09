# Model graph engine units: report

Covers waves 1 to 4 of [`PLAN.md`](PLAN.md), written on 2026-10-09 by workflow
`wf_3f76e1ed-de3`. Nothing in these waves has been built or run. Every `.cpp` file passed
`clang++ -fsyntax-only` with the flags of its `compile_commands.json` entry, one file at a time;
the templates in `steps.hpp` were not instantiated by any of those checks.

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

## What the first build and run must confirm

1. Everything compiles and links, including the instantiations of `steps.hpp`.
2. `test_steps.cpp:93-100`: the golden of 17 operations was derived by reading the passes, not
   recorded from a run.
3. Two compiled plans over one storage (plan section 12): the second case of
   `test_evaluation_step.cpp` settles it. If it fails, the fallback of SPEC lines 1022-1025
   adds a copy between the steps and keeps every signature.
4. The probe case of `test_training.cpp` assumes `ModelState::attach` skips a buffer no leaf
   reads. The wave-4 reviewer did not read `attach`.
5. `AnImageLayoutYieldsRankFourBatches` reports skipped where the MNIST files are absent.
6. The same run is the first for commit `9e9f8a7`: `record_and_run`, the memory plan without
   `values_needing_own_allocation`, the 14 lowering cases and the tail tolerance of the
   learning curve.

## Open

- `shape_of` exists in `src/io/import_weights.cpp` and `src/train/buffer_update_recorder.cpp`,
  both repeating `record_for` of `io/checkpoint.hpp`. One home for it is a new header.
- `trainer.cpp` and `evaluation_step.cpp` each adopt a memory plan with the same lines.
- `ScratchFile` is copied in two test files and `build_two_output_model` in two others; both
  belong under `tests/common`.
- The `K_NO_STATE` fatal of `from_entries` has no death test.
- The READMEs of `nn` and of the three test folders do not describe these units yet. They are
  written once, after wave 7 deletes the transitional forms and their cases.
- `test_fusion_numerics.cpp` has six lines over 100 columns that predate this work.
- Waves 5 to 7 are not started: the applications on `build_model`, the MNIST example with
  `layout`, the callers of the pool-owned forms, the deletions.
