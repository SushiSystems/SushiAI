# SushiAI on SushiProfiler: plan

**Status:** Open — 2026-10-26; rewritten after review, under its second review, no code yet
**Implements:** PROFILING#P1, PROFILING#P2, PROFILING#P3, PROFILING#P4, PROFILING#P5

Spec: `SPEC.md` in this folder. Every task closes on one `sa build` and `sa test` at one job (the
owner moved every build to the end of the programme), its tests passing, and
`python tools/check_all.py .` clean. Workers write no build file and run no build; each task lists
its build lines for the orchestrator. Each task carries its module README rows, the manual pages a
`git grep` of its changed names finds, and its changelog lines.

## Global constraints

- SushiProfiler v0.2.0 names only; nothing from v0.3.0.
- No public header of SushiAI includes a `profile` header or a SushiProfiler header; modules name
  `profile` in `PRIVATE_DEPENDS`.
- Zone groups and stages exactly as SPEC section 3; timeline names `train.step` and
  `inference.batch`.
- Breaking removals carry one changelog entry each, naming the replacement.
- Code uses SushiRuntime's R15 and R16 and SushiBLAS's B-O1 with the signatures SPEC section 1
  gives.

## Review focus

1. A run with no `--profile`: no session, no record, `mark_period` does nothing. T1's test.
2. A `--profile` path that does not exist, holds bad JSON or an unknown key: the program stops
   before training, naming the file. T4's test.
3. Two compiled steps with equal node indices: the record keeps them apart by `graph`. T3's test.
4. The optimizer's kernels: they carry the origin `step.trailing`, not an empty one. T3's test.
5. A run that ends with an error mid-step: `close()` still writes the record. T4's test.

## Tasks

| Id | Task | Who | Acceptance |
| --- | --- | --- | --- |
| A1 | Build lines: tier entry `profile core` in `cmake/SushiAILayers.cmake`; `modules/core/profile/CMakeLists.txt` with `sushiai_add_module(NAME profile TIER core SOURCES timelines.cpp)` linking `SushiProfiler::SushiProfiler` PUBLIC and defining `SA_ENABLE_PROFILER` and `SA_PROFILE_LEVEL` PUBLIC on it (no SushiProfiler link when the option is OFF); options `SA_ENABLE_PROFILER` ON and `SA_PROFILE_LEVEL` 2 in the root; `cmake/Profiler.cmake` included after `include(Log)`; `[sushisystems.sushiprofiler]` v0.2.0 and `depends_on` in `cli/sushistack.deps.toml`; `tests/core/profile/CMakeLists.txt`; the options' rows in the build manual page and their changelog line | orchestrator | Configure succeeds with the option ON and OFF |
| T1 | `profile.hpp`, `timelines.hpp`, `timeline_binding.hpp` and `source/timelines.cpp` as SPEC section 2; module README; `tests/core/profile/test_timelines.cpp` | worker | `Small_Timelines.NamesAreTrainStepAndInferenceBatch`, `Small_Timelines.MarkWithNoBindingDoesNothing` and `Medium_Timelines.ABoundSessionRecordsOnePeriodPerMark` pass |
| T2 | Layers and eager: in `modules/layers/train` (`trainer.hpp`, `trainer.cpp`) and `modules/core/eager` (`session.hpp`, `session.cpp`), remove SPEC section 4's names and add the `train`, `data`, `host`, `copy`, `device` and `eager` zones and the `TrainStep` mark of `Trainer::fit`; change the section 4 consumers (`test_convergence.cpp` and `.md`, `test_session.cpp`, `demo_mlp.cpp`, the two comments); `PRIVATE_DEPENDS profile` lines for `train` and `eager` | worker | `Medium_TrainerZones.AFitOfTwoStepsWritesTwoPeriodsWithTheirGroups` passes, and `git grep -n "node_timings\|duration_ms\|partition_ns\|options.profile" -- modules tests applications` is empty |
| T3 | Graph: in `modules/core/graph` (`compiled_step.hpp`, `compiled_step.cpp`, `lowering.cpp`), the `label` parameter, the `compile` zones, the per-rule origin scope and the `step.trailing` scope; the labels in `training_step.cpp`, `evaluation_step.cpp`, `replica_step.cpp` and `region_executor.cpp`; tests in `tests/core/graph/` that attach a `SushiRuntime::RunObserver` and read `RunObservation::table`; `PRIVATE_DEPENDS profile` for `graph` | worker | `Medium_CompiledStepAttribution.EveryTaskOfANodeCarriesItsOrigin`, `.TwoStepsHaveTwoLabels`, `.TheOptimizerCarriesTheTrailingOrigin` and `.AViewLeavesNoOriginOpen` pass |
| T5 | Vision training: zones and the `TrainStep` mark in `anchor_training_loop.cpp`, `free_training_loop.cpp`, `anchor_training_step.cpp` and `free_training_step.cpp`, and the `free_assignment_stage.cpp` assign zone; `PRIVATE_DEPENDS profile` for `vision/training` | worker | `Medium_FreeTrainingZones.AStepPutsReadbackAssignAndLossInItsPeriod` passes |
| T4 | Programs and CLI: `ProfileSession` (SPEC section 6); the zones and marks of `epoch_loop.cpp`, `trainer_run.cpp`, `inference_program.cpp`, `domains_vision.cpp` and `training_program.cpp`; `--profile` in `train.cpp` and `inference.cpp`; `sa train --profile`, `sa inference --profile` and `sa profile` in `cli/sushiai/cli.py` with tests under `cli/tests/`; a guide page on profiling a run; `PRIVATE_DEPENDS profile` for `programs` | worker | `Medium_ProfileSession.AMissingConfigNamesTheFile`, `.AnUnknownKeyNamesTheFile`, `.ARunThatFailsStillWritesItsRecord`, and the CLI tests `test_train_forwards_profile` and `test_profile_runs_sp` pass |
| T6 | SushiCore's O5 lines applied: `cmake/Log.cmake`, `Runtime.cmake`, `BLAS.cmake`, `SushiAISyclTarget.cmake`, and `services/project.py`'s `-D..._DIR` replaced by `sushi_packages`; the build manual page and changelog line | orchestrator | `sa build` configures against the packages and compiles no SushiRuntime or SushiBLAS source |
| T7 | P6: two YOLOv5n runs, one on the production path with level 1 and one with device timing; `sp compare`; the write-up in `REPORT.md` and the design topic | orchestrator | The report names each group's share of a step and the kernels and IR nodes that bound it |

## Waves

| Wave | Tasks | Files | Waits on |
| --- | --- | --- | --- |
| 1 | A1, T1 | Build files; `modules/core/profile`, `tests/core/profile` | SushiProfiler v0.2.0's names |
| 2 | T2, T3, T5 | `modules/layers/train/{trainer.*}`, `modules/core/eager` and their tests; `modules/core/graph`, `training_step.cpp`, `evaluation_step.cpp`, `replica_step.cpp`, `region_executor.cpp` and their tests; `modules/vision/training` and its tests. No file is in two sets | T1; T3 also R15, R16 and B-O1 |
| 3 | T4 | `applications/programs`, `cli/` | Wave 2 |
| 4 | T6, T7 | Build files; records | SushiCore's packages, the build |

## What the first build must confirm

- `SushiProfiler::SushiProfiler` is found from its package, and its DLL is on the CLI's path.
- `Graph::set_label`, `NodeOptions::origin` and `TaskRecorder::origin` exist with the signatures
  used.
- `SA_ZONE` with a literal group and stage compiles to a site whose group `breakdown` reads.
- A level-0 build (`SA_PROFILE_LEVEL=0`) compiles every zone away.
