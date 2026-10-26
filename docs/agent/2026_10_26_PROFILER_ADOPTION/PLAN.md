# SushiAI on SushiProfiler: plan

**Status:** Open — 2026-10-26; reviewed twice, no code yet
**Implements:** PROFILING#P1, PROFILING#P2, PROFILING#P3, PROFILING#P4, PROFILING#P5

Spec: `SPEC.md` in this folder. Every task closes on one `sa build` and `sa test` at one job (the
owner moved every build to the end of the programme), its tests passing, and
`python tools/check_all.py .` clean. Workers write no build file and run no build; each task lists
its build lines for the orchestrator. Each task carries its module README rows and the manual pages a
`git grep` of its changed names finds. Workers hand their changelog lines back in the report; the
orchestrator writes `CHANGELOG.md`, so no two workers hold it.

## Global constraints

- SushiProfiler v0.2.0 names only; nothing from v0.3.0.
- No public header of SushiAI includes a `profile` header or a SushiProfiler header; modules name
  `profile` in `PRIVATE_DEPENDS` and SushiProfiler in `EXTERNAL_PRIVATE_LINKS`.
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
| A1 | Build lines: tier entry `profile core` in `cmake/SushiAILayers.cmake`; `modules/core/profile/CMakeLists.txt` with `sushiai_add_module(NAME profile TIER core SOURCES timelines.cpp EXTERNAL_PRIVATE_LINKS SushiProfiler::SushiProfiler)` (no SushiProfiler link when the option is OFF) and `configure_file` of `profile_config.hpp.in` holding `SA_ENABLE_PROFILER` and `SA_PROFILE_LEVEL`; `PRIVATE_DEPENDS profile` and `EXTERNAL_PRIVATE_LINKS SushiProfiler::SushiProfiler` for `train`, `eager`, `graph`, `vision/training` and `programs`; options `SA_ENABLE_PROFILER` ON and `SA_PROFILE_LEVEL` 2 in the root; `cmake/Profiler.cmake` included after `include(Log)`; `[sushisystems.sushiprofiler]` v0.2.0 and `depends_on` in `cli/sushistack.deps.toml`; `tests/core/profile/CMakeLists.txt`; the options' rows in the build manual page and their changelog line | orchestrator | Configure succeeds with the option ON and OFF |
| T1 | `profile.hpp`, `timelines.hpp`, `timeline_binding.hpp` and `source/timelines.cpp` as SPEC section 2; module README; `tests/core/profile/test_timelines.cpp` | worker | `Small_Timelines.NamesAreTrainStepAndInferenceBatch`, `Small_Timelines.MarkWithNoBindingDoesNothing` and `Medium_Timelines.ABoundSessionRecordsOnePeriodPerMark` pass |
| T2 | Layers and eager: in `modules/layers/train` (`trainer.hpp`, `trainer.cpp`) and `modules/core/eager` (`session.hpp`, `session.cpp`), remove SPEC section 4's names and add the `train`, `data`, `host`, `copy`, `device` and `eager` zones and the `TrainStep` mark of `Trainer::fit`; change the section 4 consumers (`test_convergence.cpp` and `.md`, `test_session.cpp`, `demo_mlp.cpp`); `modules/layers/train/README.md` is T2's, and T3 hands its rows for it back in its report | worker | `Medium_TrainerZones.AFitOfTwoStepsWritesTwoPeriodsWithTheirGroups` passes, and `git grep -n "node_timings\|duration_ms\|partition_ns\|options.profile" -- modules tests applications` is empty |
| T3 | Graph: in `modules/core/graph` (`compiled_step.hpp`, `compiled_step.cpp`, `lowering.cpp`), the `label` parameter, the `compile` zones, the per-rule origin scope and the `step.trailing` scope; the labels in `training_step.cpp`, `evaluation_step.cpp`, `replica_step.cpp` and `region_executor.cpp`; the two comments of SPEC section 4 (`compiled_step.hpp:151-152`, `ir.hpp:184`); tests in `tests/core/graph/` that attach a `SushiRuntime::RunObserver` and read `RunObservation::table` | worker | `Medium_CompiledStepAttribution.EveryTaskOfANodeCarriesItsOrigin`, `.TwoStepsHaveTwoLabels`, `.TheOptimizerCarriesTheTrailingOrigin` and `.AViewLeavesNoOriginOpen` pass |
| T5 | Vision training: zones and the `TrainStep` mark in `anchor_training_loop.cpp`, `free_training_loop.cpp`, `anchor_training_step.cpp` and `free_training_step.cpp`, and the `free_assignment_stage.cpp` assign zone | worker | `Medium_FreeTrainingZones.AStepPutsReadbackAssignAndLossInItsPeriod` passes |
| T4 | Programs and CLI: `ProfileSession` (SPEC section 6); the zones and marks of `epoch_loop.cpp`, `trainer_run.cpp`, `inference_program.cpp`, `domains_vision.cpp` and `training_program.cpp`; `--profile` in `train.cpp` and `inference.cpp`; `sa train --profile`, `sa inference --profile` and `sa profile` in `cli/sushiai/cli.py` with tests under `cli/tests/`; a guide page on profiling a run | worker | `Medium_ProfileSession.AMissingConfigNamesTheFile`, `.AnUnknownKeyNamesTheFile`, `.ARunThatFailsStillWritesItsRecord`, and the CLI tests `test_train_forwards_profile` and `test_profile_runs_sp` pass |
| T6 | SushiCore's O5 lines applied: `cmake/Log.cmake`, `Runtime.cmake`, `BLAS.cmake`, `SushiAISyclTarget.cmake`, and `services/project.py`'s `-D..._DIR` replaced by `sushi_packages`; the build manual page and changelog line | orchestrator | `sa build` configures against the packages and compiles no SushiRuntime or SushiBLAS source |
| T7 | P6: two YOLOv5n runs, one on the production path with level 1 and one with device timing; `sp compare`; the write-up in `REPORT.md` and the design topic | orchestrator | The report names each group's share of a step and the kernels and IR nodes that bound it |

## Waves

| Wave | Tasks | Files | Waits on |
| --- | --- | --- | --- |
| 1 | A1, T1 | Build files; `modules/core/profile`, `tests/core/profile` | SushiProfiler v0.2.0's names |
| 2 | T2, T3, T5 | `modules/layers/train/{trainer.*}`, `modules/core/eager` and their tests; `modules/core/graph` (with `ir.hpp`), `training_step.cpp`, `evaluation_step.cpp`, `replica_step.cpp`, `region_executor.cpp` and their tests; `modules/vision/training` and its tests. No file is in two sets; `CHANGELOG.md` is the orchestrator's | T1; T3 also R15, R16 and B-O1 |
| 3 | T4 | `applications/programs`, `cli/` | Wave 2 |
| 4 | T6, T7 | Build files; records | SushiCore's packages, the build |

## Decisions I took

Made while coding T1 to T5, each where the code or a checker contradicted the plan or the plan said
nothing.

| Id | Decision | Reason |
| --- | --- | --- |
| D1 | `profile` names `failure` in `PRIVATE_DEPENDS` | `TimelineBinding` asserts on a double bind with `SA_ASSERT`, which lives in `failure/check.hpp` |
| D2 | `timeline_binding.hpp` forward-declares `SushiProfiler::Session`; with `SA_ENABLE_PROFILER` 0 the binding and `mark_period` are defined as no-ops | The header then compiles without SushiProfiler, and the programs' off-build links |
| D3 | The tests of `profile` are `test_timelines.cpp`, `test_profile.cpp` and `test_timeline_binding.cpp` | `check_test_mirror` wants one test file per header |
| D4 | The eager executor labels its one `DynamicGraph` `eager.step`, not `eager.region.<n>`; `RegionEmitter` takes the region id and gives `Lowering` the origin prefix `r<id>/`, through the new `Lowering::set_origin_prefix` | A `DynamicGraph` composes every region into one plan and has one label, and each region's graph numbers its nodes from 0, so equal node ids of two regions would share an origin |
| D5 | The trailing work of the eager executor opens the origin `step.trailing`, as `CompiledStep` does | It records the optimizer through the same `StepRecorder`; review focus 4 holds for both paths |
| D6 | The `eager.plan` zone covers `table_.plan`, `eager.compile` covers the `compile_region` loop, `eager.execute` covers storage allocation to release, and `eager.flush` covers `Session::run` | The old laps measured the stretches between statements; the zones need scopes. Partition-to-plan glue (`place_leaves`, `find_boundaries`) now shows as `eager.flush` self time |
| D7 | `Session::run` of `eager` keeps `StepCost` with its four counts; the test file keeps its count assertions and drops the printed rows | The times leave with the stopwatch (SPEC section 4) |
| D8 | The attribution tests are in `test_compiled_step.cpp`, suite `Medium_CompiledStepAttribution`; the zone test of the trainer is `test_trainer/test_zones.cpp` | `check_test_mirror` refuses an orphan file and a unit with both a file and a folder |
| D9 | The trainer test marks one more period after `fit` | A period with no end mark is not kept (SushiProfiler's consumer), so the last step of a run falls out of its record |
| D10 | `ProfileSession::close` marks each timeline once before it unbinds | Same reason as D9: the last step's period gets its end mark |
| D11 | In `FreeTrainingStep`, `host.readback` also covers the widening of the leaves in `measure_branch`, and `host.assign` covers the leaf writes of `feed_leaves`; the stage's own `assign` carries its zone | SPEC section 3 names `floats_of` and `widen` of the ports only; the leaf widening is the same readback, and a zone around all of `feed_leaves` would nest `host.assign` in itself |
| D12 | `EvaluationStep::run_in_order` carries `copy.input` and `device.run`; the call sites in `inference_program.cpp` and `domains_vision.cpp` add no `device.run` | Every caller of an evaluation step, the programs and the examples, gets the zones from one place, and a zone at the call site would nest `device` in itself |
| D13 | `ProfileSession` hides its parts behind an `Impl`, so its header names no SushiProfiler type; `profile_session_none.cpp` is its definition for `SA_ENABLE_PROFILER` OFF; `open` registers SushiProfiler's builtin writers on `std::cout`; `end_profiled_run`, declared in the header, gives both programs one ending | The programs must link with the option OFF; the `terminal`, `record` and `chrome_trace` writers need the builtin registration; a failed run must keep its own failure while its record is still written |
| D14 | `close()` fails with `DeviceFailed` only when `stop()` reports problems; problems of the analyses and writers are logged | SPEC section 6 asks for a failure when the session could not be stopped and a log line for each problem |
| D15 | `--profile <path>` is read by `read_profile_arguments` (`profile_arguments.hpp`), wherever it stands on the line; a missing value, a value that is a flag and a repeat are refused | `train.cpp` and `inference.cpp` need one parser, and a flag that eats the next flag by mistake would hide a typo |
| D16 | `io.checkpoint` also wraps `load_run_checkpoint` and `load_checkpoint` in `run_start.cpp` | The two loads of `sushiai train` live there, not in `training_program.cpp` |
| D17 | `sa train` and `sa inference` get no Typer option for `--profile`; their forwarding already passes it, and `test_train_forwards_profile` pins that. `sa profile` is `services/profiler.py` | A second declaration of the flag would be a second place to change |
| U1 | A test-support module `modules/core/profile_records` holds `zones_of` and the session-config helpers; the four test files that repeated them use it | One copy of the record-reading helpers, shaped like `engine_fixture` and `failure_expectations` |
| U2 | No test file is dropped for `SA_ENABLE_PROFILER` OFF: the profiler suite of `test_free_training_step.cpp` sits under `#if SA_ENABLE_PROFILER`; the wholly profiler-bound files are listed in the build only under the option | The older suites of a mixed file must keep running in an off build |
| U3 | `end_profiled_run` is declared in `profile_session.hpp` and defined in `profile_session.cpp` and `profile_session_none.cpp` | No logic and no log call in a header |
| U4 | `trainer.hpp` drops `<SushiRuntime/sushiruntime.hpp>`; each includer names what it uses | Build time: every includer of the header stops reading the runtime umbrella |
| U5 | D18 is closed: the distributed loop and the replica model carry the zones and the `train.step` mark of the single-device path | The same groups on both paths, so one record layout reads both |
| U6 | The mirror findings for `free_training_loop.hpp`, `inference_program.hpp` and `training_program.hpp` predate this work and stay | Their tests are not part of this adoption |
| D18 | (closed by U5) The distributed loop (`cluster_run.cpp`) and `training_replica_model.cpp` keep no zone and no mark | SPEC section 9 leaves the distributed path's timing to SushiRuntime |

## What the first build must confirm

- `SushiProfiler::SushiProfiler` is found from its package, and its DLL is on the CLI's path.
- `Graph::set_label`, `NodeOptions::origin` and `TaskRecorder::origin` exist with the signatures
  used.
- `SA_ZONE` with a literal group and stage compiles to a site whose group `breakdown` reads.
- A level-0 build (`SA_PROFILE_LEVEL=0`) compiles every zone away.
