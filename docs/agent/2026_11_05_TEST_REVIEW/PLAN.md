# Test review: plan

**Status:** Open — the host backend (RW-161, decisions 20 and 21) and M9 and M11 are done; M31 to M52 are left

Applies the verdicts of `REPORT.md`, in hierarchy order: the shared helpers, adapters and seams the
rewrites need come first, then the defects the review found, then each module's tests. A module job
applies every record of its module: rewrite, resize, split, merge and delete as the record says, and
writes each add. Nothing in a module job changes production code except a seam a job above built.

## Jobs

| Job | Files | Waits on | Status |
| --- | --- | --- | --- |
| T1 | `tests/support/tolerance.hpp` in use, with the ruling of decision 1 applied to every named epsilon |  | done |
| T2 | One engine and device per program through SushiRuntime's fixture base; `EngineTest` stops making one per test | SushiRuntime T1 | done |
| T3 | In-memory adapters beside their ports: a host `Dataset`, `Objective`, `EpochAware`, `ReplicaModel`, `ReplicaServeObserver`, `StepRecorder`, `GraphBuilder`, `PrecisionPolicy`, `RegionPartitioner`, `StepExecutor`, `Collator`, `SamplePipeline`, `SampleSource`, `MaskSource`, `PartnerSource`, and the failure module's block counter and fatal sink |  | done; no adapter for Dataset, the Collators, ReplicaModel (device tensors), GraphBuilder (concrete) or the block counter (a function) |
| T4 | Byte-source and stream seams for the dataset readers, the checkpoint loader and the weight importer |  | done |
| T5 | The SushiLog sink moved to SushiLog's scoped in-memory sink | sushilog T1 | done |
| T6 | The defects of this report, each with a test that fails first, and the pages corrected |  | done; open items in RW-151, RW-152, RW-153 |
| T7 | The CLI tests on sushicore's in-memory runner, console and CMake driver | sushicore T1 | done |
| T8 | `tools/testing/check_test_additions.py`, `tools/tests/test_check_test_additions.py` deleted |  | done |
| M1 | the tests of `(root)` (21 files): 148 rewrite, 23 resize, 40 split, 73 merge, 21 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-157 in 3d2cc353, not yet built or run) |
| M2 | the tests of `applications` (22 files): 11 rewrite, 24 resize, 13 split, 6 merge, 4 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-158 closed in 2cfb768b, not yet built or run) |
| M3 | the tests of `cli` (16 files): 46 rewrite, 8 resize, 10 split, 3 merge, 14 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done except `test_find_project_root_raises_outside_checkout`, which waits on a walk bound in sushicore outside this repository (RW-159 stays open for it); not yet run |
| M4 | the tests of `core/allocation` (1 file): 1 resize | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-155 seam built, not yet built or run) |
| M5 | the tests of `core/autograd` (14 files): 36 rewrite, 5 resize, 19 split, 11 merge, 6 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (NoOperationAppearsTwice as `rules/compile_elementwise.cpp`, the fold refusals as `Small_BatchNormFoldMismatch` rows; not yet built or run) |
| M6 | the tests of `core/config` (9 files): 2 rewrite, 43 resize, 9 split, 2 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-155 seam built, not yet built or run) |
| M7 | the tests of `core/darknet_reference` (1 file): 4 rewrite, 4 resize | T1, T2, T3, T4, T5, T6, T7, T8 | done (folder overloads and decode_darknet_tensor built; not yet built or run) |
| M8 | the tests of `core/data` (9 files): 25 rewrite, 9 resize, 1 split, 22 merge, 4 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-155 seams built in 7a34fa48, not yet built or run) |
| M9 | the tests of `core/distribution` (13 files): 27 rewrite, 3 resize, 11 split, 25 merge, 3 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-160 closed in ab74f5b8, not yet built or run) |
| M10 | the tests of `core/dtype` (1 file): 1 rewrite, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M11 | the tests of `core/eager` (19 files): 38 rewrite, 8 resize, 12 split, 21 merge, 10 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-162 closed, not yet built or run) |
| M12 | the tests of `core/engine_fixture` (3 files): 1 rewrite, 1 resize, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-155 seam built, not yet built or run) |
| M13 | the tests of `core/execution` (1 file): 2 rewrite, 1 split, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M14 | the tests of `core/failure` (3 files): 14 rewrite, 4 resize, 6 split, 1 merge, 4 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-155 seams built, not yet built or run) |
| M15 | the tests of `core/failure_expectations` (1 file): 1 resize, 2 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M16 | the tests of `core/graph` (33 files): 147 rewrite, 22 resize, 37 split, 2 merge, 12 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-155 seams built in 101edc81, not yet built or run) |
| M17 | the tests of `core/hash` (1 file): 2 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M18 | the tests of `core/log` (2 files): 2 rewrite, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M19 | the tests of `core/model_text` (1 file): 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M20 | the tests of `core/ops` (1 file): 12 rewrite, 2 merge | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M21 | the tests of `core/optim` (16 files): 46 rewrite, 9 resize, 10 split, 8 merge, 3 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-155 seams built; not yet built or run) |
| M22 | the tests of `core/profile` (4 files): 5 rewrite, 1 resize, 1 split, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-155 seam built, not yet built or run) |
| M23 | the tests of `core/profile_records` (1 file): 2 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M24 | the tests of `core/scratch_files` (2 files): 2 resize, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-155 seam built, not yet built or run) |
| M25 | the tests of `core/seam` (1 file): 5 merge | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M26 | the tests of `core/seeded_noise` (1 file): 1 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M27 | the tests of `core/shape` (2 files): 4 resize, 3 split, 4 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M28 | the tests of `core/tensor_values` (1 file): 2 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | done (RW-155 seam built, not yet built or run) |
| M29 | the tests of `core/version` (1 file): 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M30 | the tests of `language` (46 files): 152 rewrite, 27 resize, 33 split, 19 merge, 15 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (2026-11-09): every record of the 46 files is applied; the numeric blocks cases run on `Graph::HostEvaluator`, and the cases through `CAUSAL_MASK` stay `Medium_` because the mask has no host kernel |
| M31 | the tests of `layers/classifier_objective` (1 file): 1 split, 1 merge | T1, T2, T3, T4, T5, T6, T7, T8 | done (2026-11-09, not yet built or run) |
| M32 | the tests of `layers/io` (8 files): 15 rewrite, 19 resize, 4 split, 10 merge, 6 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (2026-11-09, not yet built or run) |
| M33 | the tests of `layers/layer_rows` (2 files): 2 rewrite, 1 split, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done (2026-11-09, not yet built or run) |
| M34 | the tests of `layers/layer_traces` (1 file): 2 rewrite, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done (2026-11-09, not yet built or run) |
| M35 | the tests of `layers/nn` (39 files): 89 rewrite, 42 resize, 36 split, 138 merge, 30 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M36 | the tests of `layers/reference_files` (1 file): 1 rewrite, 1 resize, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M37 | the tests of `layers/train` (39 files): 107 rewrite, 28 resize, 36 split, 20 merge, 20 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M38 | the tests of `layers/umbrella` (2 files): 2 rewrite, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M39 | the tests of `package` (2 files): 1 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M40 | the tests of `tools` (6 files): 7 rewrite, 85 resize, 14 split, 12 merge, 6 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M41 | the tests of `vision/blocks` (12 files): 73 rewrite, 7 resize, 20 split, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M42 | the tests of `vision/codecs` (5 files): 21 rewrite, 6 resize, 2 split, 3 merge, 3 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M43 | the tests of `vision/darknet_forward_cases` (1 file): 4 rewrite, 3 split, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M44 | the tests of `vision/darknet_import_cases` (1 file): 3 resize, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M45 | the tests of `vision/darknet_training_cases` (1 file): 1 rewrite, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M46 | the tests of `vision/datasets` (39 files): 96 rewrite, 63 resize, 38 split, 3 merge, 13 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M47 | the tests of `vision/detect` (47 files): 226 rewrite, 19 resize, 32 split, 14 merge, 18 delete, 1 characterize | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M48 | the tests of `vision/importers` (23 files): 92 rewrite, 23 resize, 14 split, 10 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M49 | the tests of `vision/registration` (3 files): 20 rewrite, 9 resize, 7 split, 4 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M50 | the tests of `vision/segment` (7 files): 8 rewrite, 4 split, 3 merge | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M51 | the tests of `vision/training` (37 files): 88 rewrite, 19 resize, 25 split, 8 merge, 9 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M52 | the tests of `vision/umbrella` (2 files): 2 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | todo |

## Acceptance

- **T1.** No comparison spells a tolerance literal. Build: the repository's test command over the touched tests.
- **T2.** No fixture builds an engine or device per test. Build: the repository's test command.
- **T3.** Each keeps its port's promise with no device or file. Build: the repository's test command over each module.
- **T4.** Each reader is tested small on bytes. Build: the repository's test command over io and vision datasets.
- **T5.** Every log capture goes through SushiLog's scoped in-memory sink. Build: the repository's test command.
- **T6.** Every defect above is fixed or has a backlog entry. Build: the repository's test command and `python tools/check_all.py .`
- **T7.** No CLI test patches the repository's own functions. Build: the CLI tests.
- **T8.** Neither file exists and the checkers pass. Build: `python tools/check_all.py .`
- **M jobs.** Every record of the module is applied: no test the report deletes remains, every kept and
  new test passes at the size its label states, and the module's tests pass through the project's own
  test command. Build: that command over the module.

The owner chooses inline or parallel mode. The IR and trace records follow decision 2: a delete where a behaviour test covers the promise, a rewrite to outputs otherwise.
