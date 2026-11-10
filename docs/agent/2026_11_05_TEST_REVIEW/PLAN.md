# Test review: plan

**Status:** Open — 45 of 60 jobs are done and 15 are failed (T2, T3, T6, M3, M6, M8, M14, M16, M21, M30, M35, M37, M46, M47, M51); each record they leave is under "Not done as written"

Applies the verdicts of `REPORT.md`, in hierarchy order: the shared helpers, adapters and seams the
rewrites need come first, then the defects the review found, then each module's tests. A module job
applies every record of its module: rewrite, resize, split, merge and delete as the record says, and
writes each add. Nothing in a module job changes production code except a seam a job above built.

## Jobs

| Job | Files | Waits on | Status |
| --- | --- | --- | --- |
| T1 | `tests/support/tolerance.hpp` in use, with the ruling of decision 1 applied to every named epsilon |  | done |
| T2 | One engine and device per program through SushiRuntime's fixture base; `EngineTest` stops making one per test | SushiRuntime T1 | failed |
| T3 | In-memory adapters beside their ports: a host `Dataset`, `Objective`, `EpochAware`, `ReplicaModel`, `ReplicaServeObserver`, `StepRecorder`, `GraphBuilder`, `PrecisionPolicy`, `RegionPartitioner`, `StepExecutor`, `Collator`, `SamplePipeline`, `SampleSource`, `MaskSource`, `PartnerSource`, and the failure module's block counter and fatal sink |  | failed |
| T4 | Byte-source and stream seams for the dataset readers, the checkpoint loader and the weight importer |  | done |
| T5 | The SushiLog sink moved to SushiLog's scoped in-memory sink | sushilog T1 | done |
| T6 | The defects of this report, each with a test that fails first, and the pages corrected |  | failed |
| T7 | The CLI tests on sushicore's in-memory runner, console and CMake driver | sushicore T1 | done |
| T8 | `tools/testing/check_test_additions.py`, `tools/tests/test_check_test_additions.py` deleted |  | done |
| M1 | the tests of `(root)` (21 files): 148 rewrite, 23 resize, 40 split, 73 merge, 21 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M2 | the tests of `applications` (22 files): 11 rewrite, 24 resize, 13 split, 6 merge, 4 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M3 | the tests of `cli` (16 files): 46 rewrite, 8 resize, 10 split, 3 merge, 14 delete | T1, T2, T3, T4, T5, T6, T7, T8 | failed |
| M4 | the tests of `core/allocation` (1 file): 1 resize | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M5 | the tests of `core/autograd` (14 files): 36 rewrite, 5 resize, 19 split, 11 merge, 6 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M6 | the tests of `core/config` (9 files): 2 rewrite, 43 resize, 9 split, 2 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | failed |
| M7 | the tests of `core/darknet_reference` (1 file): 4 rewrite, 4 resize | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M8 | the tests of `core/data` (9 files): 25 rewrite, 9 resize, 1 split, 22 merge, 4 delete | T1, T2, T3, T4, T5, T6, T7, T8 | failed |
| M9 | the tests of `core/distribution` (13 files): 27 rewrite, 3 resize, 11 split, 25 merge, 3 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M10 | the tests of `core/dtype` (1 file): 1 rewrite, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M11 | the tests of `core/eager` (19 files): 38 rewrite, 8 resize, 12 split, 21 merge, 10 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M12 | the tests of `core/engine_fixture` (3 files): 1 rewrite, 1 resize, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M13 | the tests of `core/execution` (1 file): 2 rewrite, 1 split, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M14 | the tests of `core/failure` (3 files): 14 rewrite, 4 resize, 6 split, 1 merge, 4 delete | T1, T2, T3, T4, T5, T6, T7, T8 | failed |
| M15 | the tests of `core/failure_expectations` (1 file): 1 resize, 2 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M16 | the tests of `core/graph` (33 files): 147 rewrite, 22 resize, 37 split, 2 merge, 12 delete | T1, T2, T3, T4, T5, T6, T7, T8 | failed |
| M17 | the tests of `core/hash` (1 file): 2 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M18 | the tests of `core/log` (2 files): 2 rewrite, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M19 | the tests of `core/model_text` (1 file): 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M20 | the tests of `core/ops` (1 file): 12 rewrite, 2 merge | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M21 | the tests of `core/optim` (16 files): 46 rewrite, 9 resize, 10 split, 8 merge, 3 delete | T1, T2, T3, T4, T5, T6, T7, T8 | failed |
| M22 | the tests of `core/profile` (4 files): 5 rewrite, 1 resize, 1 split, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M23 | the tests of `core/profile_records` (1 file): 2 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M24 | the tests of `core/scratch_files` (2 files): 2 resize, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M25 | the tests of `core/seam` (1 file): 5 merge | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M26 | the tests of `core/seeded_noise` (1 file): 1 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M27 | the tests of `core/shape` (2 files): 4 resize, 3 split, 4 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M28 | the tests of `core/tensor_values` (1 file): 2 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M29 | the tests of `core/version` (1 file): 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M30 | the tests of `language` (46 files): 152 rewrite, 27 resize, 33 split, 19 merge, 15 delete | T1, T2, T3, T4, T5, T6, T7, T8 | failed |
| M31 | the tests of `layers/classifier_objective` (1 file): 1 split, 1 merge | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M32 | the tests of `layers/io` (8 files): 15 rewrite, 19 resize, 4 split, 10 merge, 6 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M33 | the tests of `layers/layer_rows` (2 files): 2 rewrite, 1 split, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M34 | the tests of `layers/layer_traces` (1 file): 2 rewrite, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M35 | the tests of `layers/nn` (39 files): 89 rewrite, 42 resize, 36 split, 138 merge, 30 delete | T1, T2, T3, T4, T5, T6, T7, T8 | failed |
| M36 | the tests of `layers/reference_files` (1 file): 1 rewrite, 1 resize, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M37 | the tests of `layers/train` (39 files): 107 rewrite, 28 resize, 36 split, 20 merge, 20 delete | T1, T2, T3, T4, T5, T6, T7, T8 | failed |
| M38 | the tests of `layers/umbrella` (2 files): 2 rewrite, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M39 | the tests of `package` (2 files): 1 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M40 | the tests of `tools` (6 files): 7 rewrite, 85 resize, 14 split, 12 merge, 6 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M41 | the tests of `vision/blocks` (12 files): 73 rewrite, 7 resize, 20 split, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M42 | the tests of `vision/codecs` (5 files): 21 rewrite, 6 resize, 2 split, 3 merge, 3 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M43 | the tests of `vision/darknet_forward_cases` (1 file): 4 rewrite, 3 split, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M44 | the tests of `vision/darknet_import_cases` (1 file): 3 resize, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M45 | the tests of `vision/darknet_training_cases` (1 file): 1 rewrite, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M46 | the tests of `vision/datasets` (39 files): 96 rewrite, 63 resize, 38 split, 3 merge, 13 delete | T1, T2, T3, T4, T5, T6, T7, T8 | failed |
| M47 | the tests of `vision/detect` (47 files): 226 rewrite, 19 resize, 32 split, 14 merge, 18 delete, 1 characterize | T1, T2, T3, T4, T5, T6, T7, T8 | failed |
| M48 | the tests of `vision/importers` (23 files): 92 rewrite, 23 resize, 14 split, 10 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M49 | the tests of `vision/registration` (3 files): 20 rewrite, 9 resize, 7 split, 4 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M50 | the tests of `vision/segment` (7 files): 8 rewrite, 4 split, 3 merge | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M51 | the tests of `vision/training` (37 files): 88 rewrite, 19 resize, 25 split, 8 merge, 9 delete | T1, T2, T3, T4, T5, T6, T7, T8 | failed |
| M52 | the tests of `vision/umbrella` (2 files): 2 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | done |

## Not done as written

A job is `done` only when each of its records is applied as written or replaced by a recorded ruling.
Each entry below is a record not applied, with the decision of `REPORT.md` that records it, and the
backlog entry where one is open. A mark "(owner)" means the shape of the fix is the owner's to rule.
The audit read the tree against the records by test name, size label, seam and file. A record of
rewrite, split or merge was checked by sampling only, so this list is the floor, not the whole.

**T2.** Two programs still make a runtime per test: the single-machine step of
`tests/applications/programs/source/test_cluster_run.cpp` (decision 23 says it takes the program
backend of `EngineTest`) and the `Medium_ParameterPublish` fixture of
`tests/core/distribution/test_parameter_publish.cpp` (owner: a cluster master needs a runtime of its
own, or the fixture takes one from the program fixture).

**T3.** No adapter for the collators (decision 49; owner: whether the module gains an in-memory
`Collator`) and none for `GraphBuilder`, which is a concrete class (owner: a port first).

**T6.** RW-151 (open, blocked): `tests/README.md` and `cli/tests/conftest.py` still give 60 s, 300 s and
900 s, so the size timeouts are not set. RW-152 (open): the characterized
`DiffersFromTheSecondMaskOfYolov4TinyByOneAnchor` still has no explanation (decision 61). The
`ConfusionMatrix::make(K_MAX_CONFUSION_CLASSES)` case allocates the 4096-class matrix in a `Small_`
test, and no entry holds it (decision 38; owner: move the row to `Medium_` or bound the allocation).

**M3.** `test_find_project_root_raises_outside_checkout` in `cli/tests/test_config.py` waits on a walk
bound in sushicore (decision 16; RW-159 open) (owner: sushicore is outside this repository).

**M6.** `Medium_JsonEvents.ADocumentLongerThanTheBufferArrivesWhole` still writes a file and runs
medium. The record says resize to small, and the buffer-size argument of the ruling of 2026-11-07
(json_events) and `read_json_text_events` (decision 11) exist, so it can read text.

**M8.** Still `Medium_` in `tests/core/data/`: `TheSeedFixesTheSyntheticInputs` (`test_dataset.cpp`),
`AnUnknownOptimizerTypeListsEveryTypeThisBuildHas` (`test_dataset_registry.cpp`),
`ShareBatchCopiesNoElement` (`test_materialised_dataset.cpp`). The rulings of 2026-11-07 (data) give
the seams: a host function for the synthetic class means and labels, an engine-free registry lookup,
an allocation port.

**M14.** `Small_Check.TheUncomposedReportSaysAFatalCheckFailed` in `tests/core/failure/test_check.cpp`
is not deleted.

**M16.** Each stays `Medium_`, or reaches a process-wide instance, where the record says resize to
small. The seams of the rulings of 2026-11-07 (graph) are unbuilt: the in-memory `TensorPool` context
and `CompiledStep` do not exist. The tests: `Medium_Arena` (`AViewAddressesItsOwnElements`,
`TheEmptyPlanPacksNothing`, `APlannedPoolAllocatesOneStorageForAWholeArena`), `Medium_ArenaDeathTest`
(`AViewThatWouldRunPastTheArenaIsRefused`, `AdoptingASecondPlanIsRefused`), `Medium_CompiledStep`
(`ARestoreAfterARunIsRefusedWithTheCount`, `ARangeOutsideTheGraphIsRefused`,
`AGraphWithADroppedRefusalIsNotCompiled`), `Medium_CompiledStepAttribution`
(`TheOptimizerCarriesTheTrailingOrigin`, `AViewLeavesNoOriginOpen`), `Medium_ForwardDeathTest`
(`AnUnboundLeafIsRefusedRatherThanFilledWithGarbage`, `AnUnimplementedOperationNamesItself`), and
`Small_StateID.EachDrawIsAboveTheOneBefore`, which draws from the process-wide counter (an identity
source passed in is the seam).

**M21.** `Medium_AdamWFatal.AMomentPastTheLastParameterIsFatal` and
`Medium_StepScalarsFatal.AWriteOfAnotherLengthIsFatal` have no pure check function beside them, so the
records that ask for the index check and the length check small are not applied (decision 26 built
the other checks).

**M30.** The cases through `CAUSAL_MASK` stay `Medium_` because `Graph::HostEvaluator` has no rule for
it (`tests/language/blocks/test_causal_self_attention.cpp`; decisions 28 and 29). Decision 18 forbids
leaving a record medium for a missing seam, and decision 42 shows the rule is that seam.

**M35.** Decision 43 lists three records not applied as written: the epsilon of the gain-and-bias
layer has a test of its own and is not asserted inside the two value tests;
`TheDiscardingSinkAcceptsAnUpdateAndAddsNoValue` still passes the process-wide `NN::discard_updates()`
where the record asks for a local sink (`tests/layers/nn/test_trace.cpp`); the mul half of
`TheTutorialIsRightThatMulWouldNotBroadcast` left `test_module.cpp` because another test covers it.
The `ModelState` resizes run on a `HostExecutionBackend` and not on the `Detail::describe_parameter`
seam the record names (owner: whether that stands in for the seam).

**M37.** `Medium_Training.ABatchOfAnotherShapeIsRefusedWithBothShapes` stays `Medium_`
(`tests/layers/train/test_trainer/test_convergence.cpp`). The record says small with a header-only
seam; decision 50 keeps it medium.

**M46.** Four MNIST cases still read files through an engine in `Medium_DatasetsMNIST`
(`TheSampleCapLimitsTheBatches`, `ABatchSizeThatIsNotPositiveIsRefused`,
`AnImageFileWithoutTheIDX3HeaderIsInvalid`, `TheFactoryReadsTheImageLayout`). `make_mnist` takes no
`ByteSource`, so the seam the record names is not built (decision 47).
`DestroyingThroughTheBaseRunsTheDerivedDestructor` in `test_collator.cpp` keeps its stand-in
(decision 49; owner: whether the module gains an in-memory `Collator`).

**M47.** `Medium_DetectAnchorsFatal.ScaleRejectsStridesThatDoNotMatchThePoints`
(`tests/vision/detect/free/test_anchors.cpp`) is a death test where the record says resize to small
for a refusal decided on value sizes; no non-fatal check of `scale_points` exists.

**M51.** Decision 55 builds no seam, so these stay `Medium_`; RW-155 is closed and no entry is open
(owner: the host entry view of `NN::ModelState` and the host seam of `Train::TrainingStep` are module
surface): `AnAbsentEntryIsRefusedAndNamed` and `AHeadCountThatDiffersFromTheTaskIsRefused`
(`test_anchor_bias_init.cpp`), `RefusesAPrefixTheStateLacks` (`test_free_bias_init.cpp`),
`ACentreOutsideTheInputIsRefused`, `AnObjectiveNamingAnUnpublishedHeadIsRefused`,
`AHostStageThatReadsAValueTheGraphLacksIsRefused`, `AHostStageBehindAReaderOfItsLeafIsRefused` and
`AStepWithoutAHostStageRefusesHostWork` (`test_anchor_training_step.cpp`),
`ALabelOutsideTheClassesStopsTheFeed` (`test_pixel_cross_entropy_objective.cpp`),
`AMaskOfRankThreeIsCheckedByItsElementCount` (`test_pixel_labels.cpp`). The add for
`fill_pixel_targets` with an INT32 product is not written (owner: it needs a seam or a mask of about
two billion pixels). The `FreeTrainingStep` add sits in `tests/vision/training`, not in
`tests/language/training` as the order says.

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
