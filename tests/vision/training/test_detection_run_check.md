# tests/vision/training/test_detection_run_check.cpp


Suites: `Small_DetectionRunCheck` and `Small_DetectionRunCheckEpochs`.

Tests `check_detection_run` and `make_detection_run_plan` of `detection_run_check.hpp`, the check
that both detection training loops run before the first update. The dataset is a stub that
publishes its facts and cannot be read: 3 batches of 2 images over the three truth roles for two
slots and three classes. The subject is a run named "A test run" whose batch origin is "heads
were declared for". No case needs a device.

The loops' own files keep one case each that shows a refused run changes nothing (no update, no
run counted). The refusals themselves are decided here.

### AcceptsADatasetTheSubjectCanRead

A dataset of the subject's batch and roles passes with a default plan.

### ThePlanCopiesTheFieldsOfTheOptions

A loop's options with 4 epochs, a step schedule, first epoch 2 and first batch 1 give a plan that
holds the same four fields, the schedule by address.

### EpochsTheRunCannotUseAreRefusedNamingTheRun

Rows: 0 epochs ("A test run needs at least one epoch", "got 0") and a first epoch of -1 ("A test
run starts at an epoch index", "got -1"). Each fails with `InvalidDescription`.

### ADatasetWithoutBatchesIsRefused

A dataset of no batch fails with `InvalidDescription` naming `'stub'` and "no batches".

### AFirstBatchTheDatasetLacksIsRefused

A first batch of 3 over 3 batches fails with `Incompatible` naming "batch 3", `'stub'` and "3
batches".

### ABatchOfAnotherSizeNamesBothSizesAndTheOrigin

A dataset of 3 images per batch fails with `Incompatible` naming `'stub'`, "batch of 3" and "heads
were declared for 2".

### AnInvalidScheduleIsRefused

A step schedule of two steps and one scale fails with `InvalidDescription` naming "2 steps" and "1
scales".

### ADatasetWithoutTheBoxesRoleIsRefused

A dataset whose only role is the labels fails with `InvalidDescription` naming `'stub'` and
`'boxes'`.

### RolesOverAnotherClassCountAreRefused

A subject of four classes over a dataset of three fails with `Incompatible` naming `'stub'`,
`'labels'` and "was compiled for".
