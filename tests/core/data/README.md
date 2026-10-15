# tests/core/data

The tests of the module `core/data`, one file per header of `modules/core/data/include/`. The suite prefix of a case gives its ctest label.

## test_dataset_registry.cpp

Suites: `Medium_DatasetRegistry`.

Tests `Data::DatasetRegistry` and `Data::register_datasets`. The file uses two rows: "rings",
as `register_datasets` adds it, and "moons", a type no module registers.

### AddKeepsInsertionOrder

After `add` of the "rings" row and then the "moons" row, `rows()` has two entries, "rings"
first and "moons" second.

### FindReturnsNullForAnUnknownType

`find("rings")` is null on an empty registry. After the "rings" row is added, `find("coco")` is
null and `find("rings")` returns a row whose type is "rings".

### ATypeAddedTwiceIsRefused

A second `add` of the "rings" row fails with `InvalidDescription`, and the message holds
"'rings'" and "already registered".

### RegisterDatasetsAddsRingsOnly

After `Data::register_datasets`, `rows()` has one entry of type "rings" and `find("mnist")` is
null.

## test_epoch_permutation.cpp

Suites: `Small_EpochPermutation`.

Tests `Data::EpochPermutation`. An order is read by calling `at` for every position from 0 to
the count. The seed is 42 unless a case says otherwise.

### EverySampleIsReadOnce

For the counts 1, 2, 3, 7, 16, 17, 1000 and 4097 and the epochs 0, 1 and 5, every value `at`
returns is below the count and no value is returned twice.

### AnOrderIsAFunctionOfTheSeedAndTheEpoch

Over 10 samples the order of seed 42, epoch 0 is 4, 9, 5, 2, 1, 3, 8, 7, 6, 0, and a second
read gives the same. Epoch 1 gives 2, 5, 9, 1, 4, 3, 6, 7, 8, 0, and seed 7 at epoch 0 gives
5, 4, 1, 2, 3, 9, 0, 6, 7, 8.

### CountIsTheCountItWasBuiltOver

`count()` of a permutation built over 17 samples is 17.

## test_target.cpp

Suites: `Small_DataTargetLookup`, `Medium_DataTarget`.

Tests `Data::find_target`, `Data::allocate_batch`, `Data::prepare_batch`,
`Data::describe_input` and the labels `Data::SyntheticClassification` publishes. The
`Medium_DataTarget` fixture needs an engine. The two roles of the cases are `labels`, one int32
per sample over 5 classes, and `boxes`, a float32 [3, 4] value per sample.

### ADescriptionIsFoundByItsRole

`find_target` over the two descriptions returns the address of the first for `K_ROLE_LABELS`
and of the second for `boxes`. It returns null for `mask` and for any role over an empty list.

### ABatchTargetIsFoundByItsRole

`find_target` over two `BatchTarget` values returns the address of each by its role and null
for `mask`.

### AllocateBatchYieldsOneTensorPerDescriptionInOrder

`allocate_batch` for 8 samples of a float32 [6] input gives [8, 6] float32 inputs and two
targets: `labels` as an int32 [8] tensor and `boxes` as a float32 [8, 3, 4] tensor.

### AllocateBatchWithNoDescriptionYieldsTheInputsAlone

With no description and a [4, 4, 1] sample, the inputs have rank 4 and the batch has no target.

### AllocateBatchGivesTheInputItsStatedElementType

For an int32 [7] input and 3 samples the inputs are an int32 [3, 7] tensor, and the `boxes`
target is float32.

### PrepareBatchKeepsABatchThatHoldsTheLayoutAlone

A second `prepare_batch` with the same batch size, input and roles leaves the data pointer of
the inputs unchanged and their type int32.

### PrepareBatchReplacesAnInputOfAnotherElementType

After `prepare_batch` with a float32 [7] input, a second call with an int32 [7] input leaves
int32 inputs of rank 2 whose second extent is 7.

### ADatasetStatesTheElementTypeOfItsInput

`SyntheticClassification` reports float32 for `sample_dtype()`, and `describe_input` of it
gives float32 and a sample shape of its feature count. `describe_input` of a
`MaterialisedDataset` built over an int32 [7] input gives int32 and [7].

### SyntheticClassificationPublishesItsLabelsInRotation

A synthetic set of two batches of 4 samples, 5 features and 3 classes publishes one target:
`K_ROLE_LABELS`, an empty sample shape, int32, 3 classes. Its sample shape is [5]. The labels
of batch 0 are 0, 1, 2, 0 and those of batch 1 are 1, 2, 0, 1.

## test_threaded_loader.cpp

Suites: `Medium_ThreadedLoader`, `Small_Epoch`.

Tests `Data::ThreadedLoader` and `Data::begin_epoch`. The loader wraps a local
`RecordingDataset` of 12 batches. Each batch it reads carries one target whose role is
"<epoch>:<index>" and holds no tensor. A `Probe` records the threads that called `read`, how
far ahead of the caller a read started and the epoch last begun. The cases start worker
threads and need no engine.

### HandsTheBatchesOverInIndexOrder

With 3 workers and a prefetch of 4, reading indices 0 to 11 returns the batch named "0:<index>"
each time. At least one and at most three threads called `read`, and the test's own thread is
not among them.

### ReadsNoFurtherAheadThanThePrefetchCount

With 4 workers and a prefetch of 3, no read started more than 3 indices ahead of the index the
test had asked for.

### AFailedReadIsTheStatusOfItsIndexAlone

With a dataset whose batch 5 fails, reading index 5 fails with `InvalidFile`, "batch 5" and
"unreadable". Every other index from 0 to 11 returns its batch.

### AnIndexOutOfOrderIsRead

Reading the indices 7, 8, 2, 3, 11, 0 in that order returns the batch of each index.

### BeginEpochReachesTheDatasetAndDropsWhatWasReadAhead

After reads of indices 0 and 1, `Data::begin_epoch(*loader, 3)` sets the dataset's epoch to 3.
Reading indices 0 to 11 then returns the batches named "3:<index>".

### TheDestructorJoinsWithBatchesReadAhead

A loader of 3 workers and a prefetch of 4 is destroyed after `begin_epoch` and before any read
by the test. The case ends with `SUCCEED()` and asserts nothing else after the loader is made.

### StatesWhatItsDatasetStates

The loader reports the batch size 1, 12 batches, the name "recording", the sample type float32
and no target.

### MakeRefusesNoDatasetAndAnOptionBelowOne

`ThreadedLoader::make` fails with `InvalidDescription` for a null dataset ("no dataset"), for 0
workers ("0 threads") and for a prefetch of 0 ("0 batches ahead").

### ADatasetThatIsNotEpochAwareIsLeftAlone

`Data::begin_epoch` on a dataset that does not implement `EpochAware` returns, and the
dataset's batch count is still 1.

Suites: `Medium_Registries`.

Two registries need an Engine to exercise: datasets allocate device tensors, and optimizers
allocate their state buffers. The layer and objective registries are host-lane and live in the
unit tier.

Rings is tested on its defining property, that it is not linearly separable, and not only on
its shape. The dataset exists because a single Linear layer cannot solve it, so a test should
fail if it ever becomes solvable.

The fixture fills a `Data::DatasetRegistry` through `Data::register_datasets`, which adds the
"rings" row the dataset cases name. The optimizer cases call `Optim::make_optimizer` with the types
"adamw" and "sgd".

### RingsIsBalancedAndCarriesItsDistractors

Labels are assigned round-robin, so every batch is exactly balanced and accuracy needs no
class weighting to read correctly.

### RingsPublishesItsLabelsAsOneIndexPerSample

With three classes and a batch of eight, the dataset describes one target: the labels role, one
int32 per sample, over 3 classes. The batch holds one target tensor of rank 1 and length 8 whose
labels are `{0, 1, 2, 0, 1, 2, 0, 1}`.

### RingsReportItsSampleShape

The default options add four distractors, so a sample has shape [6]. A batch of eight samples is
rank 2 with shape [8, 6].

### RingsPlacesEachClassInItsOwnRadiusBand

The whole problem rests on this property. The class is a function of x^2 + y^2, which is
degree 2 in the inputs and therefore not a half-space. If a change made the radius bands
overlap or the labels track something linear, this test fails, where otherwise the accuracy
would rise in some other test and nobody would see why.

### TheDatasetRegistryBuildsRingsFromAConfigValue

`make_dataset` for "rings" with a batch size of 16, 2 batches, 3 classes and one distractor returns
a dataset with those extents and the sample shape `[3]`.

### TheSameSeedGivesTheSameData

The dataset draws from a host RNG and not from the engine's Philox stream, so reproducibility
is a property of the seed alone. The device stream mixes in a monotonic offset counter, which
is why `dataset.hpp` warns about `set_seed`.

### AnUnknownDatasetTypeListsEveryTypeThisBuildHas

The type "coco" is refused as `InvalidDescription`. The message contains "coco", "dataset type" and
'rings'.

### AnUnknownDatasetKeyIsRejectedByItsOwnFactory

The key "batch_sizes" raises an error whose message contains the context "data", 'batch_sizes' and
'batch_size'.

### ANonPositiveRingsExtentIsRefused

A batch size of 0 is refused as `InvalidDescription` with "positive" in the message.

### TheOptimizerRegistryBuildsBothRulesFromAConfigValue

The test passes one real parameter and gradient pair, so the optimizers allocate their state
against something and not against an empty list.

### AnUnknownOptimizerTypeListsEveryTypeThisBuildHas

The type "lion" raises an error whose message contains "lion", "optimizer type", 'adamw' and 'sgd'.

### AnUnknownOptimizerKeyIsRejectedByItsOwnFactory

"momentum" is SGD's key, not AdamW's. Asking AdamW for it must fail; it must not be a line
that does nothing.

### AConfigurationCannotReachTheReferenceUpdateKernel

`UpdateKernel` selects the spelling a byte-for-byte regression test compares the fused kernel
against. SEPARATE also cannot join the compiled plan. A config file that could set it would
be able to change what a run is, so "kernel" is not an accepted key.
