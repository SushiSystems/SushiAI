# tests/core/data

The tests of the module `core/data`, one file per header of `modules/core/data/include/`. The suite prefix of a case gives its ctest label.

## test_dataset_registry.cpp

Suites: `Small_DatasetRegistry`, `Small_RingsOptions`, `Medium_Registries`, `Medium_RingsBand`,
`Medium_OptimizerKeys`.

Tests `Data::DatasetRegistry`, `Data::register_datasets`, `Data::find_dataset_row` and
`Data::read_rings_options`. The file uses two rows: "rings", as `register_datasets` adds it, and
"moons", a type no module registers. The `Small_` cases need no device: the registry, the lookup
of a row and the reader of the options of a rings set take no engine. The `Medium_` cases build
the datasets and optimizers, which allocate tensors.

### AddKeepsInsertionOrder

After `add` of the "rings" row and then the "moons" row, `rows()` has two entries, "rings"
first and "moons" second.

### FindReturnsNullForAnUnknownType

`find("rings")` is null on an empty registry. After the "rings" row is added, `find("coco")` is
null and `find("rings")` returns a row whose type is "rings".

### ATypeAddedTwiceIsRefused

A second `add` of the "rings" row fails with `InvalidDescription`, and the message holds
"'rings'" and "already registered".

### RegisterDatasetsAddsTheRingsRow

After `Data::register_datasets`, `find("rings")` returns a row whose type is "rings".

### AnUnknownDatasetTypeListsEveryTypeThisBuildHas

`find_dataset_row` of "coco" over a registry that holds the "rings" row fails with
`InvalidDescription`; the message contains "coco", "dataset type" and 'rings'.

### AKnownDatasetTypeIsFound

`find_dataset_row` of "rings" returns the row `find` returns.

### TheKeysOfAConfigValueReachTheOptions

`read_rings_options` over a batch size of 16, 2 batches, 3 classes, one distractor, a ring width
of 0.5, a position noise of 0.25 and a seed of 7 returns those options.

### AKeyLeftOutKeepsItsDefault

`read_rings_options` over `{}` returns the defaults of `RingsOptions` for the batch size, the
distractor count and the seed.

### AValueOfTheWrongTypeIsRefused

A class count spelled "three" fails with `InvalidDescription`, and the message names the context
and "classes".

### AnUnknownDatasetKeyIsRejectedByItsOwnFactory

The key "batch_sizes" fails `read_rings_options` with an error whose message contains the context
"data", 'batch_sizes' and 'batch_size'. `make_rings` calls the same reader, so the refusal comes
before any allocation.

Suites: `Medium_Registries`, `Medium_RingsBand`, `Medium_OptimizerKeys`.

Two registries need an Engine to exercise: datasets allocate device tensors, and optimizers
allocate their state buffers. The layer and objective registries are host-lane and live in the
unit tier.

Rings is tested on its defining property, that it is not linearly separable, and not only on
its shape. The dataset exists because a single Linear layer cannot solve it, so a test should
fail if it ever becomes solvable.

The fixture fills a `Data::DatasetRegistry` through `Data::register_datasets`, which adds the
"rings" row the dataset cases name. The optimizer cases call `Optim::make_optimizer` with the
types "adamw" and "lion".

### RingsLabelsEveryBatchBalanced

Labels are assigned round-robin, so every batch is exactly balanced and accuracy needs no
class weighting to read correctly. A batch of 32 samples over 4 classes holds 8 of each.

### RingsPublishesItsLabelsAsOneIndexPerSample

With three classes and a batch of eight, the labels are `{0, 1, 2, 0, 1, 2, 0, 1}`.

### RingsInputsCarryTheBatchAxisBeforeTheSampleShape

The default options add four distractors to the two ring coordinates, so a batch of eight samples
has inputs of rank 2 and shape [8, 6].

### RingsPlacesEachClassInItsOwnRadiusBand

Rows: the classes 0 to 3. The whole problem rests on this property. The class is a function of
x^2 + y^2, which is degree 2 in the inputs and therefore not a half-space. If a change made the
radius bands overlap or the labels track something linear, this test fails, where otherwise the
accuracy would rise in some other test and nobody would see why. A radius is allowed to leave its
band by `K_RADIUS_TOLERANCE`, 2048 unit roundoffs of float, the smallest power of two above 1e-4.

### TheSameSeedGivesTheSameData

The dataset draws from a host RNG and not from the engine's Philox stream, so reproducibility
is a property of the seed alone. The device stream mixes in a monotonic offset counter, which
is why `dataset.hpp` warns about `set_seed`. Two datasets of seed 42 hold the same input bytes.

### AnUnknownOptimizerTypeListsEveryTypeThisBuildHas

The type "lion" raises an error whose message contains "lion", "optimizer type", 'adamw' and 'sgd'.

### AnUnknownOptimizerKeyIsRejectedByItsOwnFactory

Rows: the keys "momentum" and "kernel". "momentum" is SGD's key, not AdamW's. Asking AdamW for it
must fail; it must not be a line that does nothing. `UpdateKernel` selects the spelling a
byte-for-byte regression test compares the fused kernel against. SEPARATE also cannot join the
compiled plan. A config file that could set it would be able to change what a run is, so "kernel"
is not an accepted key.

## test_epoch_permutation.cpp

Suites: `Small_EpochPermutationCover`, `Small_EpochPermutation`.

Tests `Data::EpochPermutation`. An order is read by calling `at` for every position from 0 to
the count. The seed is 42 unless a case says otherwise.

### EverySampleIsReadOnce

Rows: the counts 1, 2, 3, 7, 16, 17, 1000 and 4097 crossed with the epochs 0, 1 and 5. In each
row every value `at` returns is below the count and no value is returned twice.

### AnOrderRepeatsForTheSameSeedAndEpoch

Two reads of the order of seed 42, epoch 0 over 10 samples are equal.

### AnotherEpochGivesAnotherOrder

Over 10 samples, epoch 1 of seed 42 differs from epoch 0 of seed 42.

### AnotherSeedGivesAnotherOrder

Over 10 samples, epoch 0 of seed 7 differs from epoch 0 of seed 42.

### AnOrderIsNotTheOrderOfStorage

Over 10 samples, epoch 0 of seed 42 is not 0, 1, 2, ... 9. These three cases, with the repeat
case above, state what the former pinned arrays showed: an order is a function of the seed and the
epoch, and it moves samples. The arrays themselves are not kept, since they only recorded the
output of the mapping.

## test_target.cpp

Suites: `Small_DataTargetLookup`, `Small_DataTargetAllocation`, `Medium_DataTarget`, `Medium_DataTargetPreparation`, `Medium_DataTargetLabels`.

Tests `Data::find_target`, `Data::allocate_batch`, `Data::prepare_batch`,
`Data::describe_input` and the labels `Data::SyntheticClassification` publishes. The allocation
cases run over a `Data::InMemoryBatchAllocator`, which describes each tensor and holds no storage,
and need no device; the other `Medium_` fixtures need an engine. The two roles of the cases are `labels`, one int32
per sample over 5 classes, and `boxes`, a float32 [3, 4] value per sample.

### ATargetOfEitherKindIsFoundByItsRole

`find_target` over two descriptions and over two `BatchTarget` values returns the address of the
first for `K_ROLE_LABELS` and of the second for `boxes`. It returns null for `mask` and for any
role over an empty list.

### AllocateBatchGivesTheLayoutOfItsDescriptions

Rows, over an `InMemoryBatchAllocator`: 8 samples of a float32 [6] input with the two roles give [8, 6] float32 inputs, `labels` as
an int32 [8] tensor and `boxes` as a float32 [8, 3, 4] tensor, in that order. A [4, 4, 1] sample
with no description gives rank-4 inputs and no target. An int32 [7] input and 3 samples give an
int32 [3, 7] input, and the `boxes` target stays float32.

### PrepareBatchKeepsOrReplacesTheInput

Rows: `prepare_batch` for 3 samples of an int32 [7] input leaves a batch that already holds that
layout at its data pointer. It replaces the input of a batch that held a float32 [7] input, and
the input of a batch a dataset read handed out. In every row the result is an int32 [3, 7] input.

### DescribeInputReportsTheTypeAndShapeADatasetOverrides

`describe_input` of a `MaterialisedDataset` built over an int32 [7] input gives int32 and [7].

### SyntheticClassificationPublishesItsLabelsInRotation

Rows: a synthetic set of two batches of 4 samples, 5 features and 3 classes. The labels of batch 0
are 0, 1, 2, 0 and those of batch 1 are 1, 2, 0, 1.

## test_threaded_loader.cpp

Suites: `Small_ThreadedLoader`, `Small_ThreadedLoaderOrder`, `Medium_ThreadedLoader`,
`Medium_ThreadedLoaderRows`.

Tests `Data::ThreadedLoader`. Most cases wrap a `Data::InMemoryEpochDataset` of 12 batches whose
batch maker builds one target named "<epoch>:<index>" that holds no tensor. The `Small_` cases
build the loader with `ReadExecutor::CallingThread`, so no thread starts and the order of the
reads is fixed: the dataset's journal then shows which index was read, and how far ahead. The
`Medium_` cases use the default executor, which starts worker threads, and need no engine except
`Medium_ThreadedLoaderRows`. A `Probe` records the threads that built a batch and the batches
being built. Reads through other datasets are pinned by the contract suites in `test_dataset.cpp`.

### TheBatchOfEachIndexIsHandedOverAsThatIndex

With a prefetch of 4, reading indices 0 to 11 returns the batch named "0:<index>" each time.

### ACallingThreadLoaderStartsNoThread

With `ReadExecutor::CallingThread`, every batch of indices 0 to 11 is built on the test's own
thread and on no other.

### ReadsNoFurtherAheadThanThePrefetchCount

With a prefetch of 3, after the read of index `i` the highest index the dataset's journal holds is
`min(i + 3, 11)`.

### EachBatchIsReadOnceInIndexOrder

After reading indices 0 to 11 the journal holds 0 to 11, each once, in order.

### AFailedReadIsReportedWithTheStatusOfItsIndex

With a dataset whose batch 5 fails, reading index 5 after the indices 0 to 4 fails with
`InvalidFile`, "batch 5" and "unreadable".

### ReadsAfterAFailedReadStillSucceed

After the failed read of index 5, the indices 6 to 11 return their batches.

### AnIndexOutOfOrderIsRead

Rows: the indices 7; 7, 8, 2; 7, 8, 2, 3, 11; and 11, 0, each read in that order. Each read
returns the batch of its index.

### BeginEpochReachesTheDataset

`Data::begin_epoch(*loader, 3)` sets the dataset's epoch to 3.

### BeginEpochDropsWhatWasReadAhead

After reads of indices 0 and 1 and `Data::begin_epoch(*loader, 3)`, reading indices 0 to 11 returns
the batches named "3:<index>".

### TwoLoadersOfDifferentPrefetchDeliverTheSameEpochsInTheSameOrder

A loader of a prefetch of 1 and a loader of a prefetch of 6 read epochs 0 and 1 over indices 0
to 11. Both return "<epoch>:<index>" for every batch.

### MakeRefusesNoDataset

`ThreadedLoader::make` fails with `InvalidDescription` for a null dataset ("no dataset").

### MakeRefusesNoWorker

`ThreadedLoader::make` fails with `InvalidDescription` for 0 workers ("0 threads").

### MakeRefusesNoPrefetch

`ThreadedLoader::make` fails with `InvalidDescription` for a prefetch of 0 ("0 batches ahead").

### TheBatchesAreReadOnWorkerThreadsOnly

After the reads of indices 0 to 11 with 3 workers, at least one and at most three threads built a
batch, and the test's own thread is not among them.

### TwoLoadersOfDifferentWidthDeliverTheSameEpochsInTheSameOrder

A loader of 1 worker and a prefetch of 1 and a loader of 4 workers and a prefetch of 6 read
epochs 0 and 1 over indices 0 to 11. Both return "<epoch>:<index>" for every batch.

### TheDestructorJoinsWithBatchesReadAhead

A loader of 3 workers and a prefetch of 4 is destroyed after a read of index 0. No batch is being
built when the destructor returns.

### TheLoaderStatesWhatItsInnerDatasetStates

Over a loader of three batches of 4 rings samples, the batch size, the batch count, the name, the
sample type and shape and the number of targets equal those of the inner dataset. This case uses
the engine fixture.
