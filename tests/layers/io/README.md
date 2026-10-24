# tests/layers/io

The tests of the module `layers/io`, one file per header of `modules/layers/io/include/`. The suite prefix of a case gives its ctest label.

## test_checkpoint.cpp

Suites: `Medium_Checkpoint`.

The file covers the device half of checkpointing: saving and loading records against the tensors
of a `NN::ModelState`. The byte format's malformed-file surface needs no device and lives in
`tests/layers/io/test_checkpoint_format.cpp`.

The last equivalence test is the point of the file. A `Sequential` and a `LayerStack` built
from the equivalent `model.json` are two spellings of one model. The proof that they are one
model is a checkpoint written by either loading into the other. A separate mechanism would
leave its own correctness as the open question.

### Materialised

`traced_state` traces the model over a [4, 6] input and creates its `NN::ModelState` from the
traced graph. `NN::initialize` then fills that state with a policy that leaves the final layer
random, so every parameter carries real numbers and not the zeros of the default policy. The
class owns the state, because every tensor a test reads has to outlive the comparison.

### SavesAndLoadsEveryParameterBitForBit

A model materialised from seed 11 is saved. A second one from seed 22 differs before the load, and
after `load_checkpoint` its parameters have the saved bit patterns.

### ASequentialCheckpointLoadsIntoAnEquivalentLayerStackAndBack

Two spellings of one model prove they describe the same thing by exchanging a checkpoint. The
compile-time and run-time paths then need no separate mechanism to keep them in step, and they
cannot drift without this test failing.

The names are asserted equal first. The checkpoint is matched by name, so a naming difference
would otherwise surface as "does not supply" and hide the clearer statement that the two
models disagree about what they hold.

The return trip, `LayerStack` to `Sequential`, loads into a third model. That makes it a real
load and not a comparison against values that were already there.

### AParameterTheFileDoesNotSupplyIsRefusedAndNamesIt

The file is saved from a shallower model, so it has no `2.weight` or `2.bias`. Loading it into the
deeper model is refused with a message that names `2.weight`, and the load reports
`StatusCode::Incompatible` as a failure.

### AParameterTheModelDoesNotHaveIsRefusedAndNamesIt

The direction is reversed: the file is saved from the deeper model and loaded into the shallower
one, so the file holds `2.weight`, which the model does not have. The load is refused with a
message that names `2.weight` and says that this model has no such entry.

### AShapeMismatchIsRefusedWithBothShapes

The two models have the same parameter names and different widths. An element-count comparison
would catch this pair but would miss a transpose, so the loader compares shapes.

### ADuplicatedNameIsRefusedAndNamesTheDuplicate

The file is built by hand because `save_checkpoint` cannot produce one: the walk yields each
name once. A counter-based loader would accept this file and leave a parameter reading
uninitialised device memory.

Suites: `Medium_StateCheckpoint`.

The cases save and load checkpoints over a `NN::ModelState`. The model is a batch norm named `bn`,
so the state holds two parameters and two buffers. The cases pin the round trip, the order of the
records in the file, the refusal of a file without buffers, and the refusal of an unwritable path.

### SavesAndLoadsParametersAndBuffersBitForBit

A state whose every entry is filled, buffers included, is saved. A second state of the same model,
filled with other values, differs before the load. After the load, the bit patterns of every entry
equal the saved ones.

### TheFileListsEveryEntryInWalkOrder

The file holds one record for each state entry, and the record names come in the state's entry
order. The test reads each record and its payload back, and each payload is read at the length its
header declares.

### ACheckpointWithoutBuffersIsRefusedAndNamesTheRunningMean

A file that holds only the parameters is refused by `load_checkpoint` as `Incompatible`, and the
message contains "running_mean".

### AnUnwritablePathIsInvalidFile

`save_checkpoint` to a path inside a directory that does not exist is refused as `InvalidFile`.

## test_import_weights.cpp

Suites: `Small_RenameRule`, `Medium_ImportWeights`.

The four `Small_RenameRule` cases run the prefix rule with no device. They share this file with the
integration cases, which load a checkpoint written under foreign names into a `NN::ModelState`.

### RewritesAMatchingPrefix

The rule `model.` with the empty prefix turns `model.0.weight` into `0.weight`.

### LeavesANameNoRuleMatches

A name that no rule's prefix starts is returned unchanged.

### TheFirstMatchingRuleWins

With `model.` to `first.` listed before `model.0` to `second`, `model.0.weight` becomes
`first.0.weight`. The order decides, and the longer prefix does not win.

### AnEmptyToPrefixStripsThePrefix

`model.1.bias` becomes `1.bias`.

### ARuleStripsAForeignPrefixAndEveryRecordLands

A checkpoint holds `model.0.weight` and `model.0.bias`, written from a source state. A target state
holds other values. Loading with the rule `model.` to the empty prefix leaves both target tensors
with the bytes of the source.

### WithoutTheRuleTheFirstRecordIsRefused

The same file, loaded with no rules, is refused as `Incompatible`. The message contains
`'model.0.weight'`.

### ARecordTheStateLacksIsRefusedUnderBothNames

A record named `model.9.weight` has no entry once renamed. The refusal is `Incompatible`, and its
message contains "holds 'model.9.weight'" and "renames to '9.weight'".

### AnEntryTheFileLacksIsRefusedAndNamed

The file holds only `model.0.weight`. The refusal is `Incompatible`, and its message contains
"does not supply" and "'0.bias'".

### AShapeMismatchIsRefusedWithBothShapes

A record shaped [3, 2] carries the six elements of a [2, 3] entry. The refusal is `Incompatible`,
and its message names both shapes. The shapes are compared, not the element counts.

### AMissingFileIsFileNotFound

A path that was never written is refused as `FileNotFound`, and the message contains
"could not open".

## test_run_checkpoint.cpp

Suites: `Medium_RunCheckpoint`.

A run is a classifier over 6 features and 4 classes with a batch of 8, its state, an optimizer and a
schedule, built from a seed. The cases save and load checkpoints with run state through
`IO::save_run_checkpoint` and `IO::load_run_checkpoint`, in scratch files that are removed
afterwards. Parameters are compared by their bit patterns.

### AResumedSGDRunMatchesTheUninterruptedOneBitForBit

SGD with a momentum of 0.9. One run takes 12 updates without interruption. A second takes 6 and is
saved. A third, built from another seed, differs before the load, equals the second after it with a
step count of 6, and after 6 more updates equals the uninterrupted run bit for bit with a step count
of 12. The same file loaded as weights alone leaves the step count at 0, and 6 more updates from
there do not give the uninterrupted run's parameters.

### AResumedAdamWRunMatchesTheUninterruptedOneBitForBit

The same sequence with AdamW.

### AFileWithoutRunStateIsWeightsAndNoResume

A file written by `save_checkpoint` is refused by `load_run_checkpoint` as `Incompatible`; the
message contains the file name, "version 2", "no run state" and 'weights'. `load_checkpoint` reads
it, and the parameters equal the saved ones.

### AVersionOneFileStillLoadsAsWeights

The run-state count is cut out of the header and the version byte set to 1. `load_checkpoint` reads
the file and the parameters equal the saved ones. `load_run_checkpoint` refuses it as `Incompatible`
with "version 1" and "no run state".

### AFutureVersionIsRefusedByBothLoaders

With the version byte set to 3, both loaders refuse the file as `InvalidFile`. Each message contains
"version 3" and "versions 1 to 2".

### RunStateOfAnotherOptimizerTypeIsRefused

A file saved from an SGD run is refused for an AdamW run as `Incompatible`. The message contains the
file name, 'sgd' and 'adamw'.

### AMissingOrSurplusRunStateRecordIsRefusedByName

A file saved by SGD without momentum is refused by SGD with momentum; the message contains "does not
supply" and a `.momentum` record. The file of the run with momentum is refused by the run without;
the message contains "does not keep". Both refusals are `Incompatible` and name 'sgd'.

The dropout cases train a classifier with one dropout of rate 0.25, traced with the seed
20261011, under SGD with a momentum of 0.9, over a dataset that reads the four batches in an
order `Data::EpochPermutation` draws for each epoch. Their loop keeps an `IO::RunProgress`.

### ARunCheckpointHoldsTheSeedAndThePositionOfItsRun

A progress of seed 9, 6 step runs, epoch 2 and batch 3 is saved and read back field for field.
`IO::check_run_seed` accepts 9 and refuses 10 with both. A save that states no progress reads
back as zeros.

### AResumedRunWithDropoutAndAShuffledOrderMatchesBitForBit

One run takes 12 updates. A second takes 6, which ends at epoch 1, batch 2, and is saved. A
third, built from another seed, loads the file, restores the run count of its trainer and takes
6 more updates from the loaded position: its parameters equal the first run's bit for bit, and
its progress ends at 12 step runs, epoch 3, batch 0.

### AResumeThatDropsTheRunCountOrThePositionDiverges

The same file. A run that does not restore the run count, and a run that restarts at epoch 0,
batch 0, each end with other parameters than the uninterrupted run.

### RunStateWithoutTheProgressRecordsIsRefusedOnResume

The last 151 bytes of a saved file are cut and its run-state count lowered by 4. The resume is
`Incompatible`, names `run.seed` and `weights`, and the file still loads as weights.

### AProgressRecordOfAnotherTypeOrRangeIsRefused

`run.batch` retyped to float64 is `Incompatible`; a `run.epoch` with its top bit set is
`InvalidFile`.

## test_checkpoint_format.cpp

The checkpoint format's round trip, a version 1 file, the run-state section, and its refusal of each malformed byte pattern.

Suites: `Small_CheckpointFormat`, `Small_CheckpointFormatDeathTest`.

Tests the checkpoint format's round trip and its refusal of each malformed byte pattern.

This file holds the checkpoint format's whole malformed-file surface in the device-free tier.
That is the point of splitting `modules/layers/io/include/SushiAI/io/checkpoint_format.hpp` out of
`modules/layers/io/include/SushiAI/io/checkpoint.hpp`. Every case is a byte pattern, so none of them needs an
Engine, a TensorPool or a device to run.

Each corruption is written as a patch at a known offset into a good file, not as a hand-written
byte array. A case therefore says what is wrong with the file and does not restate the whole
format.

### Field offsets

The `OFFSET_` constants are byte offsets into `sample_stream()`, whose first record is
"0.weight", FLOAT32, [2, 3]. They are stated once so that a case names the field it breaks.

### RoundTripsNamesShapesAndPayloads

The sample stream opens as two records. The first is `0.weight`, `Float32`, [2, 3], 24
payload bytes, and its payload reads back as the six floats written. The second is `0.bias`
of [2] with the two floats written. `read_count` is then 2.

### DistinguishesTwoShapesWithTheSameElementCount

This case covers the defect the format exists to close: an element count alone lets a [2, 3]
weight load into a [3, 2] one. The shape is recorded, so the two records are not
interchangeable.

### RejectsAForeignMagic

With the magic bytes replaced by "JUNK" the read fails with `InvalidFile`, "SACP" and
"not a SushiAI checkpoint".

### RejectsAFutureVersion

With the version patched to 3 the read fails with `InvalidFile`, "version 3" and
"versions 1 to 2".

### RejectsACountOverTheBound

With the record count patched to 4097 the read fails with `InvalidFile`, "4097" and "4096".

### RejectsATruncatedHeader

The first six bytes of the stream fail with `InvalidFile`, "ended while reading" and
"version".

### RejectsATruncatedPayload

The stream without its last four bytes fails with `InvalidFile`, "ended while reading" and
"0.bias".

### RejectsANameLengthOverTheBound

With the first name length patched to 257 the read fails with `InvalidFile`, "257" and "256".

### RejectsARankOverTheBound

With the first rank patched to 7 the read fails with `InvalidFile`, "0.weight", "rank 7" and
"6".

### RejectsANonPositiveDim

With the first extent patched to 0 the read fails with `InvalidFile`, "0.weight" and
"must be positive".

### RejectsAnUnknownDType

With the first element type patched to 99 the read fails with `InvalidFile`, "0.weight", "99"
and "does not know".

### RejectsACountTypeTheFormatDoesNotCarry

With the first element type patched to the `UInt32` value the read fails with `InvalidFile`,
"0.weight", "uint32" and "does not carry".

### RefusesToWriteACountTypeTheFormatDoesNotCarry

Writing a `UInt32` record fails with `InvalidDescription`, "counts", "uint32" and "does not
carry".

### RejectsMoreRecordsThanDeclared

A writer created for 2 records writes two. A third `write` is fatal with "declared 2" and
"'c'".

### RejectsFewerRecordsThanDeclared

A writer created for 2 records writes one. `finish` is fatal with "declared 2" and
"1 were written".

### RejectsATensorCountOverTheBoundAtConstruction

`CheckpointWriter::create` for 4097 records fails with `InvalidDescription`, "4097" and
"4096".

### ReadingPastTheDeclaredCountIsAnError

After both records of the sample stream are read, a third `next` is fatal with "declared 2".

### WritesTheVersionTwoHeaderByteByByte

The sample stream opens with the sixteen bytes 'S', 'A', 'C', 'P', then 2, 2 and 0 as
little-endian 32-bit values. Its size is those sixteen bytes plus 60 bytes for the weight
record and 34 for the bias record.

### AVersionOneFileReadsAsModelStateAlone

A version 1 stream opens with `version` 1, two records and a run-state count of 0, and both
payloads read back as written.

### RejectsAVersionBelowTheFirst

With the version patched to 0 the read fails with `InvalidFile`, "version 0" and
"versions 1 to 2".

### RoundTripsARunStateSection

In the run-state stream, byte 50 opens the section with the owner length 3 and "sgd", and the
run-state count field holds 2. The reader reports version 2, one model record and two
run-state records. After the model record, `begin_run_state` returns "sgd". The next records
are `step_count`, `Int64` of [1] holding seven, and `0.bias.momentum` holding the two floats
written. `read_count` is 1 and `run_state_read_count` is 2.

### AReaderOfModelStateLeavesTheRunStateUnread

Reading the model records of the run-state stream alone succeeds.

### RejectsARunStateCountOverTheBound

With the run-state count patched to 16385 the read fails with `InvalidFile`, "16385",
"16384" and "run-state". `CheckpointWriter::create` with 16385 run-state records fails with
`InvalidDescription` and the same three texts.

### RejectsAMalformedRunStateOwner

The read fails with `InvalidFile` for an owner length of 0 ("names no owner"), an owner
length of 257 ("run-state section", "257", "256") and a stream cut inside the owner
("ended while reading", "run-state owner"). `begin_run_state("")` on a writer fails with
`InvalidDescription`, "owner" and "0 bytes".

### RejectsATruncatedRunStatePayload

The run-state stream without its last three bytes fails with `InvalidFile`,
"ended while reading" and "0.bias.momentum".

### TheRunStateOpensAfterTheModelRecords

On a writer that declared one model record and wrote none, `begin_run_state` is fatal with
"declared 1 records" and "0 were written". On a reader that has read none it is fatal with
"declared 1 records" and "0 were read".

### ARunStateThatWasNotDeclaredCannotOpen

On a writer created without run-state records, `begin_run_state` is fatal with
"declared no run-state record". On a reader of a version 1 stream, after both records are
read, it is fatal with "declares no run-state record".

### TheRunStateHoldsExactlyTheRecordsDeclared

On a writer that declared one run-state record, `finish` right after `begin_run_state` is
fatal with "declared 1 run-state records" and "0 were written". After one record is written,
a second `write` is fatal with "declared 1 run-state records" and "'b'".

## test_imported_model_writer.cpp

The three files of an imported model read back, the key order of the training file with `data` null, existing files refused or replaced, and a bad tensor writing nothing.

Suites: `Small_ImportedModelWriter`.

Tests `IO::compose_training_file` and `IO::write_imported_model`. The imported model of every
case has one `linear` row over the input shape [4], a task of type `stub_task`, the float32
tensors `0.weight` of shape [2, 4] and `0.bias` of shape [2], a recipe with an `objective` and
an `optimizer` section, and one note. The cases that write files use a directory under the
system's temporary folder and remove it before and after.

### ComposesTheTrainingFileAroundTheRecipe

The composed file has the keys `sushiai_train`, `model`, `weights`, `data`, `checkpoint`,
`objective`, `optimizer`, in that order. `sushiai_train` is `Config::K_TRAIN_VERSION`, `model`
is "model.json", `weights` is "weights.sushi", `checkpoint` is "trained.sushi" and the
optimizer's `type` is "sgd".

### LeavesTheDataBlockNull

The composed file has a `data` key whose value is null.

### StatesNoWeightsKeyWithoutTensors

With no tensors the composed file has no `weights` key.

### ARecipeThatStatesAWriterKeyIsRefused

A recipe that states `model` fails with `InvalidDescription` and a message containing `'model'`
and "training recipe".

### ARecipeThatIsNoObjectIsRefused

A recipe that is the integer 3 fails with `InvalidDescription` and "must be an object".

### WritesTheThreeFiles

`write_imported_model` with `ExistingFiles::Refuse` returns three paths, `model.json`,
`weights.sushi` and `train.json` under the directory, and each is a regular file.

### TheModelFileReadsBack

`Config::load_model` over the written model file returns the input shape that was given, one
layer of type `linear` and a task of type `stub_task`.

### TheWeightFileHoldsEveryTensor

`IO::CheckpointReader` over the written weight file counts two records. Each record has the
name, the type and the shape of the tensor at its position, and its payload equals that
tensor's bytes.

### TheTrainingFileIsRefusedUntilItsDataIsFilled

`Config::load_train` over the written training file raises an error containing `'data'`, "must
be an object" and "null".

### WritesNoWeightFileWithoutTensors

With no tensors the write returns two paths and `weights.sushi` does not exist.

### AnExistingFileIsRefusedAndLeftAlone

With a `train.json` holding "kept" in the directory, the write under `ExistingFiles::Refuse`
fails with `Incompatible`, "train.json" and "exists already". The file still holds "kept" and
`model.json` does not exist.

### ReplaceOverwritesAnExistingFile

With a `train.json` holding "stale", the write under `ExistingFiles::Replace` is ok and the
file contains "sushiai_train".

### ATensorOfTheWrongSizeWritesNothing

With one byte removed from the payload of `0.bias`, the write fails with `InvalidDescription`,
`'0.bias'`, "7 bytes" and "needs 8", and `model.json` does not exist.

### ATensorNamedTwiceIsRefused

With both tensors named `0.weight`, the write fails with `InvalidDescription`, `'0.weight'` and
"twice".

## test_importer_registry.cpp

The importer rows in insertion order, a repeated format, a row's failure passed on and an unknown format listing the rows.

Suites: `Small_ImporterRegistry`.

Tests `IO::ImporterRegistry` and `IO::import_model`. The file registers two stub rows: `first`,
whose importer returns a model with the number of files it was given as its one note, and
`second`, whose importer fails with `InvalidFile` when given no file.

### AddKeepsInsertionOrder

After `add` of `first` and then `second`, `rows()` holds two rows in that order.

### FindReturnsNullForAnUnknownFormat

`find("first")` is null on an empty registry. After `first` is added, `find("onnx")` is null
and `find("first")` returns the row whose format is "first".

### AFormatAddedTwiceIsRefused

A second `add` of `first` fails with `InvalidDescription` and a message containing `'first'`
and "already registered".

### ImportModelCallsTheRowOfTheFormat

`import_model` for `first` with two files returns a model whose one note is "2".

### ImportModelReturnsTheFailureOfTheRow

`import_model` for `second` with no file fails with `InvalidFile` and "the stub reads no file".

### AnUnknownFormatListsTheFormatsTheRegistryHolds

`import_model` for `onnx` fails with `InvalidDescription` and a message containing `'onnx'` and
"'first', 'second'".

## test_run_progress.cpp

The default of a run's progress, the names of its four records, and the refusal of a resume under another seed.

Suites: `Small_RunProgress`.

Tests `IO::RunProgress` and `IO::check_run_seed` in
`modules/layers/io/include/SushiAI/io/run_progress.hpp`.

### ADefaultProgressIsTheStartOfARun

A default `RunProgress` has the seed 0, 0 step runs, the epoch 0 and the batch 0.

### TheRecordsHaveDistinctNames

`K_RUN_SEED_STATE` is "run.seed", `K_RUN_STEP_RUNS_STATE` is "run.step_runs",
`K_RUN_EPOCH_STATE` is "run.epoch" and `K_RUN_BATCH_STATE` is "run.batch".

### AResumeUnderAnotherSeedIsRefusedWithBothSeeds

For a progress saved under the seed 20261011, `check_run_seed` passes that seed. The seed 7
fails with `Incompatible`, "'last.sushi'", "seed 20261011" and "seed 7". A default progress
checked against 20261011 fails with `Incompatible`, "seed 0" and "seed 20261011".
