# tests/layers/io

The tests of the module `layers/io`, one file per header of `modules/layers/io/include/`. The suite prefix of a case gives its ctest label.

## test_checkpoint.cpp

Suites: `Medium_Checkpoint`, `Small_CheckpointRefusal`.

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

Suites: `Medium_StateCheckpoint`.

The cases save and load checkpoints over a `NN::ModelState`. The model is a batch norm named `bn`,
so the state holds two parameters and two buffers. The cases pin the round trip, the order of the
records in the file and the refusal of an unwritable path.

### SavesAndLoadsParametersAndBuffersBitForBit

A state whose every entry is filled, buffers included, is saved. A second state of the same model,
filled with other values, differs before the load. After the load, the bit patterns of every entry
equal the saved ones.

### TheFileListsEveryEntryInWalkOrder

`list_file` reads every record and its payload back and returns the names and the payload lengths.
The test compares them with the names of the state entries and with the byte length of each
entry's tensor, so the records come in the entry order of the state and each payload has the length
its header declares.

### AnUnwritablePathIsInvalidFile

`save_checkpoint` to a path inside a directory that does not exist is refused as `InvalidFile`.

### Small_CheckpointRefusal

These refusals load a checkpoint from bytes in memory into a state created on a host engine, so
they need no device and no file. `checkpoint_bytes.hpp` writes the bytes with zero payloads,
because each refusal is decided on names and shapes alone.

#### AParameterTheFileDoesNotSupplyIsRefusedAndNamesIt

The file is saved from a shallower model, so it has no `2.weight` or `2.bias`. Loading it into the
deeper model is refused with a message that names `2.weight`, and the load reports
`StatusCode::Incompatible` as a failure.

#### AParameterTheModelDoesNotHaveIsRefusedAndNamesIt

The direction is reversed: the file is saved from the deeper model and loaded into the shallower
one, so the file holds `2.weight`, which the model does not have. The load is refused with a
message that names `2.weight` and says that this model has no such entry.

#### AShapeMismatchIsRefusedWithBothShapes

The bytes are in memory and the state is on a host engine, so the case is small.

The two models have the same parameter names and different widths. An element-count comparison
would catch this pair but would miss a transpose, so the loader compares shapes.

#### ADuplicatedNameIsRefusedAndNamesTheDuplicate

The file is built by hand because `save_checkpoint` cannot produce one: the walk yields each
name once. A counter-based loader would accept this file and leave a parameter reading
uninitialised device memory.

#### ACheckpointWithoutBuffersIsRefusedAndNamesTheRunningMean

A file that holds only the parameters is refused by `load_checkpoint` as `Incompatible`, and the
message contains "running_mean".

## test_import_weights.cpp

Suites: `Small_RenameRule`, `Small_RenameRuleStrip`, `Small_MemoryInput`, `Medium_ImportWeights`,
`Small_ImportWeightsUnmatched`, `Small_ImportWeightsRefusal`.

The four `Small_RenameRule` cases run the prefix rule with no device. They share this file with the
integration cases, which load a checkpoint written under foreign names into a `NN::ModelState`.

### AnEmptyToPrefixStripsTheMatchedPrefix

The rule `model.` to the empty text turns `model.0.weight` into `0.weight` and `model.1.bias` into
`1.bias`: the matched prefix goes and the rest of the name stays. The two names are the rows of
`Small_RenameRuleStrip`.

### LeavesANameNoRuleMatches

A name that no rule's prefix starts is returned unchanged.

### TheFirstMatchingRuleWins

With `model.` to `first.` listed before `model.0` to `second`, `model.0.weight` becomes
`first.0.weight`. The order decides, and the longer prefix does not win.

### ARuleStripsAForeignPrefixAndEveryRecordLands

A checkpoint holds `model.0.weight` and `model.0.bias`, written from a source state. A target state
holds other values. Loading with the rule `model.` to the empty prefix leaves both target tensors
with the bytes of the source.

### ARecordTheStateLacksIsRefusedAndNamed

A file holding one record that matches no entry is refused as `Incompatible`, from bytes in memory.
The rows of `Small_ImportWeightsUnmatched` are the call without a rule, where the message names
`model.0.weight`, and the call under the rule that strips `model.`, where it names both
`model.9.weight` and the renamed `9.weight`.

### AnEntryTheFileLacksIsRefusedAndNamed

The bytes are in memory and the state is on a host engine, so the case is small.

The file holds only `model.0.weight`. The refusal is `Incompatible`, and its message contains
"does not supply" and "'0.bias'".

### AShapeMismatchIsRefusedWithBothShapes

A record shaped [3, 2] carries the six elements of a [2, 3] entry. The refusal is `Incompatible`,
and its message names both shapes. The shapes are compared, not the element counts.

### AMissingFileIsFileNotFound

A path that was never written is refused as `FileNotFound`, and the message contains
"could not open".

### ServesTheBytesOfACheckpointToItsReader

A checkpoint of one record, written to a string stream, opens a `CheckpointReader` through
`IO::MemoryInput`. The reader declares one record named `0.bias`, and its payload is the three
floats that were written, byte for byte. No file and no device are used.

### BytesInMemoryLoadWhatTheFileOfTheSameBytesLoads

The bytes of a two-record checkpoint load into a second state through the span form of
`import_weights`, and both tensors hold the first state's bytes.

## test_run_checkpoint.cpp

Suites: `Medium_RunCheckpoint`, `Medium_RunCheckpointResume`, `Medium_RunCheckpointState`,
`Medium_RunCheckpointProgress`, `Medium_RunCheckpointBadProgress`.

A run is a classifier over 6 features and 4 classes with a batch of 8, its state, an optimizer and a
schedule, built from a seed. The cases save and load checkpoints with run state through
`IO::save_run_checkpoint` and `IO::load_run_checkpoint`, in scratch files that are removed
afterwards. Parameters are compared by their bit patterns.

### AResumedRunMatchesTheUninterruptedOneBitForBit

Run once for each row of `Medium_RunCheckpointResume`: SGD with momentum and AdamW.
SGD with a momentum of 0.9. One run takes 12 updates without interruption. A second takes 6 and is
saved. A third, built from another seed, differs before the load, equals the second after it with a
step count of 6, and after 6 more updates equals the uninterrupted run bit for bit with a step count
of 12. The same file loaded as weights alone leaves the step count at 0, and 6 more updates from
there do not give the uninterrupted run's parameters.


### AFileWithoutRunStateRefusesAResume

A file with model state alone is refused by `load_run_checkpoint` as `Incompatible`, and the message
names the file, version 2, "no run state" and the `weights` entry point.

### AFileWithoutRunStateLoadsAsWeights

The same kind of file loads through `load_checkpoint`, and the parameters match the saved run bit
for bit.

### AVersionOneFileRefusesAResume

A file saved with model state alone and rewritten as version 1, with no run-state count, is refused
by `load_run_checkpoint` as `Incompatible`, naming version 1 and "no run state". That a version 1
file reads as weights is the format's own test, `AVersionOneFileReadsAsModelStateAlone`.

### RunStateOfAnotherOptimizerTypeIsRefused

A file saved from an SGD run is refused for an AdamW run as `Incompatible`. The message contains the
file name, 'sgd' and 'adamw'.

### ARunStateRecordTheOptimizerDisagreesOnIsRefusedByName

Run once for each row of `Medium_RunCheckpointState`. A file written without momentum, loaded into
an SGD that keeps it, is refused for the `.momentum` record it does not supply. A file written
with momentum, loaded into an SGD that keeps none, is refused for the surplus record. Both
messages name the optimizer `sgd`.

### ARunCheckpointHoldsTheSeedAndThePositionOfItsRun

Run once for each row of `Medium_RunCheckpointProgress`. A save that states the progress (seed 9,
6 step runs, epoch 2, batch 3) loads back field for field. A save that states none loads as the
start of a run under the seed 0. The refusal of another seed belongs to
`test_run_progress.cpp`.

### AResumedRunWithDropoutAndAShuffledOrderMatchesBitForBit

One run takes 12 updates. A second takes 6, which ends at epoch 1, batch 2, and is saved. A
third, built from another seed, loads the file, restores the run count of its trainer and takes
6 more updates from the loaded position: its parameters equal the first run's bit for bit, and
its progress ends at 12 step runs, epoch 3, batch 0.

The shuffled dataset is a `Data::InMemoryEpochDataset` whose batch maker reads the batch the epoch permutation names from the synthetic dataset.

### RunStateWithoutTheProgressRecordsIsRefusedOnResume

A run checkpoint cut back to the file without its four progress records is refused on a resume,
naming the file and `run.seed`.

### RunStateWithoutTheProgressRecordsStillLoadsAsWeights

The same file loads through `load_checkpoint`, and the parameters match the saved run bit for bit.

### AProgressRecordOfAnotherTypeOrRangeIsRefused

Run once for each row of `Medium_RunCheckpointBadProgress`. The last record retyped as float64 is
refused as `Incompatible` for `run.batch`. The payload of `run.epoch` made larger than the largest
epoch index is refused as `InvalidFile`.

## test_checkpoint_format.cpp

The checkpoint format's round trip, a version 1 file, the run-state section, and its refusal of each malformed byte pattern.

Suites: `Small_CheckpointFormat`, `Small_CheckpointFormatVersion`, `Small_CheckpointFormatOwner`,
`Medium_CheckpointFormatDeathTest`.

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

### RejectsAForeignMagic

With the magic bytes replaced by "JUNK" the read fails with `InvalidFile`, "SACP" and
"not a SushiAI checkpoint".

### ARejectedVersionNamesTheVersionsRead

Run once for each row of `Small_CheckpointFormatVersion`: version 3 and version 0. Each is refused
as `InvalidFile`, naming the version and "versions 1 to 2".

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

### RoundTripsARunStateSection

In the run-state stream, byte 50 opens the section with the owner length 3 and "sgd", and the
run-state count field holds 2. The reader reports version 2, one model record and two
run-state records. After the model record, `begin_run_state` returns "sgd". The next records
are `step_count`, `Int64` of [1] holding seven, and `0.bias.momentum` holding the two floats
written. `read_count` is 1 and `run_state_read_count` is 2.

### AReaderOfModelStateLeavesTheRunStateUnread

The owner length of the run-state section is set to zero, which makes the section malformed. A
reader of the model records alone still reads them, and only the reader of the run state refuses
the file, with "names no owner".

### RejectsARunStateCountOverTheBound

With the run-state count patched to 16385 the read fails with `InvalidFile`, "16385",
"16384" and "run-state". `CheckpointWriter::create` with 16385 run-state records fails with
`InvalidDescription` and the same three texts.

### AMalformedRunStateOwnerIsRefused

Run once for each row of `Small_CheckpointFormatOwner`: an owner of length zero, one of 257 bytes,
one cut short, and a writer given an empty owner. Each row states its code and its message.

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

Suites: `Small_ImportedModelWriter`, `Medium_ImportedModelWriter`, `Medium_ImportedModelFiles`,
`Medium_ImportedModelWeightFile`.

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

### TheFilesWrittenAreNamedInOrder

Run once for each row of `Medium_ImportedModelFiles`. With tensors the import writes `model.json`,
`weights.sushi` and `train.json`, in that order. Without them it writes `model.json` and
`train.json` and no weight file.

### TheModelFileReadsBack

`Config::load_model` over the written model file returns the input shape that was given, one
layer of type `linear` and a task of type `stub_task`.

### TheWeightFileHoldsTheTensor

Run once for each row of `Medium_ImportedModelWeightFile`, one per tensor. The weight file holds the
record and the bytes of that tensor at its position.

### TheTrainingFileIsRefusedUntilItsDataIsFilled

The composed training file, parsed as a value, is refused for its `data` key. The small case needs
no file.

### TheTrainingFileOnDiskIsRefusedUntilItsDataIsFilled

The training file the import wrote is read from disk by `load_train` and refused for its `data` key.

### AnExistingFileIsRefusedAndLeftAlone

With a `train.json` holding "kept" in the directory, the write under `ExistingFiles::Refuse`
fails with `Incompatible`, "train.json" and "exists already". The file still holds "kept" and
`model.json` does not exist.

### ReplaceOverwritesAnExistingFile

With a `train.json` holding "stale", the write under `ExistingFiles::Replace` is ok and the
file contains "sushiai_train".

### ATensorOfTheWrongSizeIsRefused

`check_imported_tensors` refuses a tensor whose bytes differ from its record, naming the tensor and
both sizes. It touches no file.

### ARefusedImportWritesNoFile

A refused tensor stops `write_imported_model` before the model file exists.

### ATensorNamedTwiceIsRefused

`check_imported_tensors` refuses a name the model states twice. 
With both tensors named `0.weight`, the write fails with `InvalidDescription`, `'0.weight'` and
"twice".

## test_importer_registry.cpp

The importer rows in insertion order, a repeated format, a row's failure passed on and an unknown format listing the rows.

Suites: `Small_ImporterRegistry`.

The rows are `K_COUNTING_IMPORTER_ROW` and `K_REFUSING_IMPORTER_ROW` of `in_memory_importer.hpp`.

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

## test_in_memory_importer.cpp

Suites: `Small_InMemoryImporter`.

Tests the two importer rows of `in_memory_importer.hpp`, which read no file. The counting importer
notes how many files it was given, the refusing importer refuses a call without files, and it
accepts a call that names one.

## test_run_progress.cpp

The default of a run's progress, the names of its four records, and the refusal of a resume under another seed.

Suites: `Small_RunProgress`, `Small_RunProgressName`, `Small_RunProgressSeed`.

Tests `IO::RunProgress` and `IO::check_run_seed` in
`modules/layers/io/include/SushiAI/io/run_progress.hpp`.

### ARecordHasTheNameTheFormatStates

Run once for each row of `Small_RunProgressName`: the constants for the seed, the step runs, the
epoch and the batch equal `run.seed`, `run.step_runs`, `run.epoch` and `run.batch`.

### AResumeUnderTheSavedSeedIsAccepted

`check_run_seed` accepts the seed the checkpoint holds.

### AResumeUnderAnotherSeedIsRefusedWithBothSeeds

Run once for each row of `Small_RunProgressSeed`: another seed, and a default progress that holds
seed 0. Each is refused as `Incompatible`, naming the file and both seeds.

