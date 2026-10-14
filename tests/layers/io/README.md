# tests/layers/io

The tests of the module `layers/io`, one file per header of `modules/layers/io/include/`. The suite prefix of a case gives its ctest label.

## test_checkpoint.cpp

Suites: `Integration_Checkpoint`.

The file covers the device half of checkpointing: saving and loading records against the tensors
of a `NN::ModelState`. The byte format's malformed-file surface needs no device and lives in
`modules/layers/io/tests/test_checkpoint_format.cpp`.

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

Suites: `Integration_StateCheckpoint`.

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

Suites: `Unit_RenameRule`, `Integration_ImportWeights`.

The four `Unit_RenameRule` cases run the prefix rule with no device. They share this file with the
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

Suites: `Integration_RunCheckpoint`.

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
