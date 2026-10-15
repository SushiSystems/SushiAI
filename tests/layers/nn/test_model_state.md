# tests/layers/nn/test_model_state.cpp


Suites: `Medium_ModelState`, `Medium_ModelStateDeathTest`.

`NN::ModelState` lists a model's parameters and buffers in walk order, finds them by name or by
identity, refuses two entries with one name, stops on an entry with no identity, and attaches its
storage to a graph and a pool. The cases pin each of those.

## CreateListsParametersInWalkOrder

A traced `Sequential` of Linear, GELU and Linear lists four parameters in order: `0.weight` [8, 6],
`0.bias` [8], `2.weight` [4, 8] and `2.bias` [4].

## CreateListsBuffersAfterATrainingTrace

A `ModelGraph` holding a batch norm named `bn`, traced in training mode, lists `bn.weight` and
`bn.bias` as parameters. It then lists `bn.running_mean`, with constant initial value 0, and
`bn.running_var`, with constant initial value 1, as buffers.

## AnUntracedModelGivesAnEmptyState

A model that was never traced gives a valid state with no entries.

## APrefixIsJoinedToEveryName

A prefix of "model" passed to `create()` makes the first entry `model.0.weight`.

## TwoEntriesWithOneNameAreRefused

`from_entries` with two entries named "shared" is refused as `InvalidDescription`, and the message
names "shared".

## AnEntryWithNoIdentityIsFatal

`from_entries` with an entry whose identity is 0 stops the process, as the death test expects. The
message contains "'orphan'", the entry's name, and "no identity".

## TensorByNameAndByStateAreTheSameStorage

`tensor()` called with the name and with the identity of one entry returns the same tensor object.

## AnUnknownNameIsRefusedAndNamed

`tensor("missing.weight")` is refused as `InvalidDescription`. The message contains the name and
"4 entries".

## AttachBindsEveryStateLeafToTheStatesStorage

`attach()` records four state leaves on the graph. For each leaf, the pool tensor's data pointer is
the same as the state tensor's.

## AttachRefusesAGraphOfAnotherModel

A graph traced from a second instance of the same architecture is refused by `attach()` as
`Incompatible`.

## TwoPoolsAttachedToOneStateShareStorage

Two pools attached to one state read the same numbers. A value written through a leaf of the first
pool is read back through the second.
