# tests/layers/train/test_buffer_update_recorder.cpp


Suites: `Integration_BufferUpdateRecorder`.

The recorder blends each buffer's batch statistic into its running buffer, once per replay. The
cases check how many tasks it records, the value one replay leaves behind, the empty recorder,
and the two refusals that binding makes.

## RecordsTwoTasksPerSlot

Two slots record four tasks in all, two for each slot. The test asserts that `slot_count()` is 2
and that `record_into_step` returns 4.

## OneReplayBlendsTheSourceIntoTheTarget

One slot over four float64 elements. After one replay each element equals the
`NN::updated_statistic` of its running value and its batch value, with momentum 0.1 and a source
scale of 2, within 1e-9.

## NoSlotRecordsNothing

A recorder built with no slots records zero tasks, reports zero slots and runs without error.

## BindRefusesATargetTheStateLacks

The state holds one entry, `bn.running_mean`, under identity 1. The update targets identity 7.
Binding is refused as `Incompatible`, and the message contains "no entry".

## BindRefusesADTypeMismatch

The target entry is float32 and the source tensor is float64. Binding is refused as
`Incompatible`, and the message names both dtypes by their `dtype_name`.

Suites: `Integration_EvaluationStep`.

The evaluation step replays a forward-only graph over a `NN::ModelState`. The cases check that it
compiles once, that it reads the weights a training step wrote, that a probe run leaves the
batch norm buffers alone, and the two refusals it makes.

## TheStepCompilesOnce

The step is made from an MLP's evaluation graph. `compile_count()` is 1 after `make()` and still
1 after three runs.


## AnEvaluationStepSeesAWeightTheTrainingStepJustMoved

A training pool and an evaluation pool attach to one state. The evaluation output for fixed
inputs is read before and after one training step on the synthetic dataset. The two byte
sequences differ. The test notes that the default policy starts the classifier at zero.

## AProbeRunLeavesBothRunningVectorsUnchanged

The evaluation graph of a batch norm gives its buffers their leaves, and its probe graph is run
once. The bytes of the running mean and the running variance are the same before and after.

## AnInputOfAnotherShapeIsRefusedWithBothShapes

A step compiled for a [4, 6] input refuses a [2, 6] input. The refusal is `Incompatible`, and its
message names "inputs", "[2, 6]" and "[4, 6]".

## AnUnknownOutputNameListsTheNames

`output("boxes")` is refused as `InvalidDescription`. The message names "boxes" and the single
port name, `NN::K_SINGLE_PORT_NAME`.
